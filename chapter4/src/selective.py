"""
Selective history: история по релевантности, а не только по свежести.

Глава 3 обрезала историю по позиции — оставляла последние сообщения, пока
они влезали в бюджет. Это работает ровно до первого случая, когда важное
было сказано десять реплик назад: «мой сервер называется prod-01» уезжает
из окна, и агент отвечает так, будто этого не было.

Тот же векторный поиск, что и по документам, решает и эту задачу: старые
сообщения — такой же корпус, а текущая реплика пользователя — запрос.
В окно попадают последние сообщения (свежесть всё-таки важна: разговор
должен читаться связно) плюс те старые, что относятся к делу.

Цена честная и её видно:
  * каждое сообщение приходится один раз посчитать моделью эмбеддингов —
    но только один: дальше работает кэш из embeddings.py;
  * порядок сообщений остаётся хронологическим, но в истории появляются
    дыры — модель видит куски разговора, а не сплошную ленту;
  * если Ollama не отвечает или модель эмбеддингов не скачана, отбор
    молча деградирует до обрезки по свежести из Главы 3. Агент, который
    перестаёт разговаривать из-за недоступного индекса, хуже агента,
    который помнит чуть меньше.
"""

import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter3.src.context import (
    Conversation,
    drop_orphan_observations,
    estimate_messages_tokens,
    estimate_tokens,
    is_observation,
    trim_by_tokens,
)

from .embeddings import EmbeddingError, cosine_similarity, embed_documents, embed_query

# Сколько последних сообщений остаются в окне всегда, без всякого отбора.
# Четыре — это примерно две пары «реплика — ответ»: меньше, и обрывается
# то, о чём говорят прямо сейчас.
KEEP_RECENT = 4

# Порог для старых сообщений. Ниже него сообщение не берётся, даже если
# бюджет позволяет: место в контексте лучше отдать никому, чем случайному
# «спасибо, помогло» из середины разговора.
RELEVANCE_MIN_SCORE = 0.5


class SelectiveConversation(Conversation):
    """Conversation из Главы 3, в котором окно истории собирается по смыслу.

    Отличие ровно одно — build_messages(). Всё остальное (сжатие в резюме,
    core-память, теги данных, порядок блоков) наследуется как есть.
    """

    def __init__(
        self,
        *args: Any,
        keep_recent: int = KEEP_RECENT,
        min_score: float = RELEVANCE_MIN_SCORE,
        enabled: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.keep_recent = keep_recent
        self.min_score = min_score
        self.enabled = enabled
        # Последняя причина отката на обрезку по свежести. Не для красоты:
        # без неё «агент почему-то забыл» невозможно отличить от «эмбеддинги
        # не считаются» — оба выглядят одинаково.
        self.last_fallback: str = ""

    def select_history(self) -> list[dict[str, Any]]:
        """Отбирает сообщения истории под бюджет: свежие + релевантные старые."""
        if not self.history:
            return []

        budget = self.max_history_tokens

        # Всё влезает — отбирать нечего, это самый частый случай.
        if estimate_messages_tokens(self.history) <= budget:
            return list(self.history)

        if not self.enabled:
            return trim_by_tokens(self.history, budget)

        recent = self.history[-self.keep_recent:] if self.keep_recent else []
        older = self.history[: len(self.history) - len(recent)]

        recent_cost = estimate_messages_tokens(recent)
        if not older or recent_cost >= budget:
            # Свежие сообщения уже съели бюджет: отбирать старые некуда,
            # работает обычная обрезка Главы 3.
            self.last_fallback = "свежие сообщения заняли весь бюджет"
            return trim_by_tokens(self.history, budget)

        query = self._current_query()
        if not query:
            self.last_fallback = "нет реплики пользователя для запроса"
            return trim_by_tokens(self.history, budget)

        try:
            query_vector = embed_query(query)
            older_vectors = embed_documents([self._text(msg) for msg in older])
        except EmbeddingError as e:
            # Именно здесь агент остаётся работоспособным без эмбеддингов.
            self.last_fallback = f"эмбеддинги недоступны ({e})"
            print(f"⚠️ Отбор по релевантности недоступен: {e}. Обрезаю историю по свежести.")
            return trim_by_tokens(self.history, budget)

        scored = [
            (index, cosine_similarity(query_vector, vector))
            for index, vector in enumerate(older_vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        chosen: set[int] = set()
        used = recent_cost
        for index, score in scored:
            if score < self.min_score:
                break
            candidate = {index}
            # Observation без своего вызова модель читает как результат
            # действия, которого не было (см. drop_orphan_observations
            # в Главе 3). Поэтому вызов подтягивается вместе с ответом.
            if is_observation(older[index]) and index > 0 and index - 1 not in chosen:
                candidate.add(index - 1)

            cost = sum(estimate_tokens(self._text(older[i])) for i in candidate - chosen)
            if used + cost > budget:
                continue
            chosen |= candidate
            used += cost

        self.last_fallback = ""
        # Хронологический порядок восстанавливается: отбор менял приоритет,
        # а не время. Диалог, перемешанный по релевантности, модель читает
        # как набор случайных фраз.
        selected = [older[i] for i in sorted(chosen)]
        return drop_orphan_observations(selected + recent)

    def build_messages(self, reminder: str | None = None) -> list[dict[str, Any]]:
        """Собирает сообщения так же, как Глава 3, но окно истории — отобранное."""
        # Родительский метод собирает промпт, core-память и резюме; повторять
        # эту сборку здесь означало бы завести вторую копию порядка блоков.
        # Поэтому зовём его с пустой историей и подставляем свою.
        full_history = self.history
        try:
            self.history = []
            messages = super().build_messages(reminder=None)
        finally:
            self.history = full_history

        messages.extend(self.select_history())

        if reminder:
            messages.append({"role": "system", "content": reminder})

        return messages

    # ------------------------------------------------------------ мелочи

    @staticmethod
    def _text(message: dict[str, Any]) -> str:
        return str(message.get("content") or "")

    def _current_query(self) -> str:
        """Последняя реплика пользователя — она и есть поисковый запрос.

        Именно реплика, а не результат инструмента: Observation описывает
        ответ мира, а нас интересует, о чём спросил человек.
        """
        for message in reversed(self.history):
            if message.get("role") == "user" and not is_observation(message):
                return self._text(message)
        return ""
