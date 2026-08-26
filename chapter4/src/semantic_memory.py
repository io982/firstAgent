"""
Семантический поиск по фактам из Главы 3 (пункт 4.5 ROADMAP).

Глава 3 закончилась честно названным недостатком: `recall` находит факт
только по ТОЧНОМУ ключу. Факт сохранён под ключом `user_name`, пользователь
спрашивает «как меня зовут» — модель обязана угадать ключ. На двух фактах
угадывает, на двадцати начинает перебирать, на сотне сдаётся и вызывает
`list_memories`, который упирается в потолок вывода.

Тот же векторный поиск, что и по документам, снимает это ограничение:
факт индексируется строкой «ключ: значение», а запрос ищется по смыслу.

Хранилище фактов при этом остаётся ОДНО — LongTermMemory из Главы 3.
Здесь только индекс поверх него, и он подтягивается автоматически: факт
мог быть записан инструментом `remember` минуту назад, и заставлять
пользователя вручную переиндексировать память было бы издевательством.

⚠️ ГЛАВНОЕ ОГРАНИЧЕНИЕ, И ОНО ИЗМЕРЕНО. Факт индексируется строкой
«ключ: значение», и от ЯЗЫКА КЛЮЧА зависит всё. На семи фактах и пяти
вопросах по-русски:

    ключи по-английски (user_name, dog_name, city)  — 0 попаданий из 5
    те же ключи по-русски (имя пользователя, город) — 4 попадания из 5

«Как меня зовут» с английскими ключами находит `city: Казань` и
`coffee: без сахара`, но не `user_name: Владимир`. Причина не в поиске:
nomic-embed-text заметно хуже сопоставляет тексты на разных языках, а
строка `user_name: Владимир` — это два слова, в которых почти нет смысла,
за который можно зацепиться.

Отсюда правило, которое Глава 4 добавляет в системный промпт: ключи памяти
пишутся словами и на языке разговора. Это не косметика — это разница между
работающим поиском и его имитацией.

Границу видно и здесь: даже с русскими ключами «как меня зовут» находит
«кличка собаки: Рекс» — для модели эмбеддингов оба про «как зовут». Имя
и почту спасает не поиск по смыслу, а нормализация ключей из Главы 3
(`имя`, `name`, `user` — это всё `user_name`), и достаются они обычным
`recall`. recall_like нужен для всего остального: фактов, синонимы которых
заранее в словарь не впишешь.
"""

import hashlib
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter3.src.memory import LongTermMemory, get_memory

from .embeddings import embed_documents, embed_query
from .vectorstore import Hit, MemoryVectorStore, VectorStore

# Индекс фактов лежит отдельно от индекса документов: это разные корпуса
# с разным временем жизни. Документы переиндексируются редко и целиком,
# факты меняются посреди разговора по одному.
DEFAULT_FACTS_INDEX = Path(__file__).parent.parent / "index" / "facts.json"

# Три кандидата. Строки «ключ: значение» короткие, и три штуки стоят
# в контексте дешевле одного абзаца документа, но выбирать из пяти
# маленькой модели уже тяжело: она начинает склеивать факты между собой.
FACTS_TOP_K = 3

# Слабый фильтр, а не решение о релевантности — по той же причине, что
# и MIN_SCORE в knowledge.py: близость не умеет отвечать «такого факта нет».
# На замере верные попадания давали 0.62-0.71, промахи — 0.52-0.63,
# то есть диапазоны перекрываются. 0.5 отсекает только совсем далёкое;
# окончательный выбор всё равно за моделью, и инструмент прямо говорит ей,
# что это КАНДИДАТЫ.
FACTS_MIN_SCORE = 0.5


def fact_id(key: str, value: str) -> str:
    """id факта = хэш ключа и значения.

    Значение входит в хэш намеренно: изменилось значение — изменился id,
    старая запись становится осиротевшей и удаляется при следующей сверке.
    Иначе в индексе жили бы два разных ответа под одним ключом.
    """
    return hashlib.sha1(f"{key}\x00{value}".encode()).hexdigest()[:16]


class SemanticMemory:
    """Векторный индекс поверх LongTermMemory."""

    def __init__(
        self,
        memory: LongTermMemory | None = None,
        store: VectorStore | None = None,
    ):
        self.memory = memory if memory is not None else get_memory()
        self.store = store if store is not None else MemoryVectorStore(DEFAULT_FACTS_INDEX)

    # ------------------------------------------------------------ сверка

    def sync(self) -> tuple[int, int]:
        """Приводит индекс в соответствие с фактами. Возвращает (добавлено, удалено).

        Дешёвая операция, когда ничего не изменилось: сравниваются id, и
        запрос к модели эмбеддингов уходит только за новыми фактами.
        Поэтому её не жалко звать перед каждым поиском.
        """
        facts = {str(key): str(value) for key, value in self.memory.items().items()}
        fresh = {fact_id(key, value): (key, value) for key, value in facts.items()}

        known = self.store.entries()
        stale = [doc_id for doc_id in known if doc_id not in fresh]
        removed = self.store.delete(stale) if stale else 0

        pending = [(doc_id, pair) for doc_id, pair in fresh.items() if doc_id not in known]
        if not pending:
            return 0, removed

        texts = [f"{key}: {value}" for _, (key, value) in pending]
        added = self.store.add(
            ids=[doc_id for doc_id, _ in pending],
            texts=texts,
            embeddings=embed_documents(texts),
            metadatas=[{"key": key} for _, (key, _) in pending],
        )
        return added, removed

    # ------------------------------------------------------------ поиск

    def search(
        self,
        query: str,
        top_k: int = FACTS_TOP_K,
        min_score: float = FACTS_MIN_SCORE,
    ) -> list[Hit]:
        if not query or not query.strip():
            return []

        self.sync()
        if self.store.count() == 0:
            return []

        hits = self.store.search(embed_query(query), top_k=top_k)
        return [hit for hit in hits if hit.score >= min_score]

    def recall_like(self, query: str) -> str:
        """Текст ответа для агента: похожие факты или честное «не нашёл»."""
        hits = self.search(query)
        if not hits:
            return (
                f"❌ Похожих фактов в памяти нет (искал: «{query}»). "
                f"Не выдумывай ответ — скажи пользователю, что не знаешь."
            )

        lines = [
            "🧠 Кандидаты из памяти (это ПОХОЖИЕ факты, а не готовый ответ — "
            "выбери подходящий, а если ни один не отвечает на вопрос, так и скажи):"
        ]
        for hit in hits:
            lines.append(f"  - {hit.text} (близость {hit.score:.2f})")
        return "\n".join(lines)


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory


def set_semantic_memory(memory: SemanticMemory | None) -> None:
    """Подменяет общий индекс фактов. Нужно тестам."""
    global _semantic_memory
    _semantic_memory = memory
