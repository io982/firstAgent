"""
Гибридный поиск и порог «в коде этого нет» (пункты 6.2 и 6.4).

Здесь сходится всё: векторный индекс Главы 5, лексический индекс этой главы
и слияние их выдач. Плюс то, ради чего глава затевалась, — ответ «этого
в проекте нет», которого ни у Главы 4, ни у Главы 5 не было.

**Почему порог получается у BM25 и не получался у близости.** Косинусная
близость — величина относительная: она говорит, что этот фрагмент похож
больше остальных, и не говорит, похож ли он вообще. У BM25 в основании
лежит другое: слово запроса либо встречается в корпусе, либо нет. Если
не встречается ни одно — списка вхождений нет, складывать нечего, результат
пуст. Не «плохие результаты», а ничего.

На практике вопрос почти никогда не промахивается целиком: в «как настроить
кубернетес кластер» слова «настроить» и «кластер» в проекте не встречаются,
а вот в «какой сегодня курс доллара» слова «сегодня» и «курс» встречаются
(в коде есть инструмент текущего времени). Поэтому смотрим не на факт
совпадения, а на вес лучшего фрагмента — и это число на нашем корпусе
разделяет два вида вопросов с зазором (таблица в тексте главы).

**Порог считается по ИСХОДНОМУ вопросу, а не по переписанному.** Переписывание
Главы 5 придумывает правдоподобные английские имена, и на вопросе про борщ
модель выдаст что-нибудь вроде `cook recipe soup` — слова, которых в проекте
тоже нет, но проверять надо не их. Исходный вопрос — то, что человек
действительно спросил.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from chapter4.src.knowledge import IndexReport
from chapter4.src.vectorstore import Hit
from chapter5.src.codebase import CodeIndex, demote_tests, get_code_index

from .bm25 import BM25Index
from .fusion import fuse
from .lexical import tokenize_query

# ====================================================================
# НАСТРОЙКИ
# ====================================================================

# Сколько кандидатов берём у каждого поиска ДО слияния. Больше, чем нужно
# в ответе: смысл слияния в том, чтобы фрагмент, оказавшийся у одного
# поиска пятнадцатым, а у другого вторым, вышел наверх. При выдаче
# по пять кандидатов такому фрагменту неоткуда взяться.
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20

# Сколько фрагментов уезжает в контекст. Столько же, сколько в Главах 4 и 5.
TOP_K = 5

# Способы поиска. Ручка нужна замерам главы: все числа в тексте сняты
# переключением одного этого аргумента на одном и том же индексе.
MODES = ("hybrid", "vector", "bm25")

# Каким искать по умолчанию — и это тот случай, когда замер НЕ дал
# однозначного ответа. На двенадцати вопросах главы гибрид дал 7 попаданий
# при MRR 0.40, чистый BM25 — 8 при MRR 0.42: разница в один вопрос из
# двенадцати, то есть ни о чём. А вот от чистых векторов (5 из 12,
# MRR 0.33) оторвались оба. Оставлен гибрид — он тема главы, и на нём
# фрагмент, найденный обоими поисками, выходит вперёд. Переключается:
#   PowerShell:   $env:AGENT_SEARCH_MODE = "bm25"
#   Linux/macOS:  export AGENT_SEARCH_MODE=bm25
DEFAULT_MODE = os.environ.get("AGENT_SEARCH_MODE", "hybrid")

# Порог «в проекте этого нет»: вес лучшего лексического совпадения.
#
# Значение подобрано замером на корпусе этого проекта (1064 фрагмента):
# у двенадцати вопросов, ответ на которые в коде есть, лучший вес лежит
# между 9.24 и 15.73; у двенадцати посторонних — между 0.00 и 7.21.
# Зазор ровно два пункта, порог поставлен посередине. На этом наборе
# отказано на 12 посторонних из 12 и пропущены все 12 настоящих.
# Таблица целиком печатается замером test_absence_threshold.
#
# Число привязано к корпусу: IDF растёт примерно как логарифм числа
# фрагментов, поэтому на корпусе вдесятеро больше веса будут выше,
# и порог придётся пересчитать тем же замером. Отключается нулём:
#   PowerShell:   $env:AGENT_NO_ANSWER = "0"
#   Linux/macOS:  export AGENT_NO_ANSWER=0
NO_ANSWER_BM25 = float(os.environ.get("AGENT_NO_ANSWER", "8.0"))


@dataclass
class Signal:
    """Что лексический поиск говорит о вопросе ещё до выдачи.

    Отдельным объектом, а не одним числом, потому что `missing` попадает
    в ответ пользователю: «в проекте нет упоминаний: кубернетес, кластер»
    — это ответ, а «ничего не найдено» — отписка.
    """

    best: float = 0.0
    support: float = 0.0
    tokens: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def absent(self) -> bool:
        """Похоже ли, что ответа в проекте нет. Порог 0 выключает проверку."""
        return NO_ANSWER_BM25 > 0 and self.best < NO_ANSWER_BM25

    def render(self) -> str:
        """Строка для отладочной печати агента."""
        missed = f", не встречаются: {', '.join(self.missing)}" if self.missing else ""
        return f"лексический вес {self.best:.1f}, слов найдено {self.support:.0%}{missed}"


# ====================================================================
# ГИБРИДНЫЙ ИНДЕКС
# ====================================================================

class HybridIndex:
    """Векторный индекс Главы 5 и лексический индекс этой главы рядом.

    Второй собирается из тех же самых фрагментов, что и первый, — не из
    файлов заново. Это не мелочь: id фрагмента считается по его тексту,
    и пока оба индекса наполняются из одного вызова collect(), слиянию
    есть по чему сопоставлять выдачи. Разойдись они в нарезке — и RRF
    складывал бы места разных фрагментов с одинаковым видом.
    """

    def __init__(self, code_index: CodeIndex | None = None, bm25: BM25Index | None = None):
        self.code = code_index if code_index is not None else get_code_index()
        self.lexical = bm25 if bm25 is not None else BM25Index()

    # ------------------------------------------------------------ сборка

    def sync_lexical(self, path: Path | str | None = None) -> int:
        """Пересобирает лексический индекс с нуля. Возвращает число фрагментов.

        Именно с нуля, а не досборкой: IDF считается по всему корпусу, и
        удалённый файл меняет вес каждого слова, которое в нём было. Это
        не проблема, потому что сборка не требует ни модели, ни сети —
        на этом проекте она занимает доли секунды, и число печатает
        замер главы.
        """
        chunks, _ = self.code.collect(path)
        self.lexical.clear()
        return self.lexical.add(
            ids=[chunk.id for chunk in chunks],
            texts=[chunk.text for chunk in chunks],
            metadatas=[chunk.to_metadata() for chunk in chunks],
        )

    def build(self, path: Path | str | None = None, force: bool = False) -> IndexReport:
        """Сверяет векторный индекс и пересобирает лексический."""
        report = self.code.index(path, force=force)
        self.sync_lexical(path)
        return report

    # ------------------------------------------------------------ порог

    def lexical_signal(self, query: str) -> Signal:
        """Что известно о вопросе до того, как показана выдача."""
        tokens = tokenize_query(query)
        if not tokens or not self.lexical.count():
            return Signal(tokens=tokens, missing=tokens)

        missing = [token for token in tokens if not self.lexical.document_frequency(token)]
        hits = self.lexical.search(query, top_k=1)
        return Signal(
            best=hits[0].score if hits else 0.0,
            support=1 - len(missing) / len(tokens),
            tokens=tokens,
            missing=missing,
        )

    def looks_absent(self, query: str) -> bool:
        """Похоже ли, что ответа на этот вопрос в проекте нет.

        Осторожность намеренно однобокая. Ложное «не знаю» на настоящем
        вопросе — потерянный ответ; ложное «знаю» на постороннем возвращает
        нас ровно к поведению Главы 5, то есть не хуже, чем было. Поэтому
        порог поставлен ближе к посторонним вопросам, чем к настоящим.
        """
        return self.lexical_signal(query).absent

    # ------------------------------------------------------------ поиск

    def candidates(self, query: str, mode: str | None = None) -> tuple[list[Hit], list[Hit]]:
        """Две выдачи до слияния: векторная и лексическая.

        Отдельным методом, потому что замеры главы смотрят именно на них:
        сколько нашёл каждый поиск по отдельности и что из этого попало
        в общий список.
        """
        mode = mode or DEFAULT_MODE
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим поиска: {mode}. Доступны: {', '.join(MODES)}")

        vector: list[Hit] = []
        lexical: list[Hit] = []

        if mode in ("hybrid", "vector") and self.code.store.count():
            # score_gap=1.0 отключает отсечение по отставанию от лучшего:
            # фрагмент, отставший по близости, может выиграть по слову,
            # и выбрасывать его ДО слияния значит слиянию мешать.
            vector = self.code.search(query, top_k=VECTOR_CANDIDATES, score_gap=1.0)

        if mode in ("hybrid", "bm25"):
            # Понижение тестов — из Главы 5, и лексическому поиску оно нужнее.
            # Замер: на запрос `is_safe_query` первые три места занимают
            # тесты, где это имя встречается чаще, чем в самой реализации.
            lexical = demote_tests(self.lexical.search(query, top_k=BM25_CANDIDATES), query)

        return vector, lexical

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        mode: str | None = None,
        method: str = "rrf",
    ) -> list[Hit]:
        """Ищет фрагменты кода обоими способами и сливает выдачи."""
        if not query or not query.strip():
            return []

        mode = mode or DEFAULT_MODE
        vector, lexical = self.candidates(query, mode=mode)
        if mode == "vector":
            return vector[:top_k]
        if mode == "bm25":
            return lexical[:top_k]

        # Отсечения по отставанию от лучшего здесь нет, в отличие от Главы 5.
        # После слияния score — это сумма обратных мест, величина порядка
        # сотых, и «отставание на 0.05» в ней не значит ничего.
        return fuse([vector, lexical], method=method)[:top_k]

    def retrieve(
        self,
        query: str,
        budget_tokens: int,
        top_k: int = TOP_K,
        mode: str | None = None,
    ) -> str:
        """Поиск и сборка под бюджет одним вызовом.

        Сборка — из Главы 5 целиком: шапка с адресом `файл:строки`,
        ограждение по языку, обрезка по строкам. Гибрид меняет то, ЧТО
        попадает в блок, и не меняет того, как блок выглядит.
        """
        return self.code.build_context(self.search(query, top_k=top_k, mode=mode), budget_tokens)

    # ------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        """Что лежит в обоих индексах разом."""
        stats = dict(self.code.stats())
        stats["lexical"] = self.lexical.stats()
        return stats


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_hybrid_index: HybridIndex | None = None


def get_hybrid_index() -> HybridIndex:
    """Общий гибридный индекс (singleton, как индекс кода в Главе 5).

    Лексическая половина собирается при первом обращении: она не требует
    ни модели, ни сети, поэтому отдельной команды на это заводить незачем.
    """
    global _hybrid_index
    if _hybrid_index is None:
        _hybrid_index = HybridIndex()
        _hybrid_index.sync_lexical()
    return _hybrid_index


def set_hybrid_index(index: HybridIndex | None) -> None:
    """Подменяет общий индекс. Нужно тестам, чтобы не трогать настоящий."""
    global _hybrid_index
    _hybrid_index = index
