"""
Индекс кода: свой корпус, свой бюджет (пункт 5.4).

Устроен как KnowledgeBase Главы 4 и переиспользует её эмбеддинги и её
хранилище — но живёт в отдельной коллекции. Причина та же, по которой
в Главе 4 разъехались документы и факты: в одной коллекции короткая функция
на пять строк конкурировала бы с абзацем документации, а потолок выдачи
у них разный.

Три вещи, которых в Главе 4 не было:

  * **эмбеддится не то, что возвращается.** В модель уходит карточка плюс
    код (см. cards.py), в базе лежит код, пользователю показывается код.
    Векторная база это позволяет: вектор и документ у неё — разные поля;
  * **у фрагмента есть адрес.** `chapter1/agent.py:82-91` едет в шапке
    фрагмента, потому что ответ про код без адреса нельзя проверить;
  * **корпус меняется каждый день.** Документы правят раз в неделю, код —
    каждую минуту, поэтому цена повторной индексации здесь не мелочь,
    а то, из-за чего сверку либо делают при каждом запуске, либо не делают
    вовсе.
"""

import os
import time
from pathlib import Path

from chapter3.src.context import estimate_tokens
from chapter4.src.embeddings import embed_documents, embed_query, model_slug
from chapter4.src.knowledge import IndexReport
from chapter4.src.vectorstore import (
    CHROMA_PERSIST_DIR,
    DEFAULT_BACKEND,
    ChromaVectorStore,
    Hit,
    MemoryVectorStore,
    VectorStore,
)

from .cards import embedding_text
from .codechunks import CodeChunk, chunk_source
from .languages import is_test_source, iter_sources
from .repomap import DEFAULT_ROOT

# ====================================================================
# НАСТРОЙКИ
# ====================================================================

# Коллекция Главы 5 в той же базе Chroma. Имена коллекций там плоские,
# поэтому префикс главы обязателен: иначе индекс кода и корпус документов
# Главы 4 оказались бы в одной куче.
CODE_COLLECTION = "chapter5_code"

# Куда MemoryVectorStore кладёт свой JSON. Папка в .gitignore: индекс —
# производная от кода, его не жалко удалить и пересобрать.
INDEX_DIR = Path(__file__).parent.parent / "index"

# Индексировать ли документацию (.md) вместе с кодом. ПО УМОЛЧАНИЮ НЕТ —
# и это результат замера, а не экономия места.
#
# Первый индекс этого репозитория собирался со всей документацией: 1462
# фрагмента, из них 673 (46%) — разделы markdown. Вопрос «где вычисляется
# арифметическое выражение» на таком индексе даёт первую шестёрку из одних
# README: 0.734 — раздел «ВНИМАНИЕ» корневого README, дальше пять разделов
# текстов глав, и ни одного фрагмента кода. Причина ровно та, с которой
# началась глава: вопрос по-русски ближе к русской прозе, чем к английскому
# коду, — и проза о коде выигрывает у самого кода на его же вопросах.
#
# Тексты глав при этом никуда не деваются: проза — корпус Главы 4, и
# маршрутизация агента выбирает между корпусами (см. route в agent.py).
# Включить документацию в индекс кода можно для сравнения:
#   PowerShell:   $env:AGENT_CODE_DOCS = "1"
#   Linux/macOS:  export AGENT_CODE_DOCS=1
INDEX_DOCS = os.environ.get("AGENT_CODE_DOCS", "0") != "0"

# Сколько фрагментов достаём. Как и в Главе 4, с запасом: разброс близости
# внутри выдачи — сотые доли, и «тот самый» фрагмент регулярно оказывается
# третьим.
TOP_K = 5

# Насколько фрагмент может отставать от лучшего, чтобы попасть в выдачу.
SCORE_GAP = 0.05

# Абсолютный порог. По умолчанию выключен — по той же причине, что и
# в Главе 4: близость ранжирует, но не умеет отвечать «этого здесь нет».
MIN_SCORE = 0.0

# Место под служебную строку «Найдено фрагментов: N из M» и пустую строку
# после неё — считается по самой строке, а не выбирается на глаз.
HEAD_COST = estimate_tokens("Найдено фрагментов: 99 из 99.") + 1

# Отметка об обрезке. Обрезка проговаривается вслух: молча укороченный
# код модель считает кодом целиком и делает вывод по половине функции.
CUT_MARK = "# […фрагмент обрезан по бюджету контекста]"

# Короче этого обрезать бессмысленно: три строки заголовка функции
# не отвечают ни на что, а место занимают.
MIN_USEFUL_LINES = 3


def code_collection() -> str:
    """Имя коллекции с моделью эмбеддингов внутри: chapter5_code_bge_m3.

    Индекс принадлежит модели: у nomic-embed-text вектор из 768 чисел,
    у bge-m3 из 1024, и в одной коллекции они не уживутся. Имя с моделью
    внутри позволяет держать оба и переключаться, ничего не пересобирая.
    """
    return f"{CODE_COLLECTION}_{model_slug()}"


def get_code_store(backend: str | None = None) -> VectorStore:
    """Хранилище под код: та же пара классов Главы 4, свой корпус.

    Фабрика Главы 4 (get_store) сюда не подошла: она знает только про свои
    корпуса и подставляет им свой префикс коллекции. Класс при этом
    переиспользуется целиком — интерфейс из пяти операций для того и был
    нужен.
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "memory":
        return MemoryVectorStore(persist_path=INDEX_DIR / "code.json")
    if backend == "chroma":
        return ChromaVectorStore(
            collection=code_collection(), persist_dir=CHROMA_PERSIST_DIR
        )
    raise ValueError(f"Неизвестное хранилище: {backend}. Доступны: memory, chroma")


# ====================================================================
# ИНДЕКС КОДА
# ====================================================================

class CodeIndex:
    """Фрагменты кода, их векторы и поиск по ним."""

    def __init__(
        self,
        store: VectorStore | None = None,
        root: Path | str | None = None,
        index_docs: bool | None = None,
    ):
        self.store = store if store is not None else get_code_store()
        self.root = Path(root) if root else DEFAULT_ROOT
        self.index_docs = INDEX_DOCS if index_docs is None else index_docs

    # ------------------------------------------------------------ индексация

    def collect(self, path: Path | str | None = None) -> tuple[list[CodeChunk], int]:
        """Обходит репозиторий и режет всё, что нашла. Без единого эмбеддинга.

        Отдельным методом — потому что этим же обходом считаются замеры
        главы: сколько фрагментов даёт нарезка по определениям, сколько
        абзацная, и сколько времени занимает разбор без модели.
        """
        root = Path(path) if path else self.root
        chunks: list[CodeChunk] = []
        files = 0

        for file_path in iter_sources(root):
            file_chunks = chunk_source(file_path, root=root if root.is_dir() else root.parent)
            if not self.index_docs:
                file_chunks = [chunk for chunk in file_chunks if chunk.language != "markdown"]
            if file_chunks:
                files += 1
            chunks.extend(file_chunks)

        return chunks, files

    def index(self, path: Path | str | None = None, force: bool = False) -> IndexReport:
        """Собирает или обновляет индекс кода.

        Логика сверки — из Главы 4, и она здесь важнее, чем там. id фрагмента
        зависит от его текста, поэтому неизменившиеся функции не уходят
        в модель эмбеддингов, а изменившиеся получают новый id — и старая
        версия попадает в список устаревших. Без её удаления агент отвечал бы
        кодом, которого в файле уже нет: для кода это не теоретическая
        неприятность, а норма жизни через неделю работы.
        """
        started = time.time()
        report = IndexReport()
        root = Path(path) if path else self.root

        if not root.exists():
            print(f"⚠️ Нет такой папки: {root}")
            return report

        chunks, report.files = self.collect(root)
        report.chunks = len(chunks)

        known = self.store.entries()
        fresh_ids = {chunk.id for chunk in chunks}

        if root.is_dir():
            stale = [doc_id for doc_id in known if doc_id not in fresh_ids]
        else:
            indexed_sources = {chunk.source for chunk in chunks}
            stale = [
                doc_id
                for doc_id, metadata in known.items()
                if metadata.get("source") in indexed_sources and doc_id not in fresh_ids
            ]
        if stale:
            report.removed = self.store.delete(stale)

        pending = [chunk for chunk in chunks if force or chunk.id not in known]
        report.unchanged = len(chunks) - len(pending)

        if pending:
            # Вот единственное место, где карточка встречается с кодом.
            # В базу при этом уезжает chunk.text — то есть код как он есть.
            vectors = embed_documents([embedding_text(chunk) for chunk in pending])
            report.added = self.store.add(
                ids=[chunk.id for chunk in pending],
                texts=[chunk.text for chunk in pending],
                embeddings=vectors,
                metadatas=[chunk.to_metadata() for chunk in pending],
            )

        report.seconds = time.time() - started
        return report

    def clear(self) -> None:
        self.store.clear()

    # ------------------------------------------------------------ поиск

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
        score_gap: float = SCORE_GAP,
    ) -> list[Hit]:
        """Ищет фрагменты кода по смыслу вопроса.

        Выдача с запасом (top_k * 2) сокращается до top_k уже после того,
        как тесты уехали вниз, — иначе понижать было бы нечего.
        """
        if not query or not query.strip():
            return []
        if self.store.count() == 0:
            return []

        hits = [
            hit for hit in self.store.search(embed_query(query), top_k=top_k * 2)
            if hit.score >= min_score
        ]
        if not hits:
            return []

        hits = demote_tests(hits, query)[:top_k]
        best = hits[0].score
        return [hit for hit in hits if best - hit.score <= score_gap]

    # ------------------------------------------------------------ выдача

    def build_context(self, hits: list[Hit], budget_tokens: int) -> str:
        """Собирает найденный код в блок, который влезает в бюджет.

        Отличий от Главы 4 два, и оба про то, что код — не проза.

        **Шапка фрагмента несёт адрес.** `chapter1/agent.py:82-91 · функция
        calculator` — этого достаточно, чтобы читатель открыл файл и
        проверил ответ. Номера в каждой строке (`82: def calculator(...)`)
        дали бы то же самое, но стоили бы четыре-пять символов на строку —
        на пяти фрагментах это сотня токенов ни за что.

        **Обрезается по строкам.** Оборванная посреди строки скобка сбивает
        модель сильнее, чем отсутствие последних трёх строк функции.
        """
        if not hits or budget_tokens <= 0:
            return ""

        # Место под служебную строку «Найдено фрагментов: N из M» резервируется
        # заранее — как в Главе 4. Иначе она дописывается поверх уже
        # израсходованного бюджета, и блок, который «влезает», вылезает ровно
        # на эту строку.
        limit = budget_tokens - (HEAD_COST if len(hits) > 1 else 0)

        parts: list[str] = []
        used = 0

        for number, hit in enumerate(hits, 1):
            # Близости в шапке НЕТ, в отличие от Главы 4. Причина — живой
            # прогон: получив «(близость 0.70)», модель переписала её
            # в ответ пользователю. Мера близости нужна нам для отладки
            # (её видно в тестах и в Hit.score), а человеку в ответе
            # «калькулятор реализован в chapter1/agent.py (близость 0.70)»
            # она не говорит ничего.
            header = f"[{number}] {describe(hit)}"
            fence = language_fence(hit)
            # Ограждение блока кода тоже стоит токенов: ```python сверху,
            # ``` снизу, пустая строка между фрагментами. Мелочь, которую
            # легко забыть, — и тогда на пяти фрагментах бюджет уезжает
            # на полсотни токенов.
            overhead = estimate_tokens(f"```{fence}") + estimate_tokens("```") + 3
            left = limit - used - estimate_tokens(header) - overhead
            if left <= 0:
                break

            text = fit_lines(hit.text, left)
            if not text:
                break

            parts.append(f"{header}\n```{fence}\n{text}\n```")
            used += estimate_tokens(header) + estimate_tokens(text) + overhead

        if not parts:
            return ""

        # Последний рубеж: оценка складывалась из кусков, а estimate_tokens
        # округляет вниз на каждом. Лишний фрагмент выбрасывается целиком —
        # обрезать блок посередине нельзя, иначе оборвётся ограждение кода
        # и модель увидит незакрытый ```.
        while parts:
            shown = len(parts)
            head = f"Найдено фрагментов: {shown} из {len(hits)}." if shown < len(hits) else ""
            result = "\n\n".join(([head] if head else []) + parts)
            if estimate_tokens(result) <= budget_tokens:
                return result
            parts.pop()

        return ""

    def retrieve(self, query: str, budget_tokens: int, top_k: int = TOP_K) -> str:
        """Поиск и сборка одним вызовом. То, что зовёт инструмент агента."""
        return self.build_context(self.search(query, top_k=top_k), budget_tokens)

    # ------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        """Что сейчас в индексе: по файлам, по языкам, по видам фрагментов."""
        entries = self.store.entries()
        sources: dict[str, int] = {}
        languages: dict[str, int] = {}
        kinds: dict[str, int] = {}

        def count(shelf: dict[str, int], value: object) -> None:
            key = str(value or "?")
            shelf[key] = shelf.get(key, 0) + 1

        for metadata in entries.values():
            count(sources, metadata.get("source"))
            count(languages, metadata.get("language"))
            count(kinds, metadata.get("kind"))

        return {
            "chunks": len(entries),
            "files": len(sources),
            "sources": dict(sorted(sources.items())),
            "languages": dict(sorted(languages.items(), key=lambda item: -item[1])),
            "kinds": dict(sorted(kinds.items(), key=lambda item: -item[1])),
            "store": type(self.store).__name__,
            "root": str(self.root),
        }


# ====================================================================
# РАНЖИРОВАНИЕ: ТЕСТЫ — НЕ РЕАЛИЗАЦИЯ
# ====================================================================

# Слова, по которым видно, что спрашивают ИМЕННО про тесты. Тогда понижать
# их нельзя — это и есть ответ.
TEST_WORDS = ("тест", "test", "проверя", "assert", "pytest")


def demote_tests(hits: list[Hit], query: str) -> list[Hit]:
    """Опускает фрагменты тестов ниже фрагментов реализации.

    Замер, из-за которого это появилось: на вопрос «где реализован
    калькулятор» в выдачу попали `chapter3/tests.py:1721` и
    `chapter2/tests.py:86` — тесты, где `calculator` вызывается, — и агент
    назвал их наравне с настоящей реализацией. Для вопроса «где реализовано»
    тест хуже реализации всегда, а не иногда: он про то же самое имя,
    но не про то же самое место.

    Понижение мягкое: тесты не выбрасываются, а уезжают в хвост. Если
    в корпусе вообще нет реализации, тест лучше пустоты — и если спросили
    про тесты, порядок не трогается вовсе.
    """
    if any(word in query.lower() for word in TEST_WORDS):
        return hits
    # Сортировка стабильная, поэтому внутри каждой группы порядок
    # по близости сохраняется.
    return sorted(hits, key=lambda hit: is_test_source(hit.source))


# ====================================================================
# ОФОРМЛЕНИЕ ВЫДАЧИ
# ====================================================================

FENCES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "markdown": "markdown",
    "config": "",
    "text": "",
}

TITLES = {
    "module": "шапка модуля",
    "function": "функция",
    "class": "класс",
    "method": "метод",
    "type": "тип",
    "block": "фрагмент",
    "section": "раздел",
}


def describe(hit: Hit) -> str:
    """Шапка фрагмента: адрес и что это такое.

    Собирается из метаданных, а не из текста: текст — это код, и приписывать
    к нему что-либо мы договорились только в карточке для эмбеддинга.
    """
    metadata = hit.metadata or {}
    source = str(metadata.get("source", "?"))
    start = int(metadata.get("start_line", 0) or 0)
    end = int(metadata.get("end_line", 0) or 0)
    name = str(metadata.get("name", ""))
    kind = TITLES.get(str(metadata.get("kind", "")), "фрагмент")
    parts = int(metadata.get("parts", 1) or 1)
    part = int(metadata.get("part", 1) or 1)

    where = f"{source}:{start}-{end}" if start else source
    what = f"{kind} {name}" if name else kind
    tail = f", часть {part} из {parts}" if parts > 1 else ""
    return f"{where} · {what}{tail}"


def language_fence(hit: Hit) -> str:
    """Язык для ограждения блока кода — подсказка модели, что перед ней код."""
    return FENCES.get(str((hit.metadata or {}).get("language", "")), "")


def fit_lines(text: str, budget_tokens: int) -> str:
    """Обрезает фрагмент по строкам так, чтобы он влез в бюджет.

    Возвращает пустую строку, если не влезает даже начало: полтора заголовка
    функции — не ответ, и место под них тратить незачем.
    """
    if estimate_tokens(text) <= budget_tokens:
        return text

    lines = text.splitlines()
    kept: list[str] = []
    used = estimate_tokens(CUT_MARK)

    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens:
            break
        kept.append(line)
        used += cost

    if len(kept) < MIN_USEFUL_LINES:
        return ""
    return "\n".join(kept + [CUT_MARK])


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_code_index: CodeIndex | None = None


def get_code_index() -> CodeIndex:
    """Общий индекс кода (singleton, как база знаний в Главе 4)."""
    global _code_index
    if _code_index is None:
        _code_index = CodeIndex()
    return _code_index


def set_code_index(index: CodeIndex | None) -> None:
    """Подменяет общий индекс. Нужно тестам, чтобы не трогать настоящий."""
    global _code_index
    _code_index = index
