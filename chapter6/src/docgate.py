"""
Ответ «в документах этого нет» — тот же приём, что для кода (пункт 6.2).

Долг, который Глава 6 сначала оставила открытым. Отказ по коду она сделала,
а корпус документов Главы 4 остался как был: любой вопрос возвращал до пяти
самых похожих абзацев, и модель отвечала по ним.

Живой прогон показал, чем это оборачивается, и виноваты были две опечатки.
Вопрос «где реализоано распознование изображений» не подошёл под маркеры
кода (в списке `реализов`, а в вопросе `реализоано`), уехал в документы —
и получил ответ «Распознавание изображений реализовано в главе 1 курса,
в этой главе описывается, как проверять запрос на инъекцию». Выдумка
от первого слова до последнего, и ничто в системе её не остановило.

Ничего нового здесь нет: тот же BM25Index, тот же Signal, то же правило
«ни одного содержательного слова». Меняется корпус — вместо фрагментов кода
абзацы документации, — и меняется список слов формы вопроса: у вопроса
к документам своя рамка («расскажи», «что такое», «объясни»), а «класс»
и «метод» в нём как раз содержательные.
"""

from chapter4.src.chunking import chunk_file
from chapter4.src.knowledge import KnowledgeBase, get_knowledge_base, iter_documents

from .bm25 import BM25Index
from .hybrid import Signal
from .lexical import _TOKEN as WORD_PATTERN
from .lexical import keep

# Слова формы вопроса К ДОКУМЕНТАМ. Список СВОЙ, а не общий с кодом,
# и это не дублирование. В вопросе о коде «класс» и «метод» — форма
# («где реализован класс X»), а в вопросе к документации ровно они
# и есть предмет: «что написано про класс Conversation». Один список
# на оба корпуса выбрасывал бы из проверки то самое слово, ради которого
# вопрос и задан.
DOC_FRAME_TOKENS = frozenset({
    "расскажи", "объясни", "опиши", "покажи", "напомни", "перечисли",
    "такое", "значит", "означает", "написано", "сказано", "говорится",
    "документ", "документа", "документе", "документации", "документах",
    "глава", "главе", "главы", "раздел", "разделе", "разделы",
    "правило", "правила", "правил", "курс", "курса", "курсе",
})

FRAME = DOC_FRAME_TOKENS


class DocumentGate:
    """Лексический индекс поверх корпуса документов — только ради отказа.

    Ранжировать документы он не будет: этим занимается векторный поиск
    Главы 4, и замер Главы 6 показал, что на хорошей модели эмбеддингов
    он лучше поиска по словам. Здесь нужен один ответ: встречается ли
    хоть одно содержательное слово вопроса в документах вообще.
    """

    def __init__(self, base: KnowledgeBase | None = None):
        self.base = base if base is not None else get_knowledge_base()
        self.lexical = BM25Index()

    def sync(self) -> int:
        """Пересобирает индекс из тех же файлов, что читает база знаний."""
        root = self.base.docs_dir
        if not root.exists():
            return 0

        chunks = []
        for path in iter_documents(root):
            chunks.extend(chunk_file(path, root=root if root.is_dir() else root.parent))

        self.lexical.clear()
        return self.lexical.add(
            ids=[chunk.id for chunk in chunks],
            texts=[chunk.text for chunk in chunks],
        )

    def content_tokens(self, question: str) -> list[str]:
        """Слова о предмете вопроса, без слов о его форме."""
        seen: set[str] = set()
        content: list[str] = []
        for match in WORD_PATTERN.finditer(question):
            word = match.group(0).lower()
            if keep(word) and word not in FRAME and word not in seen:
                seen.add(word)
                content.append(word)
        return content

    def signal(self, question: str) -> Signal:
        """Что известно о вопросе до того, как показана выдача."""
        tokens = self.content_tokens(question)
        if not self.lexical.count():
            # Индекса нет — это «искать негде», а не «ответа не существует».
            # Отказывать здесь значило бы молчать на каждый вопрос, пока
            # корпус не собран.
            return Signal()
        if not tokens:
            return Signal(tokens=tokens, missing=tokens)

        missing = [token for token in tokens if not self.lexical.document_frequency(token)]
        hits = self.lexical.search(question, top_k=1)
        return Signal(
            best=hits[0].score if hits else 0.0,
            support=1 - len(missing) / len(tokens),
            tokens=tokens,
            missing=missing,
        )

    def looks_absent(self, question: str) -> bool:
        return self.signal(question).absent


_gate: DocumentGate | None = None


def get_document_gate() -> DocumentGate:
    """Общие ворота (singleton). Собираются при первом обращении."""
    global _gate
    if _gate is None:
        _gate = DocumentGate()
        _gate.sync()
    return _gate


def set_document_gate(gate: DocumentGate | None) -> None:
    """Подменяет ворота. Нужно тестам, чтобы не трогать настоящий корпус."""
    global _gate
    _gate = gate
