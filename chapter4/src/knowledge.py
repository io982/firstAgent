"""
База знаний: индексация документов и поиск под бюджет (пункты 4.4, 4.5).

Здесь собирается весь конвейер главы:

    файлы → нарезка → эмбеддинги → индекс → поиск → фрагменты под бюджет

Две вещи, которые в этом конвейере делаются не так, как в туториалах:

  1. **Индексация идемпотентна.** id чанка — хэш его содержимого, поэтому
     повторный запуск не удваивает корпус, а неизменившиеся файлы вообще
     не уходят в модель эмбеддингов. Заодно удаляются осиротевшие чанки
     от старых версий файлов — иначе агент продолжает отвечать по тексту,
     которого в документе уже нет.
  2. **У выдачи есть потолок в токенах.** Найденное едет в то же окно, где
     уже лежат системный промпт, пересказ прошлой сессии и история разговора.
     Потолок считает агент и ставит его сюда извне (см. set_retrieval_budget):
     без потолка один вызов поиска вытесняет весь разговор.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# estimate_tokens берётся из Главы 3, а не копируется сюда: у оценки размера
# контекста должно быть одно место жительства, иначе копии разъезжаются
# (ровно та история, что описана в chapter2/agent.py про is_safe_query).
from chapter3.src.context import estimate_tokens

from .chunking import Chunk, chunk_file, iter_documents
from .embeddings import embed_documents, embed_query
from .vectorstore import Hit, VectorStore, get_store

# ====================================================================
# НАСТРОЙКИ ПОИСКА
# ====================================================================

# Корпус главы. Можно указать свою папку:
#   PowerShell:   $env:AGENT_DOCS_DIR = "C:\docs"
#   Linux/macOS:  export AGENT_DOCS_DIR=~/docs
DEFAULT_DOCS_DIR = Path(
    os.environ.get("AGENT_DOCS_DIR", str(Path(__file__).parent.parent / "docs"))
)

# Сколько фрагментов достаём.
#
# Сначала здесь стояло три — из соображения «меньше мусора в контексте».
# Замер на корпусе главы показал, чем это оборачивается. Вопрос «Какое
# контекстное окно у агента?»: фрагмент, где написано про 8192, оказался
# на ЧЕТВЁРТОМ месте — со счётом 0.719 против 0.740 у первого. Три верхних
# места заняли соседние куски того же файла, каждый про контекст вообще.
#
# Разрыв между «тем самым» фрагментом и случайным соседом — сотые доли,
# и это не поправить порогом (см. MIN_SCORE): весь список упакован
# в полосу шириной 0.04. Пока ранжирование такое плотное, единственная
# защита — брать с запасом. Пять фрагментов по ~300 токенов = 1500,
# в потолок выдачи расширенного окна помещаются.
#
# Настоящее решение — не k, а другое ранжирование: лексический поиск рядом
# с векторным и реранкер поверх обоих. Это темы следующих глав.
TOP_K = 5

# Абсолютный порог близости. По умолчанию ВЫКЛЮЧЕН (0.0), и это результат
# замера, а не лень.
#
# Ожидание было такое: релевантные фрагменты дают близость сильно выше
# случайных, порог отсекает мусор, и агент честно говорит «не знаю».
# На корпусе главы это ожидание не подтвердилось. Запрос «Как приготовить
# борщ?» получает лучший фрагмент с близостью 0.743 — выше, чем четыре
# по-настоящему релевантных вопроса из пяти («Как запускать тесты?» — 0.733,
# «Зачем оборачивают вывод в теги?» — 0.729, «Какие переменные окружения
# есть?» — 0.707). Нормировка на среднее по корпусу (z-оценка) тоже
# не разделяет: у борща 2.65, у вопроса про потолок поля памяти — 1.61.
#
# Причина не в поломке, а в том, как устроены эмбеддинги: близость измеряет
# «про то же самое ли это вообще», а не «есть ли здесь ответ». Любой русский
# текст-вопрос ближе к любому русскому тексту-документу, чем к случайному
# шуму, и базовый уровень поднимается для всех сразу.
#
# Отсюда честный вывод главы: РЕШЕНИЕ «ОТВЕТА НЕТ» ВЕКТОРНЫЙ ПОИСК ПРИНЯТЬ
# НЕ МОЖЕТ. Он ранжирует, а не отвечает. Отказ приходится возлагать на
# модель («в найденных фрагментах ответа нет — так и скажи») и на приёмы
# следующих глав: лексический поиск и реранкер.
#
# Порог оставлен параметром: на своём корпусе замерьте и поставьте своё.
MIN_SCORE = 0.0

# Место под служебную строку «Найдено фрагментов: N из M» и пустую строку
# после неё. Замерено по самой строке, а не выбрано на глаз: она дописывается
# в конце, и недооценка означает блок, который вылезает за бюджет.
HEAD_COST = estimate_tokens("Найдено фрагментов: 99 из 99.") + 1

# Отметка об обрезке. Её длина вычитается из бюджета — иначе обрезанный
# фрагмент вместе с отметкой снова не влезает.
CUT_MARK = " […фрагмент обрезан по бюджету контекста]"

# Короче этого обрезать бессмысленно — см. build_context.
MIN_USEFUL_CHARS = 80

# Насколько фрагмент может отставать от лучшего, чтобы попасть в выдачу.
#
# Этот фильтр, в отличие от абсолютного порога, хотя бы имеет смысл:
# фрагменты сравниваются между собой внутри одного запроса, и общий уровень
# близости сокращается. У кого есть явный победитель — лишнее отсекается,
# у кого все идут вплотную — берутся все. На корпусе главы:
#
#   «Как оформлять README главы?»       0.759 0.739 0.728 0.714 0.706 → 4
#   «Какое контекстное окно у агента?»  0.740 0.724 0.721 0.719 0.716 → 5
#
# Видно и то, насколько плотно упакована выдача: разброс по всей пятёрке —
# сотые доли. Отсюда и TOP_K=5 вместо трёх, и бессилие порога выше.
SCORE_GAP = 0.05


@dataclass
class IndexReport:
    """Что произошло при индексации. Печатается пользователю, проверяется тестами."""

    files: int = 0
    chunks: int = 0
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"📚 Проиндексировано: {self.files} файлов, {self.chunks} фрагментов "
            f"(новых {self.added}, без изменений {self.unchanged}, "
            f"удалено устаревших {self.removed}) за {self.seconds:.1f} с"
        )


# ====================================================================
# БАЗА ЗНАНИЙ
# ====================================================================

class KnowledgeBase:
    """Документы, их векторы и поиск по ним."""

    def __init__(
        self,
        store: VectorStore | None = None,
        docs_dir: Path | str | None = None,
    ):
        self.store = store if store is not None else get_store()
        self.docs_dir = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR

    # ------------------------------------------------------------ индексация

    def index(self, path: Path | str | None = None, force: bool = False) -> IndexReport:
        """Собирает (или обновляет) индекс по папке с документами.

        Args:
            path: Папка или файл. По умолчанию — корпус главы.
            force: Пересчитать векторы даже для неизменившихся чанков.
                Нужно при смене модели эмбеддингов: старые векторы считала
                другая модель, и сравнивать их с новыми бессмысленно.
        """
        started = time.time()
        root = Path(path) if path else self.docs_dir
        report = IndexReport()

        if not root.exists():
            print(f"⚠️ Нет папки с документами: {root}")
            return report

        files = iter_documents(root)
        report.files = len(files)

        chunks: list[Chunk] = []
        for file_path in files:
            chunks.extend(chunk_file(file_path, root=root if root.is_dir() else root.parent))
        report.chunks = len(chunks)

        known = self.store.entries()
        fresh_ids = {chunk.id for chunk in chunks}

        # Устаревшие чанки — те, которых в новой нарезке нет. Есть два способа
        # ими стать, и оба заканчиваются одинаково: агент отвечает по тексту,
        # которого в документах уже нет, молча и уверенно.
        #
        #   * файл поправили — текст нарезался иначе, id изменились;
        #   * файл удалили — его вообще больше нет среди найденных.
        #
        # Второй случай ловится только при сверке ПАПКИ целиком: тогда
        # найденные файлы — это и есть весь корпус, и всё, чего в них нет,
        # можно смело выбрасывать. При сверке одного файла такого права у нас
        # нет: остальной корпус мы в этот раз просто не смотрели.
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
            vectors = embed_documents([chunk.text for chunk in pending])
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
        """Ищет фрагменты по смыслу запроса.

        Фильтрация — здесь, а не в хранилище: хранилище обязано честно
        ранжировать всё, что у него есть, а какие из этого фрагменты
        показывать модели — решает база знаний.
        """
        if not query or not query.strip():
            return []
        if self.store.count() == 0:
            return []

        hits = [hit for hit in self.store.search(embed_query(query), top_k=top_k)
                if hit.score >= min_score]
        if not hits:
            return []

        best = hits[0].score
        return [hit for hit in hits if best - hit.score <= score_gap]

    def build_context(self, hits: list[Hit], budget_tokens: int) -> str:
        """Собирает найденное в один блок, который влезает в бюджет.

        Фрагменты идут по убыванию близости, и обрезается последний, а не
        первый: если места хватает на полтора фрагмента, лучше отдать самый
        близкий целиком.

        Обрезка проговаривается вслух — по той же причине, что и в read_file
        Главы 2: молча укороченный текст модель считает текстом целиком.
        """
        if not hits or budget_tokens <= 0:
            return ""

        # Место под строку «Найдено фрагментов: N из M» резервируется заранее.
        # Иначе она дописывается поверх уже израсходованного бюджета — и блок,
        # который «влезает», на самом деле вылезает ровно на эту строку.
        limit = budget_tokens - (HEAD_COST if len(hits) > 1 else 0)

        parts: list[str] = []
        used = 0

        for number, hit in enumerate(hits, 1):
            header = f"[{number}] {hit.label()} (близость {hit.score:.2f})"
            # Плюс токен на пустую строку между фрагментами: мелочь, но
            # бюджет должен сходиться с тем, что реально уедет в контекст.
            left = limit - used - estimate_tokens(header) - (1 if parts else 0)
            if left <= 0:
                break

            # Крошка «файл › заголовок» уже напечатана в шапке фрагмента,
            # а внутри чанка она лежит потому, что участвовала в эмбеддинге
            # (см. chunk_text). В контекст модели её хватит одного раза.
            text = hit.text
            for crumb in (hit.label(), hit.heading):
                if crumb and text.startswith(crumb + "\n"):
                    text = text[len(crumb) + 1:]
                    break

            if estimate_tokens(text) > left:
                # estimate_tokens — это len // 2, поэтому обратный перевод
                # бюджета в символы честный: left токенов ≈ 2 * left символов.
                cut = max(0, left * 2 - len(CUT_MARK))
                if cut < MIN_USEFUL_CHARS:
                    # Огрызок в полстроки не отвечает ни на что, а место
                    # занимает наравне с целым фрагментом.
                    break
                text = text[:cut] + CUT_MARK

            parts.append(f"{header}\n{text}")
            used += estimate_tokens(header) + estimate_tokens(text) + (1 if len(parts) > 1 else 0)

        if not parts:
            return ""

        shown = len(parts)
        head = f"Найдено фрагментов: {shown} из {len(hits)}." if shown < len(hits) else ""
        result = "\n\n".join(([head] if head else []) + parts)

        # Последний рубеж. Бюджет выше складывался из кусков, а estimate_tokens
        # округляет вниз на каждом — сумма оценок бывает на пару токенов меньше
        # оценки склейки. Один токен перебора ничего не рушит, но обещание
        # «влезает в бюджет» должно выполняться буквально, иначе его нельзя
        # проверить тестом.
        if estimate_tokens(result) > budget_tokens:
            result = result[: max(0, budget_tokens * 2 - len(CUT_MARK))] + CUT_MARK

        return result

    def retrieve(self, query: str, budget_tokens: int, top_k: int = TOP_K) -> str:
        """Поиск + сборка одним вызовом. То, что зовёт инструмент агента."""
        return self.build_context(self.search(query, top_k=top_k), budget_tokens)

    # ------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        """Что сейчас в индексе. Нужно REPL-команде и тестам."""
        entries = self.store.entries()
        sources: dict[str, int] = {}
        for metadata in entries.values():
            source = str(metadata.get("source", "?"))
            sources[source] = sources.get(source, 0) + 1
        return {
            "chunks": len(entries),
            "sources": dict(sorted(sources.items())),
            "store": type(self.store).__name__,
            "docs_dir": str(self.docs_dir),
        }


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_knowledge_base: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Возвращает общую базу знаний (singleton, как get_memory в Главе 3)."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def set_knowledge_base(base: KnowledgeBase | None) -> None:
    """Подменяет общую базу знаний. Нужно тестам, чтобы не трогать настоящий индекс."""
    global _knowledge_base
    _knowledge_base = base
