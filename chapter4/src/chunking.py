"""
Document pipeline, шаг 1: разбор и нарезка документов (пункт 4.4 ROADMAP).

Нарезка — самая недооценённая часть RAG. Ошибиться в ней легче, чем
в эмбеддингах и базе вместе взятых, а последствия видны не сразу:

  * чанк слишком большой — в контекст влезает один фрагмент вместо трёх,
    и половина его текста не про вопрос;
  * чанк слишком маленький — мысль разорвана пополам, и ни одна половина
    не отвечает на вопрос целиком;
  * чанк без заголовка — «поддерживается через переменную окружения» само
    по себе не значит ничего: непонятно, что именно поддерживается.

Отсюда три решения этого модуля: режем по границам разделов и абзацев,
а не по символам; переносим заголовок раздела внутрь каждого чанка;
соседние чанки перекрываются, чтобы фраза на стыке не пропала.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# ====================================================================
# РАЗМЕРЫ
# ====================================================================

# Размер чанка в СИМВОЛАХ, а не в токенах: точный подсчёт токенов требует
# токенизатора модели, а нам нужна оценка (см. estimate_tokens в Главе 3,
# где один токен — примерно два символа русского текста).
#
# 600 символов ≈ 300 токенов, то есть три фрагмента — около 900. Это с запасом
# помещается в потолок выдачи Главы 4, и размер выбран уже не бюджетом, а
# смыслом: 600 символов — это два-три абзаца, законченная мысль с контекстом
# вокруг неё. Чанк вдвое крупнее чаще всего наполовину не про вопрос, вдвое
# мельче — обрывает мысль на середине.
CHUNK_SIZE = 600

# Перекрытие соседних чанков. Нужно из-за границ: предложение, разрезанное
# пополам, не находится ни по одной половине. 120 символов — примерно одно
# предложение, то есть минимальный кусок, который ещё имеет смысл.
CHUNK_OVERLAP = 120

# Хвост короче этого приклеивается к предыдущему чанку. Обрывок в двадцать
# символов — это не документ, а мусор в выдаче: он всегда чуть-чуть похож
# на всё сразу.
MIN_CHUNK = 80

# Что считаем документами. Глава 4 — про текстовые базы знаний;
# исходники проекта разбираются иначе и живут в отдельной главе.
DOCUMENT_EXTENSIONS = (".md", ".txt", ".rst")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


# ====================================================================
# ЧАНК
# ====================================================================

@dataclass(frozen=True)
class Chunk:
    """Кусок документа вместе с тем, откуда он взялся.

    Метаданные — не украшение. Без источника агент не может сослаться
    на документ, а пользователь — проверить ответ; именно проверяемость
    отличает RAG от красивой выдумки.
    """

    text: str
    source: str
    position: int
    heading: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            # frozen=True запрещает обычное присваивание, но id обязан
            # зависеть от содержимого, а не задаваться снаружи руками.
            object.__setattr__(self, "id", make_chunk_id(self.source, self.position, self.text))

    def to_metadata(self) -> dict[str, str | int]:
        return {"source": self.source, "position": self.position, "heading": self.heading}

    def label(self) -> str:
        """Человекочитаемая ссылка на фрагмент: «файл › заголовок»."""
        return f"{self.source} › {self.heading}" if self.heading else self.source


def make_chunk_id(source: str, position: int, text: str) -> str:
    """Идентификатор чанка = хэш от источника, позиции и текста.

    Детерминированность здесь работает как дедупликация: переиндексация
    неизменившегося файла даёт те же id, база перезаписывает те же записи,
    и корпус не раздувается копиями. Случайный uuid дал бы при каждом
    запуске новый набор документов поверх старого.
    """
    raw = f"{source}\x00{position}\x00{text}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


# ====================================================================
# РАЗБОР
# ====================================================================

def normalize_text(text: str) -> str:
    """Приводит текст к одному виду перед нарезкой.

    Разные переводы строки и лишние пустые строки — это не косметика:
    от них зависит, где пройдут границы абзацев, а значит и границы чанков.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str, markdown: bool = True) -> list[tuple[str, str]]:
    """Разбивает markdown на разделы по заголовкам.

    Возвращает пары (заголовок, тело). Заголовок — путь из вложенных
    заголовков через « › », чтобы «Ограничения» из одного раздела не
    выглядели в выдаче так же, как «Ограничения» из другого.

    `markdown=False` — для файлов, где решётка означает не заголовок.
    Проверено на исходнике из Главы 1, положенном в корпус под именем
    `agent.txt`: строки-разделители `# =====================` разобрались
    как заголовки, и в выдаче у фрагментов вместо темы стояли ряды знаков
    равенства. Разметка — свойство формата, а не текста вообще.
    """
    text = normalize_text(text)
    if not text:
        return []

    matches = list(HEADING_RE.finditer(text)) if markdown else []
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    # Стек заголовков: на каждом уровне помним последний. Заголовок третьего
    # уровня без своего второго — обычное дело в реальных файлах, поэтому
    # стек именно обрезается по уровню, а не считается строго вложенным.
    path: list[str] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        del path[level - 1:]
        path.append(title)

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((" › ".join(path), body))

    return sections


def split_paragraphs(text: str) -> list[str]:
    """Абзацы. Пустая строка — самая надёжная граница смысла в тексте."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Режет слишком длинный абзац.

    Сначала по границам предложений, и только если предложение само длиннее
    лимита (таблица, длинный код, ссылка на пол-экрана) — по символам.
    Резать по символам сразу проще, но именно так получаются чанки,
    начинающиеся с середины слова.
    """
    if len(text) <= size:
        return [text]

    sentences = re.split(r"(?<=[.!?…:])\s+", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        while len(sentence) > size:
            # Предложение длиннее лимита: отрезаем ровно по размеру.
            pieces.append(sentence[:size])
            sentence = sentence[size - overlap:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Хвост предыдущего чанка для перекрытия, обрезанный по границе слова."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


# ====================================================================
# НАРЕЗКА
# ====================================================================

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    markdown: bool = True,
) -> list[Chunk]:
    """Режет текст документа на чанки.

    Порядок: разделы → абзацы → набор абзацев до лимита → перекрытие.

    В начало каждого чанка дописывается «хлебная крошка» — имя файла и путь
    заголовков. Фрагмент должен отвечать на вопрос сам по себе, без соседей,
    которых рядом не будет, а имя файла — часть ответа.

    Имя файла попало в текст не сразу, и вот замер, который его туда привёл.
    Пока в тексте был только заголовок, вопрос «что в agent.txt» находил
    список переменных окружения из conventions.md (близость 0.770) — просто
    потому, что там много строк вида `AGENT_MODEL`. Ни один из сорока чанков
    самого agent.txt в тройку не попадал: имя файла жило в метаданных, а
    ищем мы по тексту. С крошкой тот же вопрос находит нужный файл (0.810),
    и остальные четыре проверочных вопроса отвечаются как раньше.
    """
    chunks: list[Chunk] = []
    position = 0

    for heading, body in split_sections(text, markdown=markdown):
        breadcrumb = f"{source} › {heading}" if heading else source
        prefix = f"{breadcrumb}\n"
        # Крошка занимает место внутри чанка, значит на текст его остаётся
        # меньше. Без этой поправки чанки с длинным путём заголовков вылезают
        # за лимит — незаметно, потому что считается он до склейки.
        body_limit = max(MIN_CHUNK, chunk_size - len(prefix))

        pieces: list[str] = []
        for paragraph in split_paragraphs(body):
            pieces.extend(hard_split(paragraph, body_limit, overlap))

        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > body_limit:
                chunks.append(Chunk(prefix + current, source, position, heading))
                position += 1
                carry = _tail(current, overlap)
                # Перекрытие не имеет права ломать потолок: кусок, нарезанный
                # ровно по лимиту, вместе с хвостом предыдущего вылезет за него.
                # Если не влезает — жертвуем перекрытием, а не размером.
                if carry and len(carry) + 2 + len(piece) <= body_limit:
                    current = f"{carry}\n\n{piece}"
                else:
                    current = piece
            else:
                current = candidate

        if current:
            if len(current) < MIN_CHUNK and chunks and chunks[-1].heading == heading:
                # Короткий хвост не образует отдельный документ: сам по себе
                # он ни на что не отвечает, но в выдаче конкурирует наравне.
                previous = chunks.pop()
                position -= 1
                merged = f"{previous.text}\n\n{current}"
                chunks.append(Chunk(merged, source, position, heading))
                position += 1
            else:
                chunks.append(Chunk(prefix + current, source, position, heading))
                position += 1

    return chunks


def chunk_file(path: Path | str, root: Path | str | None = None) -> list[Chunk]:
    """Читает файл и режет его. Источник — путь относительно корня корпуса."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"⚠️ Пропускаю {path}: {e}")
        return []

    source = str(path.relative_to(root)) if root else path.name
    # Заголовки ищем только там, где они действительно заголовки.
    return chunk_text(text, source.replace("\\", "/"), markdown=path.suffix.lower() == ".md")


def iter_documents(
    root: Path | str,
    extensions: tuple[str, ...] = DOCUMENT_EXTENSIONS,
) -> list[Path]:
    """Находит документы в папке (рекурсивно), в стабильном порядке.

    Сортировка нужна не для красоты: без неё порядок обхода зависит от
    файловой системы, а вместе с ним — позиции чанков и их id.
    """
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []

    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(found)
