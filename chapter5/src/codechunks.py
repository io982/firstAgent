"""
Нарезка кода по определениям (пункт 5.2).

Долг Главы 4: её нарезка режет текст по абзацам, а у кода абзацев нет.
Пустая строка внутри функции — не граница мысли, а форматирование, и
фрагмент, отрезанный по ней, начинается с середины тела: ни имени, ни
сигнатуры, ни докстроки. Здесь границы задаёт само определение — функция,
класс, метод, шапка модуля с импортами.

Три способа разбора, от точного к грубому, и выбирается способ по языку:

  * **ast** для Python — настоящий разбор: границы определений известны
    точно, вместе с именами, сигнатурами и докстроками;
  * **скобочный сканер** для JavaScript и TypeScript — разбора нет,
    считаем фигурные скобки мимо строк и комментариев. Где именно он
    ошибается, написано у chunk_braces();
  * **построчные окна** для всего остального — конфигов, незнакомых
    языков и файлов с синтаксической ошибкой.

Последний способ важнее, чем кажется: файл в репозитории регулярно бывает
сломан ровно в тот момент, когда его индексируют. Индексация, падающая
на неудачной правке, бесполезна.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from chapter4.src.chunking import chunk_text, make_chunk_id

from .languages import language_of, read_source

# ====================================================================
# РАЗМЕРЫ
# ====================================================================

# Потолок фрагмента в строках и в символах — считаются оба, потому что
# ломаются они по-разному: сгенерированный словарь на 300 коротких строк
# и одна строка регулярного выражения на 2000 символов одинаково плохи
# в выдаче, но ловятся разными счётчиками.
#
# 60 строк выбраны по бюджету, а не на глаз: фрагмент такого размера — это
# примерно 2000 символов, то есть около 1000 токенов по оценке Главы 3.
# Три таких фрагмента уже не помещаются в потолок выдачи (см. codebase.py),
# поэтому всё, что длиннее, режется на части.
MAX_CHUNK_LINES = 60
MAX_CHUNK_CHARS = 2000

# Перекрытие частей длинной функции. Для кода оно измеряется в строках,
# а не в символах: половина строки кода не значит ничего.
OVERLAP_LINES = 4

# Размер окна в фоллбэке. Меньше потолка: у окна нет ни имени, ни границ
# по смыслу, поэтому пусть его будет проще прочитать целиком.
WINDOW_LINES = 40

# Фрагмент короче — не фрагмент. Строка `import os` сама по себе одинаково
# похожа на любой вопрос про импорты в любом файле.
MIN_CHUNK_CHARS = 40


# ====================================================================
# ФРАГМЕНТ
# ====================================================================

@dataclass(frozen=True)
class CodeChunk:
    """Кусок кода вместе с тем, что о нём известно из разбора.

    Разница с Chunk Главы 4 — в двух полях. Первое: у фрагмента кода есть
    АДРЕС (файл и диапазон строк), и он должен доехать до ответа, иначе
    проверить ответ нельзя. Второе: сигнатура и докстрока — материал для
    карточки (см. cards.py), то есть для связи вопроса по-русски с кодом,
    который на этом языке не написан.
    """

    text: str
    source: str
    language: str
    kind: str            # module | function | class | method | type | block | section
    name: str = ""       # квалифицированное имя: TodoList.add
    signature: str = ""
    docstring: str = ""
    start_line: int = 0  # 1-based, включительно; 0 — строки неизвестны
    end_line: int = 0
    part: int = 1
    parts: int = 1
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            # id зависит от содержимого и адреса, как в Главе 4: тогда
            # переиндексация неизменившегося файла даёт те же id, и корпус
            # не раздувается копиями.
            object.__setattr__(
                self, "id", make_chunk_id(self.source, self.start_line, self.text)
            )

    def label(self) -> str:
        """Ссылка на фрагмент: `файл:строки`, как её печатают линтеры и трейсбэки."""
        if not self.start_line:
            return f"{self.source} › {self.name}" if self.name else self.source
        where = f"{self.source}:{self.start_line}-{self.end_line}"
        return f"{where} ({self.part}/{self.parts})" if self.parts > 1 else where

    def title(self) -> str:
        """Человеческое название фрагмента для шапки выдачи."""
        kinds = {
            "module": "шапка модуля",
            "function": "функция",
            "class": "класс",
            "method": "метод",
            "type": "тип",
            "block": "фрагмент",
            "section": "раздел",
        }
        kind = kinds.get(self.kind, self.kind)
        return f"{kind} {self.name}" if self.name else kind

    def to_metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "language": self.language,
            "kind": self.kind,
            "name": self.name,
            "signature": self.signature,
            "docstring": self.docstring,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "part": self.part,
            "parts": self.parts,
        }


# ====================================================================
# ОБЩЕЕ
# ====================================================================

def _line_span(lines: list[str], start: int, end: int) -> str:
    """Текст строк с start по end включительно (нумерация с единицы)."""
    return "\n".join(lines[start - 1:end]).rstrip()


def _grab_comments_above(lines: list[str], start: int) -> int:
    """Сдвигает начало фрагмента вверх, на комментарий над определением.

    Комментарий, объясняющий функцию, стоит НАД ней, и по разбору он
    к функции не относится — ast его вообще не видит. Оставить его снаружи
    значит выбросить самый человеческий текст, какой есть у фрагмента:
    в этом курсе объяснение «почему» живёт именно в таких комментариях.

    Пустая строка обрывает захват: комментарий через строку от определения
    относится не к нему.
    """
    line = start - 1
    while line >= 1 and lines[line - 1].lstrip().startswith("#"):
        line -= 1
    return line + 1


def _split_long(chunk: CodeChunk, lines: list[str]) -> list[CodeChunk]:
    """Режет слишком длинный фрагмент на части с перекрытием.

    Части остаются частями одного определения: имя, сигнатура и докстрока
    копируются в каждую, и в выдаче каждая часть подписана «2/3». Иначе
    вторая половина длинной функции приезжает к модели безымянной — ровно
    та беда, из-за которой абзацная нарезка на коде и не работает.
    """
    if chunk.end_line - chunk.start_line + 1 <= MAX_CHUNK_LINES and len(chunk.text) <= MAX_CHUNK_CHARS:
        return [chunk]

    # Оба потолка считаются в одном цикле, а не по очереди. Потолок в строках
    # один не спасает: шестьдесят строк сплошного словаря — это втрое больше
    # символов, чем шестьдесят строк обычного кода, и такой «фрагмент»
    # съедает бюджет выдачи целиком.
    spans: list[tuple[int, int]] = []
    start = chunk.start_line
    while start <= chunk.end_line:
        end = start
        used = len(lines[start - 1]) + 1
        while end < chunk.end_line:
            following = len(lines[end]) + 1
            if (end - start + 2) > MAX_CHUNK_LINES or used + following > MAX_CHUNK_CHARS:
                break
            used += following
            end += 1
        spans.append((start, end))
        if end >= chunk.end_line:
            break
        start = max(start + 1, end + 1 - OVERLAP_LINES)

    return [
        CodeChunk(
            text=_line_span(lines, start, end),
            source=chunk.source,
            language=chunk.language,
            kind=chunk.kind,
            name=chunk.name,
            signature=chunk.signature,
            docstring=chunk.docstring,
            start_line=start,
            end_line=end,
            part=number,
            parts=len(spans),
        )
        for number, (start, end) in enumerate(spans, 1)
    ]


def _keep(chunk: CodeChunk) -> bool:
    """Отсеивает фрагменты, которые не несут смысла сами по себе.

    Потолок на длину действует на всё, кроме ОПРЕДЕЛЕНИЙ. Определение
    остаётся в индексе любой длины: у `def clear(self): self._records.clear()`
    две строки, но и имя, и адрес, и карточка у него есть, а вопрос «где
    очищается хранилище» без него не отвечается. Шапка модуля из одной
    строки — другое дело: имя у неё техническое, и отвечать ей нечем.

    Проверено на этом же курсе: без исключения из индекса выпадала шапка
    короткого класса — вместе с его докстрокой, то есть с единственным
    человеческим описанием класса.
    """
    if chunk.kind in ("function", "class", "method", "type") and chunk.name:
        return True
    return len(chunk.text.strip()) >= MIN_CHUNK_CHARS


# ====================================================================
# PYTHON: РАЗБОР ЧЕРЕЗ AST
# ====================================================================

def signature_of(node: ast.AST) -> str:
    """Сигнатура определения, собранная из дерева, а не из текста.

    ast.unparse печатает аргументы ровно так, как их написал автор,
    вместе с типами и значениями по умолчанию, — и не спотыкается
    на сигнатуре, растянутой на пять строк.
    """
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"

    return ""


def first_doc_line(text: str | None) -> str:
    """Первая содержательная строка докстроки — та, что описывает смысл."""
    if not text:
        return ""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _node_start(node: ast.AST, lines: list[str]) -> int:
    """Первая строка определения с учётом декораторов и комментариев над ним."""
    start = node.lineno
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        # Декоратор — часть определения: @tool над функцией говорит о ней
        # больше, чем половина её тела.
        start = min(start, min(d.lineno for d in decorators))
    return _grab_comments_above(lines, start)


def chunk_python(text: str, source: str) -> list[CodeChunk]:
    """Режет модуль Python по определениям.

    Что получается на выходе:

      * **шапка модуля** — докстрока, импорты и константы до первого
        определения. Это фрагмент, который отвечает на вопросы «что этот
        модуль делает» и «от чего он зависит»;
      * **функция** — целиком, вместе с декораторами и комментарием над ней;
      * **класс** — целиком, если помещается; иначе шапка класса отдельно,
        а каждый метод отдельным фрагментом с именем `Класс.метод`;
      * **фрагмент** — всё, что между определениями: константы, которые
        объявили посередине файла, и блок `if __name__ == "__main__"`.
        Последний в этом курсе содержит REPL целиком, то есть половину
        интересного в главе.

    Синтаксическая ошибка не исключение, а обычное состояние файла, который
    прямо сейчас правят: в этом случае модуль уходит в построчную нарезку.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_lines(text, source, "python", note="файл не разобрался: синтаксическая ошибка")

    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    # Группируем тело модуля: подряд идущие НЕ-определения — один фрагмент.
    # Первая такая группа — шапка модуля, остальные — просто код между
    # определениями.
    pending: list[ast.stmt] = []
    seen_definition = False

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        start = _grab_comments_above(lines, pending[0].lineno)
        end = max(node.end_lineno or node.lineno for node in pending)
        chunk = CodeChunk(
            text=_line_span(lines, start, end),
            source=source,
            language="python",
            kind="block" if seen_definition else "module",
            name="" if seen_definition else Path(source).stem,
            docstring="" if seen_definition else first_doc_line(ast.get_docstring(tree)),
            start_line=start,
            end_line=end,
        )
        chunks.extend(_split_long(chunk, lines))
        pending = []

    for node in tree.body:
        if isinstance(node, definitions):
            flush()
            chunks.extend(_chunk_definition(node, lines, source))
            seen_definition = True
        else:
            pending.append(node)
    flush()

    return [chunk for chunk in chunks if _keep(chunk)]


def _chunk_definition(node: ast.stmt, lines: list[str], source: str) -> list[CodeChunk]:
    """Фрагменты одного определения верхнего уровня."""
    start = _node_start(node, lines)
    end = node.end_lineno or node.lineno

    if isinstance(node, ast.ClassDef):
        methods = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        fits = (end - start + 1) <= MAX_CHUNK_LINES and len(_line_span(lines, start, end)) <= MAX_CHUNK_CHARS

        # Маленький класс не разбирается на методы: пять коротких методов
        # по отдельности отвечают на вопросы хуже, чем класс целиком,
        # потому что каждый из них без соседей ничего не значит.
        if fits or not methods:
            return _split_long(
                CodeChunk(
                    text=_line_span(lines, start, end),
                    source=source,
                    language="python",
                    kind="class",
                    name=node.name,
                    signature=signature_of(node),
                    docstring=first_doc_line(ast.get_docstring(node)),
                    start_line=start,
                    end_line=end,
                ),
                lines,
            )

        chunks: list[CodeChunk] = []
        header_end = _node_start(methods[0], lines) - 1
        if header_end >= start:
            # Шапка класса: объявление, докстрока и поля до первого метода.
            chunks.extend(_split_long(
                CodeChunk(
                    text=_line_span(lines, start, header_end),
                    source=source,
                    language="python",
                    kind="class",
                    name=node.name,
                    signature=signature_of(node),
                    docstring=first_doc_line(ast.get_docstring(node)),
                    start_line=start,
                    end_line=header_end,
                ),
                lines,
            ))

        for method in methods:
            method_start = _node_start(method, lines)
            method_end = method.end_lineno or method.lineno
            chunks.extend(_split_long(
                CodeChunk(
                    text=_line_span(lines, method_start, method_end),
                    source=source,
                    language="python",
                    kind="method",
                    # Имя метода без класса бесполезно: search() есть
                    # в трёх классах курса, и все три разные.
                    name=f"{node.name}.{method.name}",
                    signature=signature_of(method),
                    docstring=first_doc_line(ast.get_docstring(method)),
                    start_line=method_start,
                    end_line=method_end,
                ),
                lines,
            ))
        return chunks

    return _split_long(
        CodeChunk(
            text=_line_span(lines, start, end),
            source=source,
            language="python",
            kind="function",
            name=node.name,
            signature=signature_of(node),
            docstring=first_doc_line(ast.get_docstring(node)),
            start_line=start,
            end_line=end,
        ),
        lines,
    )


# ====================================================================
# JAVASCRIPT / TYPESCRIPT: СКОБОЧНЫЙ СКАНЕР
# ====================================================================

# Заголовки определений верхнего уровня. Регулярное выражение — это не разбор
# языка, и здесь оно и не притворяется: заголовок ищется только в начале
# строки на нулевой глубине вложенности, где в осмысленно отформатированном
# файле и стоят определения.
JS_HEADERS = (
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([\w$]+)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([\w$]+)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(|function|[\w$]+\s*=>)"),
    re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+([\w$]+)"),
)

JS_KINDS = (
    (re.compile(r"\bclass\s"), "class"),
    # interface, type и enum — не классы: тела с кодом у них нет, и вопросы
    # к ним другие («какие поля у Todo»), поэтому и вид отдельный.
    (re.compile(r"\b(?:interface|type|enum)\s"), "type"),
    (re.compile(r"\bfunction\b|=>"), "function"),
)

# Строка документации JSDoc над определением — тот же материал, что докстрока
# в Python, только снаружи определения.
JSDOC_TEXT = re.compile(r"^\s*\*?\s*(.+?)\s*$")


def _scan_line(line: str, in_block_comment: bool, in_template: bool) -> tuple[int, bool, bool]:
    """Считает изменение глубины скобок в строке, пропуская строки и комментарии.

    Это и есть весь «разбор» JavaScript в этой главе — двадцать строк
    состояния. Он честно обрабатывает три случая, на которых ломается
    наивный подсчёт: `{` внутри строки, `{` внутри комментария и
    многострочный шаблон в обратных кавычках.

    Чего он НЕ умеет: скобки внутри регулярного выражения (`/[{]/`) и
    внутри JSX. Оба случая ломают глубину, и оба лечатся только настоящим
    парсером. Что происходит при поломке, описано у chunk_braces().
    """
    depth = 0
    index = 0
    in_string: str | None = None

    while index < len(line):
        char = line[index]
        pair = line[index:index + 2]

        if in_block_comment:
            if pair == "*/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if in_template:
            if char == "`":
                in_template = False
            elif char == "\\":
                index += 2
                continue
            index += 1
            continue

        if in_string:
            if char == in_string:
                in_string = None
            elif char == "\\":
                index += 2
                continue
            index += 1
            continue

        if pair == "//":
            break
        if pair == "/*":
            in_block_comment = True
            index += 2
            continue
        if char in "\"'":
            in_string = char
        elif char == "`":
            in_template = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1

    return depth, in_block_comment, in_template


def _js_docstring(lines: list[str], start: int) -> str:
    """Первая содержательная строка блока /** ... */ или строки // над определением."""
    line = start - 1
    collected: list[str] = []
    while line >= 1:
        stripped = lines[line - 1].strip()
        if stripped.startswith("//"):
            collected.append(stripped.lstrip("/ ").strip())
        elif stripped.endswith("*/") or stripped.startswith("*") or stripped.startswith("/*"):
            cleaned = stripped.strip("/*").strip()
            if cleaned:
                collected.append(cleaned)
        else:
            break
        line -= 1

    for candidate in reversed(collected):
        match = JSDOC_TEXT.match(candidate)
        if match and len(match.group(1)) > 3:
            return match.group(1)
    return ""


def _js_comment_start(lines: list[str], start: int) -> int:
    """Начало комментария над определением: и `//`, и `/** ... */`."""
    line = start - 1
    while line >= 1:
        stripped = lines[line - 1].strip()
        if stripped.startswith(("//", "*", "/*")) or stripped.endswith("*/"):
            line -= 1
            continue
        break
    return line + 1


def chunk_braces(text: str, source: str, language: str) -> list[CodeChunk]:
    """Режет JavaScript или TypeScript по определениям верхнего уровня.

    Разбора языка здесь нет и быть не может: единственный настоящий парсер
    JavaScript живёт внутри движка JavaScript, а ставить ради главы
    tree-sitter с бинарными колёсами — значит отменить обещание курса
    «на стандартной библиотеке».

    Поэтому сканер: заголовок определения ищется в начале строки на нулевой
    глубине, конец — там, где скобки снова сошлись в ноль. На нормально
    отформатированном файле это работает.

    Где он ошибается — честно, чтобы не выяснять это потом на своём коде:

      * **регулярное выражение со скобкой** (`/\\{/`) сдвигает глубину,
        и следующее определение приклеивается к предыдущему;
      * **JSX** — то же самое: `<div>{value}</div>` считается как блок;
      * **методы внутри класса** не выделяются в отдельные фрагменты,
        в отличие от Python: без разбора отличить метод от вызова функции
        со скобкой можно только гаданием. Длинный класс поэтому режется
        на части по строкам, а не по методам.

    Цена ошибки при этом ограничена: фрагмент получается больше нужного
    или с чужим хвостом, но индексация не падает и файл не теряется.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    depth = 0
    in_block_comment = False
    in_template = False
    index = 0
    pending_start: int | None = None  # начало «фрагмента между определениями»

    while index < len(lines):
        line = lines[index]
        number = index + 1
        header = None

        if depth == 0 and not in_block_comment and not in_template:
            for pattern in JS_HEADERS:
                match = pattern.match(line)
                if match:
                    header = match.group(1)
                    break

        if header is None:
            if line.strip() and pending_start is None:
                pending_start = number
            delta, in_block_comment, in_template = _scan_line(line, in_block_comment, in_template)
            depth = max(0, depth + delta)
            index += 1
            continue

        # Нашли определение: сначала закрываем накопившийся хвост перед ним.
        start = _js_comment_start(lines, number)
        if pending_start is not None and pending_start < start:
            chunks.extend(_split_long(
                CodeChunk(
                    text=_line_span(lines, pending_start, start - 1),
                    source=source,
                    language=language,
                    kind="block",
                    start_line=pending_start,
                    end_line=start - 1,
                ),
                lines,
            ))
        pending_start = None

        # Тело определения: до строки, на которой скобки снова сошлись.
        # Определение без тела (`type X = ...`) заканчивается на своей же
        # строке — тогда цикл выходит сразу.
        end = number
        opened = False
        while end <= len(lines):
            delta, in_block_comment, in_template = _scan_line(
                lines[end - 1], in_block_comment, in_template
            )
            depth += delta
            if delta > 0:
                opened = True
            if depth <= 0:
                depth = 0
                break
            end += 1
        else:
            end = len(lines)

        if not opened:
            # Однострочное объявление: `export type TodoFilter = "all" | "done";`
            end = number

        kind = "function"
        for pattern, guess in JS_KINDS:
            if pattern.search(line):
                kind = guess
                break

        chunks.extend(_split_long(
            CodeChunk(
                text=_line_span(lines, start, end),
                source=source,
                language=language,
                kind=kind,
                name=header,
                signature=line.strip().rstrip("{").strip(),
                docstring=_js_docstring(lines, number),
                start_line=start,
                end_line=end,
            ),
            lines,
        ))
        index = end

    if pending_start is not None and pending_start <= len(lines):
        chunks.extend(_split_long(
            CodeChunk(
                text=_line_span(lines, pending_start, len(lines)),
                source=source,
                language=language,
                kind="block",
                start_line=pending_start,
                end_line=len(lines),
            ),
            lines,
        ))

    return [chunk for chunk in chunks if _keep(chunk)]


# ====================================================================
# ФОЛЛБЭК: ПОСТРОЧНЫЕ ОКНА
# ====================================================================

def chunk_lines(
    text: str,
    source: str,
    language: str,
    window: int = WINDOW_LINES,
    overlap: int = OVERLAP_LINES,
    note: str = "",
) -> list[CodeChunk]:
    """Режет файл окнами по строкам с перекрытием.

    Способ грубый, и применяется он там, где ничего лучше нет: конфиги,
    незнакомые языки, файл с синтаксической ошибкой. Границы окон не значат
    ничего, зато адрес у фрагмента честный, а перекрытие спасает определение,
    разрезанное границей окна.
    """
    lines = text.splitlines()
    if not lines:
        return []

    spans: list[tuple[int, int]] = []
    start = 1
    while start <= len(lines):
        end = min(start + window - 1, len(lines))
        spans.append((start, end))
        if end == len(lines):
            break
        start = end + 1 - overlap

    chunks = [
        CodeChunk(
            text=_line_span(lines, first, last),
            source=source,
            language=language,
            kind="block",
            name=Path(source).name,
            docstring=note,
            start_line=first,
            end_line=last,
            part=number,
            parts=len(spans),
        )
        for number, (first, last) in enumerate(spans, 1)
    ]
    return [chunk for chunk in chunks if _keep(chunk)]


# ====================================================================
# MARKDOWN
# ====================================================================

def chunk_markdown(text: str, source: str) -> list[CodeChunk]:
    """Режет документацию нарезкой Главы 4 — своей заводить незачем.

    README рядом с кодом объясняет код, и выбрасывать его из индекса было
    бы странно. Но проза остаётся прозой: она режется по разделам и абзацам,
    ровно как в Главе 4, и попадает в тот же индекс кода. Номеров строк
    у таких фрагментов нет — вместо адреса у них путь заголовков.

    Индексацию документации можно выключить (см. languages и AGENT_CODE_DOCS
    в codebase.py): именно так замеряется, мешает ли проза коду в одном
    индексе — тот самый вопрос, с которого началась эта глава.
    """
    return [
        CodeChunk(
            text=chunk.text,
            source=source,
            language="markdown",
            kind="section",
            name=chunk.heading,
            docstring="",
            start_line=0,
            end_line=0,
        )
        for chunk in chunk_text(text, source, markdown=source.lower().endswith(".md"))
    ]


# ====================================================================
# ТОЧКА ВХОДА
# ====================================================================

def chunk_code(text: str, source: str, language: str) -> list[CodeChunk]:
    """Выбирает способ нарезки по языку файла."""
    if language == "python":
        return chunk_python(text, source)
    if language in ("javascript", "typescript"):
        return chunk_braces(text, source, language)
    if language == "markdown":
        return chunk_markdown(text, source)
    return chunk_lines(text, source, language)


def chunk_source(path: Path | str, root: Path | str | None = None) -> list[CodeChunk]:
    """Читает файл и режет его. Источник — путь относительно корня репозитория."""
    path = Path(path)
    language = language_of(path)
    if not language:
        return []

    text = read_source(path)
    if text is None:
        return []

    if root:
        try:
            source = str(path.relative_to(root))
        except ValueError:
            source = path.name
    else:
        source = path.name

    return chunk_code(text, source.replace("\\", "/"), language)
