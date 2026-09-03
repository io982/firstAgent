"""
Карта кода рабочего каталога: какие функции есть и что каждая делает.

Зачем она агенту. Правку надо куда-то положить, и сейчас место ищется
текстовым поиском: по слову из задачи или по последнему изменённому
файлу. Работает, пока человек цитирует свой код; перестаёт, как только
он говорит по-человечески — «сделай чтобы приложение выводило ещё
и производную». Тогда искать нечего, и агент либо угадывает файл,
либо заводит новый рядом с тем, который просили поправить.

Карта отвечает на этот вопрос иначе: перед правкой модель видит СПИСОК
функций проекта с назначением каждой и выбирает одну. Выбор из списка —
это выбор из `enum`, и его можно навязать схемой: назвать функцию,
которой нет, модель тогда физически не сможет. Ровно тот вид работы,
на котором глава уже выигрывала дважды — схема правки и имя файла.

Что здесь считает КОД, а что модель:

  * имена, аргументы, вид и диапазон строк — разбор `ast`, без модели.
    Это факты о файле, и спрашивать их у модели значит менять точный
    ответ на правдоподобный;
  * назначение — из докстринга, если он есть. Автор уже написал, что
    делает функция, и переспрашивать это у модели — трата запроса;
  * назначение того, у чего докстринга нет, — единственное место,
    где нужна модель. Такие функции пишет сам агент: в его коде
    докстрингов обычно нет.

Диапазон строк здесь не украшение. Он даёт форму правки, которой у нас
до сих пор не было: заменить функцию целиком по её собственным
границам. Не надо ни цитировать якорь (на нём модель промахивается),
ни перепечатывать файл (на этом теряется чужой код).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from chapter1.agent import request_model
from chapter3.src.memory import LongTermMemory
from chapter8.src import guard

# Файл, где хранятся назначения, добытые у модели. Отдельно от памяти
# сессии: там три ключа про то, где мы работаем, а здесь их столько,
# сколько в проекте функций, и смешивать эти два хранилища значит
# сделать нечитаемым первое.
DEFAULT_MAP_PATH = Path(__file__).parent.parent / "codemap.json"

# Сколько функций показывать в одном запросе к модели. Предел не про
# токены, а про качество: список из полусотни имён модель на 3B читает
# невнимательно, и выбор из него становится случайным.
MAP_LIMIT = 40

# Сколько функций описывать одним запросом. Один запрос на функцию —
# это минута на средний файл; одним запросом на всех модель отвечает
# хуже. Десяток — компромисс, проверенный тем же способом, что и всё
# остальное в главе: замером, а не рассуждением.
DESCRIBE_BATCH = 10


@dataclass
class Definition:
    """Одно определение из файла проекта."""

    path: str
    name: str          # `solve` или `Cart.add` — с классом, если это метод
    args: str
    start: int         # с первой строки декоратора, а не с `def`
    end: int
    kind: str          # «функция», «метод», «класс»
    purpose: str = ""  # из докстринга или от модели

    @property
    def where(self) -> str:
        """Адрес в том же виде, в каком его печатает поиск Главы 5."""
        return f"{self.path}:{self.start}-{self.end}"

    def line(self) -> str:
        """Одна строка для списка в промпте."""
        signature = f"{self.name}({self.args})" if self.kind != "класс" else self.name
        tail = f" — {self.purpose}" if self.purpose else ""
        return f"{self.path}:{self.start} {signature}{tail}"


def state_path() -> Path:
    """Где лежат добытые у модели назначения. Переменная сильнее умолчания.

    Та же причина, что у памяти сессии: хранилище глобальное, и без
    подмены пути тесты читали бы файл разработчика.
    """
    return Path(os.environ.get("AGENT_CODEMAP_FILE") or DEFAULT_MAP_PATH)


def definitions_of(path: str, text: str) -> list[Definition]:
    """Определения одного файла — разбором, без модели.

    Методы получают имя с классом (`Cart.add`), потому что `add`
    в проекте бывает не один, а выбирать модель будет по имени.
    Вложенные функции пропускаются: заменять замыкание отдельно
    от той функции, внутри которой оно живёт, бессмысленно.
    """
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    found: list[Definition] = []

    def add(node, name: str, kind: str) -> None:
        # Декораторы входят в диапазон: замена функции по строкам
        # без них оставила бы `@tool` висеть над чужим определением.
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        args = ", ".join(a.arg for a in node.args.args) if hasattr(node, "args") else ""
        found.append(Definition(
            path=path, name=name, args=args, start=start,
            end=getattr(node, "end_lineno", start), kind=kind,
            purpose=_first_phrase(ast.get_docstring(node) or ""),
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, "функция")
        elif isinstance(node, ast.ClassDef):
            add(node, node.name, "класс")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, f"{node.name}.{item.name}", "метод")
    return found


# Длина назначения в карте. Карта уезжает в промпт целиком, и полсотни
# развёрнутых описаний вытеснят из окна саму задачу.
PURPOSE_LIMIT = 90


def _first_phrase(docstring: str) -> str:
    """Первая фраза докстринга — остальное в список не помещается."""
    return _shorten(docstring.strip().split("\n", 1)[0].strip())


def _shorten(text: str) -> str:
    """Обрезает по границе слова, а не по счётчику символов."""
    text = " ".join(text.split())
    if len(text) <= PURPOSE_LIMIT:
        return text
    cut = text[:PURPOSE_LIMIT].rsplit(" ", 1)[0]
    return (cut or text[:PURPOSE_LIMIT]).rstrip(" ,.;:") + "…"


def tidy_purpose(name: str, text: str) -> str:
    """Снимает у ответа модели начало «Функция solve(a, b, c) ...».

    Правило «без слова „функция“ в начале» в промпте есть, и модель его
    нарушает: описания в её обучающих данных начинаются именно так.
    Снять начало кодом дешевле, чем спорить промптом, — тот же приём,
    что `_strip_fences` для тройных кавычек.
    """
    phrase = " ".join(str(text).split())
    lowered = phrase.lower()
    for prefix in ("функция ", "метод ", "класс "):
        if lowered.startswith(prefix):
            phrase = phrase[len(prefix):]
            lowered = phrase.lower()
    # Имя снимается только целым словом. Без проверки границы описание
    # метода `add` со словом «Adds» превращалось в «S an item to the
    # cart»: начало съедалось по буквам, а не по слову.
    bare = name.rsplit(".", 1)[-1].lower()
    rest = phrase[len(bare):]
    if lowered.startswith(bare) and not rest[:1].isalnum() and rest[:1] != "_":
        if rest.startswith("("):
            rest = rest.split(")", 1)[-1]
        phrase = rest.lstrip(" —-:,")
    return _shorten(phrase[:1].upper() + phrase[1:] if phrase else "")


def scan() -> list[Definition]:
    """Все определения рабочего каталога. Пересобирается по времени файла.

    Кэш по паре «путь, время изменения»: разбор дешёвый, но карта нужна
    перед каждой правкой, а каталог курса — это полтора десятка тысяч
    строк. Файл изменился — пересчитывается только он.
    """
    root = guard.get_workspace()
    found: list[Definition] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            continue
        name = guard.relative(path)
        cached = _CACHE.get(name)
        if cached and cached[0] == stamp:
            found.extend(cached[1])
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        parsed = definitions_of(name, text)
        _fill_purposes(parsed, text)
        _CACHE[name] = (stamp, parsed)
        found.extend(parsed)
    return found


_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules"})
_CACHE: dict[str, tuple[int, list[Definition]]] = {}


def forget_cache() -> None:
    """Забыть разобранное. Нужно тестам и смене рабочего каталога."""
    _CACHE.clear()


def find(name: str) -> Definition | None:
    """Определение по имени. Сначала точное совпадение, потом по концу.

    Совпадение по концу нужно для методов: модель называет `add`,
    а в карте лежит `Cart.add`. Обратное — назвать `Cart.add`, когда
    в карте `add`, — тоже бывает, и оно тоже ловится.
    """
    wanted = (name or "").strip()
    if not wanted:
        return None
    everything = scan()
    for item in everything:
        if item.name == wanted:
            return item
    for item in everything:
        if item.name.endswith("." + wanted) or wanted.endswith("." + item.name):
            return item
    return None


def render(limit: int = MAP_LIMIT) -> str:
    """Карта строками — то, что уезжает в промпт выбора места."""
    items = scan()[:limit]
    if not items:
        return "В проекте нет ни одной функции."
    return "\n".join(item.line() for item in items)


def names(limit: int = MAP_LIMIT) -> list[str]:
    """Имена определений — для `enum` в схеме ответа.

    Ради этого списка карта и заводилась: выбор из перечисления модель
    делает заметно надёжнее, чем сочинение имени, а невозможный ответ
    отсекается грамматикой, а не проверкой после.
    """
    return [item.name for item in scan()[:limit]]


# --------------------------------------------------------------------
# НАЗНАЧЕНИЕ: ДОКСТРИНГ, ПОТОМ ХРАНИЛИЩЕ, ПОТОМ МОДЕЛЬ
# --------------------------------------------------------------------

def _body_key(path: str, name: str, text: str, item: Definition) -> str:
    """Ключ назначения: путь, имя и отпечаток ТЕЛА функции.

    Отпечаток обязателен. Назначение живёт дольше кода, а функция
    меняется — и описание «складывает два числа» после правки может
    относиться к чему угодно. Тело изменилось, значит ключ другой,
    значит старое описание не подставится.
    """
    lines = text.splitlines()[item.start - 1: item.end]
    digest = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]
    return f"{path}:{name}:{digest}"


def _store() -> LongTermMemory:
    return LongTermMemory(state_path())


def _fill_purposes(items: list[Definition], text: str) -> None:
    """Подставляет назначения, добытые у модели раньше. Модель не зовётся."""
    if all(item.purpose for item in items):
        return
    memory = _store()
    for item in items:
        if item.purpose:
            continue
        answer = memory.recall(_body_key(item.path, item.name, text, item))
        if answer.startswith("📖"):
            item.purpose = answer.split(" = ", 1)[-1].strip()


def purpose_schema(wanted: list[str]) -> dict:
    """Схема ответа «что делает каждая из этих функций».

    Ответ — ПАРЫ «имя, назначение», и имя берётся из `enum`. Первая
    версия просила список строк в том же порядке, что и вопрос, и имён
    не спрашивала вовсе — рассуждение было такое: не давать модели
    возможности переврать имя. Живой прогон показал, чем это кончается:
    на паре `get_roots, main` модель вернула два описания, и второе
    описывало первую функцию. Перепутанный порядок не отличить
    от правильного ничем, а перепутанное имя — отличить нечем, потому
    что `enum` его не пропустит.
    """
    return {
        "type": "object",
        "properties": {
            "purposes": {
                "type": "array",
                "minItems": len(wanted),
                "maxItems": len(wanted),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": wanted},
                        "purpose": {"type": "string"},
                    },
                    "required": ["name", "purpose"],
                },
            }
        },
        "required": ["purposes"],
    }


DESCRIBE_RULES = """Ты описываешь, что делает каждая функция. Кода не пишешь.

Правила:
1. На каждую функцию — пара «имя, назначение». Имя бери из списка как есть.
2. Одна короткая фраза на функцию, без слова «функция» в начале.
3. Пиши, ЧТО функция делает и что возвращает, а не как она устроена.
4. Описывай ИМЕННО ту функцию, чьё имя стоит рядом, а не соседнюю.
5. Не знаешь — напиши «назначение неясно». Придумывать не надо."""


def describe(items: list[Definition], model: str | None = None) -> int:
    """Спрашивает у модели назначение функций без докстринга. Возвращает число описанных.

    Зовётся отдельно, а не из `scan()`, и это важно: карта нужна перед
    каждой правкой, а запрос к модели стоит секунд. Разбор бесплатный
    и всегда свежий, описание — дорогое и потому кэшируется на диске
    по отпечатку тела функции.
    """
    todo = [item for item in items if not item.purpose][:DESCRIBE_BATCH]
    if not todo:
        return 0

    listing = "\n".join(
        f"{index + 1}. {item.path}: {item.name}({item.args})" for index, item in enumerate(todo)
    )
    bodies = "\n\n".join(_source_of(item) for item in todo)
    messages = [
        {"role": "system", "content": DESCRIBE_RULES},
        {"role": "user", "content": f"Функции:\n{listing}\n\nИсходный код:\n{bodies}"},
    ]

    try:
        raw = request_model(messages, response_format=purpose_schema([item.name for item in todo]))
        answers = json.loads(raw).get("purposes", [])
    except Exception:  # noqa: BLE001
        return 0

    # Сопоставление ПО ИМЕНИ, а не по порядку. Лишнее и повторное имя
    # отбрасывается: описание, приехавшее дважды, означает, что модель
    # спутала функции, и второе такое же неверно, как первое.
    by_name = {}
    for answer in answers:
        name = str(answer.get("name", ""))
        if name and name not in by_name:
            by_name[name] = str(answer.get("purpose", ""))

    memory = _store()
    described = 0
    for item in todo:
        phrase = tidy_purpose(item.name, by_name.get(item.name, ""))
        if not phrase:
            continue
        item.purpose = phrase
        text = _text_of(item.path)
        if text:
            memory.remember(_body_key(item.path, item.name, text, item), phrase)
        described += 1
    return described


def _text_of(path: str) -> str:
    try:
        return guard.resolve_path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, guard.OutsideWorkspace):
        return ""


def source_of(item: Definition) -> str:
    """Текст самого определения — то, что уезжает в правку вместо файла."""
    text = _text_of(item.path)
    if not text:
        return ""
    return "\n".join(text.splitlines()[item.start - 1: item.end])


def _source_of(item: Definition) -> str:
    """То же, но с заголовком: в запросе описывается несколько функций сразу."""
    body = source_of(item)
    return f"--- {item.path}: {item.name} ---\n{body}" if body else ""
