"""
Карточка фрагмента: мост между вопросом по-русски и кодом (пункт 5.3).

Вторая половина долга Главы 4. Нарезка по определениям чинит границы
фрагментов, но не чинит главного: вопрос «где реализован калькулятор»
и текст `def calculator(expression: str)` написаны на разных языках,
и близость между ними низкая просто потому, что общих слов нет.

Материал для моста лежит в самом коде, и его больше, чем кажется:

    имя         calculator, chunkTextWithLines, KnowledgeBase.build_context
    сигнатура   какие аргументы принимает и что возвращает
    докстрока   зачем эта функция нужна — уже человеческим языком
    путь        chapter4/src/knowledge.py — тема файла тоже часть ответа

Карточка собирает это в связный текст и приклеивается к коду ПЕРЕД
эмбеддингом. При этом пользователю возвращается по-прежнему код: то, что
мы кодируем, и то, что показываем, — разные тексты, и векторная база это
позволяет (вектор и документ хранятся отдельно, см. VectorStore.add
Главы 4).

Отдельно про имена. `chunkTextWithLines` для модели эмбеддингов — одно
незнакомое слово; `chunk text with lines` — четыре знакомых. Разбор
идентификатора на слова стоит десять строк кода и добавляет в карточку
единственное, что связывает английский код с английскими же терминами
в русском вопросе («чанк», «токен», «эмбеддинг» читатель напишет
по-русски, а вот `search`, `index`, `budget` — скорее по-английски).
"""

import os
import re

from .codechunks import CodeChunk

# ЧТО ИМЕННО УХОДИТ В МОДЕЛЬ ЭМБЕДДИНГОВ. Три режима, и разница между ними
# — главный замер главы:
#
#   card+code   карточка плюс код (по умолчанию)
#   card        ТОЛЬКО карточка: имя, сигнатура, докстрока, слова
#   code        только код, без карточки — то, с чего глава начиналась
#
# Меняется переменной окружения:
#   PowerShell:   $env:AGENT_CODE_EMBED = "card"
#   Linux/macOS:  export AGENT_CODE_EMBED=card
#
# Значение читается один раз при импорте, как и остальные переключатели
# курса. Тесты меняют его через monkeypatch модуля.
EMBED_MODES = ("card+code", "card", "code")
EMBED_MODE = os.environ.get("AGENT_CODE_EMBED", "card+code")

# Слова короче двух букв выбрасываются: `i`, `n`, `x` не помогают найти
# ничего, а в карточке каждой второй функции они есть.
MIN_WORD = 2

# Сколько слов оставляем в карточке. Ограничение нужно из-за длинных
# сигнатур: перечисление двадцати аргументов размывает вектор фрагмента
# ровно так же, как размывал бы лишний абзац.
MAX_WORDS = 12

# Служебные имена, которые есть в каждом втором файле и не значат ничего:
# по ним ищут только тогда, когда спрашивают именно про них, а в карточке
# они добавляют шум.
STOP_WORDS = frozenset({
    "self", "cls", "args", "kwargs", "init", "main", "src", "py", "js", "ts",
    "str", "int", "bool", "list", "dict", "none", "true", "false", "return",
})

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Разделитель — всё, что не буква и не цифра, И ПОДЧЁРКИВАНИЕ ТОЖЕ.
# `\w` в Python включает `_`, и с ним snake_case не разбирается вовсе:
# `chunk_text_with_lines` остаётся одним словом, то есть карточка теряет
# ровно то, ради чего она нужна.
_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)
_PARAM_NAMES = re.compile(r"[\w$]+")


def split_identifier(name: str) -> list[str]:
    """Разбирает идентификатор на слова: и snake_case, и camelCase.

        split_identifier("chunk_text_with_lines")  → chunk, text, with, lines
        split_identifier("KnowledgeBase.search")   → knowledge, base, search
        split_identifier("HTTPResponseCode")       → http, response, code

    Третий пример объясняет, почему регулярное выражение выглядит сложнее
    ожидаемого: аббревиатура из заглавных букв — не одно слово и не пять,
    граница проходит перед последней заглавной, за которой идёт строчная.
    """
    if not name:
        return []

    words: list[str] = []
    for part in _SEPARATORS.split(name):
        for piece in _CAMEL.split(part):
            piece = piece.strip().lower()
            if len(piece) >= MIN_WORD and piece not in STOP_WORDS and not piece.isdigit():
                words.append(piece)

    # Порядок сохраняем, повторы убираем: `chunk_text` в файле `chunking.py`
    # не должен давать «chunk» дважды.
    seen: set[str] = set()
    unique = [word for word in words if not (word in seen or seen.add(word))]
    return unique[:MAX_WORDS]


def _parameter_words(signature: str) -> list[str]:
    """Слова из имён аргументов: `budget_tokens` → budget, tokens.

    Типы и значения по умолчанию отбрасываются — `int`, `None` и `False`
    одинаковы у половины функций проекта.
    """
    inside = signature[signature.find("(") + 1: signature.rfind(")")] if "(" in signature else ""
    words: list[str] = []
    for token in _PARAM_NAMES.findall(inside):
        words.extend(split_identifier(token))
    return words


def build_card(chunk: CodeChunk) -> str:
    """Собирает карточку фрагмента — то, чего в коде нет по-русски.

    Порядок строк не случаен: сначала адрес и вид («функция calculator
    в файле chapter1/agent.py»), потом сигнатура, потом объяснение автора,
    и только в конце — россыпь слов. Первые строки читаются как связный
    текст, а модель эмбеддингов к связному тексту чувствительнее, чем
    к списку ключевых слов.

    Фрагментам документации карточка не нужна: нарезка Главы 4 уже кладёт
    внутрь чанка «хлебную крошку» с именем файла и путём заголовков.
    """
    if chunk.kind == "section":
        return ""

    where = f"{chunk.source}"
    title = chunk.title()
    lines = [f"{title} в файле {where}"]

    if chunk.signature:
        lines.append(chunk.signature)
    if chunk.docstring:
        lines.append(chunk.docstring)
    if chunk.parts > 1:
        # Часть длинной функции без этой строки выглядит самостоятельным
        # фрагментом, который просто начинается с середины.
        lines.append(f"часть {chunk.part} из {chunk.parts}")

    words = split_identifier(chunk.name) + _parameter_words(chunk.signature)
    words += split_identifier(chunk.source.rsplit("/", 1)[-1].rsplit(".", 1)[0])

    seen: set[str] = set()
    unique = [word for word in words if not (word in seen or seen.add(word))]
    if unique:
        lines.append("ключевые слова: " + ", ".join(unique[:MAX_WORDS]))

    return "\n".join(lines)


def embedding_text(chunk: CodeChunk, mode: str | None = None) -> str:
    """Текст, который реально уходит в модель эмбеддингов.

    Это единственное место, где карточка встречается с кодом. Всё остальное
    — индекс, поиск, выдача — работает с `chunk.text`, то есть с настоящим
    кодом: пользователь должен видеть файл, а не наши приписки к нему.

    Режим `card` (карточка без кода) — не экзотика, а рабочий приём: вектор
    считается по описанию фрагмента, а не по его тексту. Сорок строк
    английских идентификаторов размывают вектор, и единственная строка
    русской докстроки в нём тонет — а найти фрагмент по-русски мы хотим
    именно по ней.
    """
    mode = (mode or EMBED_MODE).lower()
    if mode == "code":
        return chunk.text

    card = build_card(chunk)
    if not card:
        # У документации карточки нет: «хлебную крошку» кладёт внутрь
        # текста ещё нарезка Главы 4.
        return chunk.text
    if mode == "card":
        return card
    return f"{card}\n\n{chunk.text}"
