"""
Файловые инструменты кодинг-агента: посмотреть, найти, прочитать, изменить (8.1).

Семь инструментов, и делятся они не поровну. Три первых только читают
и потому ничего не спрашивают. Четыре последних меняют диск и потому
целиком проходят через `guard`: рабочий каталог, сухой прогон,
подтверждение, запись в журнал для отката.

Разделение проведено здесь, а не в каждом инструменте по отдельности,
потому что забыть проверку легче всего в четвёртом однотипном месте.
Вся запись идёт через одну функцию `_commit()` — если проверка есть в ней,
она есть везде.

Что отличает эти инструменты от `read_file` Главы 2:

  * **номера строк.** Глава 2 отдавала текст как есть. Здесь каждая строка
    печатается как `42| def foo():` — и это не украшение: `replace_lines`
    принимает ровно такие номера, а поиск по коду Главы 5 отвечает ровно
    таким адресом. Один и тот же способ назвать место работает во всей
    цепочке «нашёл — прочитал — поправил»;
  * **границы.** Путь проходит через `resolve_path`, и прочитать файл
    из соседнего каталога больше нельзя;
  * **потолок вывода.** Всё, что вернёт инструмент, уедет в контекст
    модели, а он 4096 токенов на весь разговор.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

from chapter2.src.tools import tool
from chapter8.src import guard
from chapter8.src.edits import (
    apply_anchor,
    apply_append,
    apply_full,
    apply_lines,
    doubled_main,
    duplicate_definitions,
    lost_definitions,
    stray_definitions,
    syntax_ok,
    unified,
    unreachable_code,
)
from chapter8.src.shell import undefined_in_text

# Потолки вывода. В символах и штуках, а не в токенах: считать токены
# на каждый чих дорого, а связь между ними линейная — потолок в символах
# держит и потолок в токенах.
OUTPUT_LIMIT = 2000     # символов на один ответ инструмента
MAX_LINES = 200         # строк за одно чтение
SEARCH_HITS = 30        # совпадений в выдаче поиска
LIST_ENTRIES = 60       # записей в листинге каталога

# Каталоги, в которые агент не заходит. Не запрет, а гигиена: в `.git`
# и `__pycache__` лежат не исходники, а их производные, и попадание их
# в контекст стоит места и не даёт ничего.
SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", "chroma_db", ".pytest_cache", ".ruff_cache"}
)


def _int(value: str | int, default: int) -> int:
    """Число из аргумента инструмента.

    Нужно потому, что схема реестра Главы 2 объявляет все параметры
    строками, и `start` приходит как "42", а иногда как "строка 42".
    Падать на этом инструмент не должен: непонятное значение —
    это значение по умолчанию, а не исключение в середине правки.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Обрезает вывод и говорит об этом вслух.

    Вслух — по той же причине, что и в Главе 2: молча укороченный вывод
    модель считает полным и уверенно отвечает по половине данных.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...обрезано: показаны первые {limit} символов]"


def _read_text(path: Path) -> str | None:
    """Текст файла или None, если он не текстовый."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _commit(path: Path, before: str, after: str, action: str, replace: bool = False) -> str:
    """Единственное место в главе, где содержимое файла попадает на диск.

    Порядок здесь важнее самой записи:

      1. проверка синтаксиса — сломанный файл не пишем вообще;
      2. проверка на пропавшие определения;
      3. подтверждение с показом diff — человек видит, что подтверждает;
      4. запись в журнал ДО записи на диск — иначе откатывать будет не к чему;
      5. и только потом запись.

    `replace=True` означает «человек просил написать этот файл заново».
    Тогда пропавшие определения не отменяют запись, а уезжают в текст
    подтверждения: это не промах номера строки, а то, о чём просили.
    Разница нужна с тех пор, как повтор задачи «сделай calc.py» стал
    законной перезаписью, а не правкой.
    """
    ok, problem = syntax_ok(path.name, after)
    if not ok:
        return (
            f"Правка отменена: после неё файл перестаёт разбираться ({problem}). "
            "Файл на диске не тронут."
        )

    stray = stray_definitions(path.name, before, after)
    if stray and not replace:
        return (
            f"Правка отменена: определения {', '.join(stray)} оказались ВНУТРИ чужого блока "
            "(if/try/for/while), а не на своём уровне. Скорее всего сбит отступ: "
            "в конце файла лежит вложенный блок, и дописанное приклеилось к нему. "
            "Поставьте определение с нулевым отступом. Файл на диске не тронут."
        )

    lost = lost_definitions(path.name, before, after)
    warning = ""
    if lost and not replace:
        return (
            f"Правка отменена: из файла пропали определения — {', '.join(lost)}. "
            "Однострочная правка не удаляет функции; скорее всего сбит номер строки "
            "или файл перезаписан не целиком. Файл на диске не тронут."
        )
    if lost:
        warning = f"\n⚠️ Из файла пропадут определения: {', '.join(lost)}\n"

    if doubled_main(path.name, before, after) and not replace:
        return (
            "Правка отменена: в файле появился второй блок "
            "`if __name__ == \"__main__\"`. Он бывает ровно один — иначе "
            "программа спросит у человека одно и то же дважды. "
            "Файл на диске не тронут."
        )

    # Неопределённое имя ловится ПРИ ЗАПИСИ, а не после прогона.
    # Живой прогон: правка вписала `print(2*a*x + b)` в функцию
    # `solve_quadratic(a, b, c)`, где никакого `x` нет. Поймалось это
    # запуском в конце плана — то есть после двух кругов починки
    # и отката всей работы, — а линтер знал об этом сразу.
    #
    # Считаются только НОВЫЕ имена: файл мог приехать к нам уже
    # с неопределённым именем, и отменять из-за этого правку значит
    # запретить чинить как раз то, что сломано.
    # Линтер спрашивается о НОВОМ тексте, и только если он что-то нашёл —
    # о старом. Порядок не для красоты: чистых правок подавляющее
    # большинство, а каждый вызов линтера это отдельный процесс.
    found = undefined_in_text(path.name, after) if not replace else []
    was_undefined = set(undefined_in_text(path.name, before)) if found else set()
    now_undefined = [name for name in found if name not in was_undefined]
    if now_undefined:
        return (
            f"Правка отменена: имена {', '.join(now_undefined)} нигде не определены — "
            "программа упадёт NameError на первой же строке, где они встретятся. "
            "Пользуйтесь тем, что функции доступно: её аргументами и именами файла. "
            "Файл на диске не тронут."
        )

    twice = duplicate_definitions(path.name, before, after)
    if twice and not replace:
        return (
            f"Правка отменена: определения {', '.join(twice)} после неё оказались в файле "
            "дважды. Менять надо СУЩЕСТВУЮЩЕЕ определение, а не дописывать рядом второе. "
            "Файл на диске не тронут."
        )

    dead = unreachable_code(path.name, before, after)
    if dead and not replace:
        where = "; ".join(dead[:3])
        return (
            f"Правка отменена: после неё код «{where}» стоит за return или raise — "
            "до него выполнение не дойдёт никогда. Поставьте его ПЕРЕД выходом "
            "из функции. Файл на диске не тронут."
        )

    diff = unified(before, after, guard.relative(path))
    verdict = guard.check(action, warning + (diff or "изменений в тексте нет"))
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, action)

    guard.record(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(after, encoding="utf-8")
    return ""


def put_file(path: str, content: str, replace: bool = False) -> str:
    """Записывает файл целиком: создаёт или перезаписывает.

    Не инструмент, а внутренняя функция, и разница существенная.
    Инструмент `write_file` работает по строгим правилам: потерять
    определения он не даёт. Шаг плана `create` — это прямое «напиши
    вот этот файл», подтверждённое человеком по имени, и там замена
    содержимого целиком законна. Разводить эти два случая флагом
    у инструмента нельзя: параметры инструмента видит модель, и флаг
    «можно терять код» в её распоряжении — плохая идея.
    """
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)

    before = _read_text(target) if target.is_file() else None
    if target.exists() and before is None:
        return f"Файл не читается как текст, перезапись отменена: {guard.relative(target)}"

    if before is None:
        # Новый файл — сравнивать не с чем, и apply_full() здесь не нужен:
        # у него вся работа в сравнении со старым текстом.
        ok, problem = syntax_ok(target.name, content)
        if not ok:
            return f"Файл не создан: он не разбирается ({problem})."
        verdict = guard.check(
            f"создать {guard.relative(target)}",
            f"новый файл, {len(content.splitlines())} строк",
        )
        if verdict != guard.ALLOW:
            return guard.verdict_message(verdict, f"создать {guard.relative(target)}")
        guard.record(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"{guard.relative(target)}: файл создан ({len(content.splitlines())} строк)."

    result = apply_full(before, content)
    if not result.ok:
        return f"Правка не применена. {result.message}"

    action = f"переписать {guard.relative(target)} заново" if replace else f"перезаписать {guard.relative(target)}"
    problem = _commit(target, before, result.text, action, replace=replace)
    return problem or f"{guard.relative(target)}: {result.message}"


# ====================================================================
# ЧТЕНИЕ
# ====================================================================

@tool
def list_dir(path: str = ".") -> str:
    """Показывает файлы и папки внутри рабочего каталога."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.exists():
        return f"Нет такого пути: {guard.relative(target)}"
    if target.is_file():
        return f"Это файл, а не папка: {guard.relative(target)} ({target.stat().st_size} байт)"

    dirs, files = [], []
    for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir():
            dirs.append(f"{entry.name}/")
        else:
            files.append(f"{entry.name}  ({entry.stat().st_size} байт)")

    rows = dirs + files
    if not rows:
        return f"{guard.relative(target)}: пусто"
    shown = rows[:LIST_ENTRIES]
    out = f"{guard.relative(target)}:\n" + "\n".join(f"  {row}" for row in shown)
    if len(rows) > LIST_ENTRIES:
        out += f"\n  [...ещё {len(rows) - LIST_ENTRIES} записей]"
    return _clip(out)


@tool
def read_lines(path: str, start: str = "1", end: str = "") -> str:
    """Читает файл с номерами строк — в том же виде, в каком их принимает правка."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"

    text = _read_text(target)
    if text is None:
        return f"Файл не читается как текст: {guard.relative(target)}"

    lines = text.splitlines()
    total = len(lines)
    first = max(1, _int(start, 1))
    # Пустой `end` значит «до конца», но не дальше потолка. Потолок здесь
    # ограничивает не файл, а контекст: файл в 3000 строк выдавить из
    # контекста разговор может, а пользы от такого чтения нет.
    last = _int(end, 0) or min(total, first + MAX_LINES - 1)
    last = min(last, total, first + MAX_LINES - 1)

    if first > total:
        return f"В файле {guard.relative(target)} всего {total} строк, а запрошена {first}."
    # Конец раньше начала — это ошибка вызова, и молчать о ней нельзя.
    # Прежде такой запрос отвечал заголовком «a.py:3-1» и пустым телом:
    # модель получала бессмысленный ответ, из которого не следует,
    # что делать дальше. `apply_lines` про то же говорит прямо, и чтение
    # должно вести себя так же — иначе агент чинит не ту беду.
    if last < first:
        return f"Конец раньше начала: запрошены строки {first}-{last}."

    width = len(str(last))
    body = "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(first, last + 1))
    head = f"{guard.relative(target)}:{first}-{last} (всего строк: {total})"
    return _clip(f"{head}\n{body}")


@tool
def search_files(pattern: str, glob: str = "*.py") -> str:
    """Ищет строку по файлам рабочего каталога, отвечает адресами «файл:строка»."""
    needle = (pattern or "").strip()
    if not needle:
        return "Пустой запрос: нечего искать."

    root = guard.get_workspace()
    lowered = needle.lower()
    hits: list[str] = []
    scanned = 0

    for file in sorted(root.rglob("*")):
        if len(hits) >= SEARCH_HITS:
            break
        if not file.is_file():
            continue
        if any(part in SKIP_DIRS for part in file.parts):
            continue
        if not fnmatch.fnmatch(file.name, glob):
            continue
        text = _read_text(file)
        if text is None:
            continue
        scanned += 1
        for number, line in enumerate(text.splitlines(), start=1):
            if lowered in line.lower():
                hits.append(f"{guard.relative(file)}:{number}: {line.strip()[:120]}")
                if len(hits) >= SEARCH_HITS:
                    break

    if not hits:
        return f"Ничего не найдено: '{needle}' в файлах {glob} (просмотрено файлов: {scanned})."
    head = f"Найдено совпадений: {len(hits)}" + (f" (показаны первые {SEARCH_HITS})" if len(hits) >= SEARCH_HITS else "")
    return _clip(head + "\n" + "\n".join(hits))


# ====================================================================
# ИЗМЕНЕНИЕ
# ====================================================================

@tool
def edit_file(path: str, old: str, new: str) -> str:
    """Заменяет единственное вхождение фрагмента old на new."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"

    before = _read_text(target)
    if before is None:
        return f"Файл не читается как текст: {guard.relative(target)}"

    result = apply_anchor(before, old, new)
    if not result.ok:
        return f"Правка не применена. {result.message}"

    problem = _commit(target, before, result.text, f"изменить {guard.relative(target)} по якорю")
    return problem or f"{guard.relative(target)}: {result.message}"


@tool
def replace_lines(path: str, start: str, end: str, content: str) -> str:
    """Заменяет строки с start по end включительно — адрес берётся из поиска по коду."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"

    before = _read_text(target)
    if before is None:
        return f"Файл не читается как текст: {guard.relative(target)}"

    first = _int(start, 0)
    last = _int(end, 0) or first
    result = apply_lines(before, first, last, content)
    if not result.ok:
        return f"Правка не применена. {result.message}"

    action = f"заменить строки {first}-{last} в {guard.relative(target)}"
    problem = _commit(target, before, result.text, action)
    return problem or f"{guard.relative(target)}: {result.message}"


@tool
def append_to_file(path: str, content: str) -> str:
    """Дописывает текст в конец файла, ничего не удаляя."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"

    before = _read_text(target)
    if before is None:
        return f"Файл не читается как текст: {guard.relative(target)}"

    result = apply_append(before, content)
    if not result.ok:
        return f"Правка не применена. {result.message}"

    problem = _commit(target, before, result.text, f"дописать в конец {guard.relative(target)}")
    return problem or f"{guard.relative(target)}: {result.message}"


@tool
def write_file(path: str, content: str) -> str:
    """Создаёт файл или перезаписывает его целиком."""
    return put_file(path, content, replace=False)


# Имена файловых инструментов одним списком — для выборки специалиста
# и для схемы ответа. Список, а не автоопределение по модулю: реестр
# Главы 2 общий, и «что относится к файлам» решает эта глава, а не
# порядок импортов.
FS_TOOLS = [
    "list_dir",
    "read_lines",
    "search_files",
    "edit_file",
    "replace_lines",
    "append_to_file",
    "write_file",
]
