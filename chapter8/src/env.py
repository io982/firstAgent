"""
Окружение проекта: своё виртуальное, зависимости, requirements.txt.

Агент, который создаёт проект с нуля, упирается в это на втором шаге.
Написать `import requests` он может; запустить получившийся файл —
уже нет, потому что пакета на машине может не быть. Дальше два пути:
сдаться и сказать человеку «поставьте сами» или разобраться самому.

Этот модуль про второй путь, и стоит он дороже остальной главы —
не в строках, а в правах. Установка пакета это загрузка кода из сети
и его выполнение при сборке. Никакой белый список программ этого
не ограничивает: `pip` в списке означает «ставить что угодно».

Поэтому здесь нет собственных ограничений — они все в `guard`, и здесь
работает ровно одно: **человек видит имена пакетов и подтверждает**.
Список пакетов печатается перед вопросом целиком, без сокращений.

Как устроено «работать в окружении проекта». Активации окружения,
какую делает человек командой `activate`, здесь нет: она меняет
переменные текущей оболочки, а мы запускаем процессы напрямую.
Вместо неё — один вопрос перед каждым запуском: есть ли в рабочем
каталоге `.venv`? Есть — берём его интерпретатор, нет — тот, которым
запущен агент. Это `shell.interpreter()`, и через него ходят все
запуски главы, включая тесты и линтер.

Отдельная работа модуля — **понять, чего не хватает**. `missing_imports()`
читает исходник, достаёт из него импорты и спрашивает у ЦЕЛЕВОГО
интерпретатора, какие из них не разрешаются. Спрашивает запуском,
а не проверкой у себя: у агента и у проекта интерпретаторы разные,
и ответ «у меня стоит» про проект не говорит ничего.
"""
from __future__ import annotations

import ast
import json
import sys

from chapter2.src.tools import tool
from chapter8.src import guard
from chapter8.src.shell import VENV_DIR, clip, execute, interpreter, venv_python

# Сколько ждать сборку окружения и установку пакетов. Отдельно от общего
# предела: `venv` на холодном диске идёт десятки секунд, а установка
# пакета с зависимостями — минуты. Общий предел в 30 с их просто убьёт.
ENV_TIMEOUT = 300.0

# Имя файла со списком зависимостей. То же, что у самого курса.
REQUIREMENTS = "requirements.txt"

# Пакеты, чьё имя при импорте не совпадает с именем при установке.
# Список короткий и заведомо неполный — он покрывает то, что реально
# встречается в учебных задачах. Полного списка не существует: связь
# «имя пакета -> имя модуля» нигде не объявлена и живёт в метаданных
# каждого пакета по отдельности.
IMPORT_TO_PACKAGE = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}


def has_venv() -> bool:
    """Есть ли у проекта своё окружение."""
    return venv_python() is not None


def env_report() -> str:
    """Строка о том, каким интерпретатором работает агент."""
    if has_venv():
        return f"Окружение проекта: {VENV_DIR} ({interpreter()})"
    return f"Окружения проекта нет, работаем интерпретатором агента: {sys.executable}"


# --------------------------------------------------------------------
# ЧТО ИМПОРТИРУЕТ КОД И ЧЕГО НЕ ХВАТАЕТ
# --------------------------------------------------------------------

def imported_modules(text: str) -> list[str]:
    """Имена модулей верхнего уровня, которые импортирует исходник.

    Разбором, а не поиском слова «import»: строка `import` встречается
    и в комментарии, и внутри текстовой константы. Берётся только первая
    часть пути — `import os.path` это зависимость от `os`, а не от
    `os.path`, и ставить их порознь нельзя.

    Относительные импорты (`from . import x`) пропускаются: это свои
    же файлы проекта, ставить их неоткуда.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.append(node.module.split(".")[0])

    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen


def local_modules() -> set[str]:
    """Имена модулей, которые лежат в самом проекте.

    Свой файл `calc.py` даёт `import calc`, и предлагать поставить
    пакет `calc` из сети было бы худшим, что тут можно сделать:
    в лучшем случае установка провалится, в худшем — поставится
    что-то чужое с таким именем.
    """
    root = guard.get_workspace()
    names = set()
    for path in root.glob("*.py"):
        names.add(path.stem)
    # Каталог с любым .py внутри — тоже свой пакет, даже без
    # `__init__.py`. Пространства имён (PEP 420) импортируются точно
    # так же, и `import pkg.sub` из такого каталога отправлял агента
    # искать в сети пакет `pkg`, которого там нет и быть не должно.
    for path in root.glob("*/*.py"):
        names.add(path.parent.name)
    return names


def missing_imports(text: str) -> list[str]:
    """Модули из исходника, которых нет в интерпретаторе проекта.

    Проверка делается ЗАПУСКОМ целевого интерпретатора, а не вызовом
    importlib у себя. Разница принципиальная, когда у проекта своё
    окружение: у агента пакет стоит, у проекта — нет, и ответ «всё
    на месте» был бы ответом не про тот Python.
    """
    wanted = [
        name for name in imported_modules(text)
        if name not in sys.stdlib_module_names and name not in local_modules()
    ]
    if not wanted:
        return []

    probe = (
        "import importlib.util, json, sys;"
        "print(json.dumps([m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))"
    )
    run = execute([interpreter(), "-c", probe, *wanted], timeout=60.0)
    if not run.ok:
        # Интерпретатор не ответил — честнее сказать «не знаю»
        # пустым списком, чем объявить недостающим всё подряд.
        return []
    try:
        return json.loads(run.out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return []


def package_for(module: str) -> str:
    """Как называется пакет, которым ставится этот модуль."""
    return IMPORT_TO_PACKAGE.get(module, module)


# --------------------------------------------------------------------
# ИНСТРУМЕНТЫ
# --------------------------------------------------------------------

@tool
def create_venv() -> str:
    """Создаёт виртуальное окружение проекта в папке .venv."""
    if has_venv():
        return f"Окружение уже есть: {interpreter()}"

    target = guard.get_workspace() / VENV_DIR
    verdict = guard.check(
        f"создать виртуальное окружение в {VENV_DIR}",
        f"каталог {target}\nинтерпретатор-родитель {sys.executable}",
    )
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, "создать окружение")

    # Через sys.executable, а не через interpreter(): окружения ещё нет,
    # и создавать его нечем, кроме того Python, которым запущен агент.
    run = execute([sys.executable, "-m", "venv", VENV_DIR], timeout=ENV_TIMEOUT)
    if not run.ok:
        return f"Окружение не создано (код {run.code}):\n{clip(run.text(), 1200)}"
    guard.record(target / "pyvenv.cfg")
    return f"Окружение создано: {interpreter()}"


@tool
def install(packages: str) -> str:
    """Ставит пакеты в окружение проекта. Имена перечисляются через пробел."""
    names = [p for p in (packages or "").replace(",", " ").split() if p]
    if not names:
        return "Не названо ни одного пакета."

    # Имена уезжают в командную строку, поэтому проверяются здесь,
    # а не «pip разберётся». Ключ вместо имени пакета — самый простой
    # способ попросить pip не о том, о чём думает человек.
    bad = [n for n in names if n.startswith("-") or any(c in n for c in " ;&|<>`$")]
    if bad:
        return f"Недопустимые имена пакетов: {', '.join(bad)}"

    where = f"в окружение проекта ({VENV_DIR})" if has_venv() else "в интерпретатор агента"
    verdict = guard.check(
        f"установить {len(names)} пакет(ов) {where}",
        "Пакеты будут скачаны из сети и выполнены при сборке:\n"
        + "\n".join(f"  {n}" for n in names),
    )
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, "установка пакетов")

    run = execute([interpreter(), "-m", "pip", "install", *names], timeout=ENV_TIMEOUT)
    if not run.ok:
        return f"Установка не удалась (код {run.code}):\n{clip(run.text(), 1500)}"
    return f"Установлено: {', '.join(names)}.\n{clip(run.out.strip().splitlines()[-1] if run.out.strip() else '', 300)}"


@tool
def write_requirements() -> str:
    """Записывает список установленных пакетов в requirements.txt."""
    run = execute([interpreter(), "-m", "pip", "freeze"], timeout=ENV_TIMEOUT)
    if not run.ok:
        return f"pip freeze не отработал (код {run.code}):\n{clip(run.text(), 800)}"

    # Без окружения проекта freeze перечисляет ВСЁ, что стоит у агента,
    # и такой requirements.txt описывает не проект, а чужую машину.
    if not has_venv():
        return (
            "Окружения проекта нет, и pip freeze перечислит все пакеты машины. "
            "Такой requirements.txt описывал бы не проект. Сначала create_venv."
        )

    lines = [line for line in run.out.splitlines() if line.strip() and not line.startswith("-e ")]
    target = guard.resolve_path(REQUIREMENTS)
    body = "\n".join(lines) + "\n"

    verdict = guard.check(f"записать {REQUIREMENTS}", f"{len(lines)} пакет(ов):\n" + "\n".join(lines[:20]))
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, f"записать {REQUIREMENTS}")

    guard.record(target)
    target.write_text(body, encoding="utf-8")
    return f"{REQUIREMENTS}: записано {len(lines)} пакет(ов)."


@tool
def check_imports(path: str) -> str:
    """Проверяет, все ли импорты файла разрешаются в окружении проекта."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"

    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"Файл не читается как текст: {guard.relative(target)}"

    missing = missing_imports(text)
    if not missing:
        return f"{guard.relative(target)}: все импорты разрешаются."
    packages = [package_for(m) for m in missing]
    return (
        f"{guard.relative(target)}: не разрешаются импорты — {', '.join(missing)}.\n"
        f"Поставить: install(\"{' '.join(packages)}\")"
    )


# Имена инструментов окружения — для выборки специалиста и схемы ответа.
ENV_TOOLS = ["create_venv", "install", "write_requirements", "check_imports"]
