"""
Запуск процессов: код, тесты, линтер (8.3).

Это то место, где учебный агент перестаёт быть безобидным. Всё
предыдущее в курсе читало и считало; здесь агент запускает программу,
и программа делает что хочет — в том числе то, чего никто не просил.

Поэтому запуск обставлен тремя ограничениями:

  * **подтверждение человеком** — перед каждой произвольной командой,
    с показом самой команды. После того как белый список программ
    выключили по умолчанию (см. `guard`), это главная проверка,
    а не дополнительная;
  * **предел по времени** — зависший процесс снимается, а не держит
    агента вечно;
  * **без командной оболочки** — процесс запускается напрямую, поэтому
    ни `&&`, ни перенаправление вывода не работают в принципе, а не
    «не рекомендуются».

Где здесь проходит граница честности. Подтверждение спрашивается перед
произвольной командой и перед запуском файла, но НЕ перед `run_tests`
и `run_lint`: у этих двух форма команды фиксирована, а в конвейере
они вызываются на каждом круге, и вопрос человеку на каждом круге
превратился бы в кнопку, которую жмут не глядя. Цена такого решения
тоже понятна и называется вслух: `pytest` выполняет `conftest.py`
проекта, то есть чужой код всё-таки исполняется без вопроса. Ограничение
работает против случайности, а не против злого умысла.

Отдельная работа модуля — **выбрать интерпретатор**. Агент, создавший
проекту своё окружение, обязан запускать код именно им, иначе тесты
пойдут мимо поставленных зависимостей. Это `interpreter()`, и через
неё ходят все запуски главы.

Отдельная работа этого модуля — **сжать вывод**. Полный отчёт pytest
о трёх упавших тестах — это тысячи символов, а контекст модели 4096
токенов на весь разговор. `first_error()` достаёт из отчёта строки,
по которым правку вообще можно сделать, и выбрасывает остальное.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from chapter2.src.tools import tool
from chapter8.src import guard

# Потолок на вывод одного запуска, символов. Больше, чем у файловых
# инструментов: сообщение об ошибке без окружающих строк часто
# бесполезно, а половина отчёта pytest — это как раз окружающие строки.
RUN_OUTPUT_LIMIT = 3000

# Сколько символов оставить с начала при обрезке. Остальное берётся
# с конца: pytest пишет итог последней строкой, и обрезка «первые N
# символов» выбросила бы ровно то, ради чего запускали.
HEAD_SHARE = 0.3


@dataclass
class Run:
    """Результат одного запуска процесса."""

    command: str
    code: int
    out: str
    err: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Успех — это нулевой код возврата и отсутствие обрыва по времени."""
        return self.code == 0 and not self.timed_out

    def text(self) -> str:
        """Вывод процесса одним куском: stdout, затем stderr."""
        parts = [self.out.strip(), self.err.strip()]
        return "\n".join(p for p in parts if p)

    def summary(self) -> str:
        """Короткий отчёт для модели и человека."""
        if self.timed_out:
            return (
                f"$ {self.command}\n"
                f"Прервано по времени: процесс не закончился за {self.seconds:.0f} с."
            )
        head = f"$ {self.command}\nКод возврата: {self.code}, время: {self.seconds:.1f} с"
        body = clip(self.text())
        return f"{head}\n{body}" if body else f"{head}\n(вывода нет)"


def clip(text: str, limit: int = RUN_OUTPUT_LIMIT) -> str:
    """Обрезает вывод, оставляя начало И конец.

    Обычная обрезка «первые N символов» на выводе тестов теряет самое
    важное: сколько упало и почему, pytest пишет в самом конце.
    """
    if len(text) <= limit:
        return text
    head = int(limit * HEAD_SHARE)
    tail = limit - head
    return f"{text[:head]}\n\n[...вырезано {len(text) - limit} символов...]\n\n{text[-tail:]}"


# Имя каталога виртуального окружения внутри проекта. Одно на курс,
# чтобы «где тут окружение» не было вопросом с несколькими ответами.
VENV_DIR = ".venv"


def venv_python(root=None):
    """Интерпретатор виртуального окружения проекта или None.

    Смотрим оба варианта раскладки: `Scripts/python.exe` на Windows
    и `bin/python` на всём остальном. Проверяется существование файла,
    а не наличие папки: недособранное окружение выглядит как папка,
    но интерпретатора в нём нет.
    """
    base = Path(root or guard.get_workspace()) / VENV_DIR
    for candidate in (base / "Scripts" / "python.exe", base / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def interpreter() -> str:
    """Каким Python запускать код проекта.

    Порядок один и тот же во всей главе: если в рабочем каталоге есть
    своё окружение — его интерпретатором, иначе тем, которым запущен
    агент. Это и есть весь механизм «работать в окружении проекта»:
    отдельного режима нет, есть один вопрос, задаваемый перед каждым
    запуском.

    Почему не активация окружения, как это делает человек. Активация
    меняет переменные ТЕКУЩЕЙ оболочки, а мы запускаем процессы напрямую,
    без оболочки. Прямой вызов нужного интерпретатора делает ровно то же
    самое и не зависит от того, что было в окружении до нас.
    """
    found = venv_python()
    return str(found) if found else sys.executable


def _environment() -> dict[str, str]:
    """Окружение дочернего процесса.

    PYTHONIOENCODING нужен на Windows: без него дочерний Python пишет
    вывод в кодировке консоли, а мы читаем его как UTF-8 — и русский
    текст в сообщении об ошибке превращается в мусор ровно тогда,
    когда он нужнее всего.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # Отключаем .pyc: агент правит файлы на ходу, и кэш байткода в этих
    # условиях иногда подсовывает прошлую версию модуля.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def execute(command: str | list[str], timeout: float | None = None) -> Run:
    """Запускает команду в рабочем каталоге. Проверок доступа НЕ делает.

    Проверки остались снаружи намеренно: эта функция — только запуск,
    и её зовут в том числе git-инструменты со своими правилами. Один
    слой отвечает за «как запустить», другой — за «можно ли».

    Команду можно передать списком аргументов, а не строкой, и для
    git-коммита это единственный рабочий способ: сообщение коммита
    пишет человек, в нём бывают и кавычки, и пробелы, и разбирать его
    обратно из строки значит ломать ровно то, что он написал.
    """
    if isinstance(command, str):
        parts = guard.split_command(command)
        shown = command
    else:
        parts = list(command)
        shown = " ".join(parts)
    if not parts:
        return Run(shown, 1, "", "Пустая команда", 0.0)

    # Подменяем `python` на интерпретатор проекта. Иначе на машине
    # с несколькими Python команда уйдёт в системный, где нет ни pytest,
    # ни зависимостей проекта.
    if parts[0].lower() in ("python", "python3", "python.exe"):
        parts[0] = interpreter()

    limit = timeout if timeout is not None else guard.get_policy().timeout
    started = time.monotonic()
    try:
        done = subprocess.run(
            parts,
            cwd=str(guard.get_workspace()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=limit,
            env=_environment(),
            shell=False,
            # Пустой ввод, а не наш терминал. Программа, написанная
            # агентом, вполне может ждать ввода с клавиатуры — и, получив
            # наш терминал, заберёт его себе: человек увидит зависший
            # агент вместо приглашения. С закрытым вводом она честно
            # падает на EOF, и это состояние, с которым можно работать.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return Run(shown, 124, "", "", time.monotonic() - started, timed_out=True)
    except FileNotFoundError:
        return Run(shown, 127, "", f"Программа не найдена: {parts[0]}", time.monotonic() - started)
    except OSError as exc:
        return Run(shown, 126, "", f"Не удалось запустить: {exc}", time.monotonic() - started)

    return Run(
        command=shown,
        code=done.returncode,
        out=done.stdout or "",
        err=done.stderr or "",
        seconds=time.monotonic() - started,
    )


def _guarded_run(command: str, ask: bool, timeout: float | None = None) -> str:
    """Общий путь всех инструментов запуска: проверить, спросить, выполнить."""
    allowed, reason = guard.command_allowed(command)
    if not allowed:
        return f"Команда не выполнена. {reason}"

    if ask:
        verdict = guard.check(f"выполнить команду: {command}", f"в каталоге {guard.get_workspace()}")
        if verdict != guard.ALLOW:
            return guard.verdict_message(verdict, f"выполнить {command}")

    return execute(command, timeout=timeout).summary()


# ====================================================================
# РАЗБОР ВЫВОДА: ЧТО ИМЕННО СЛОМАЛОСЬ
# ====================================================================

# Строка итога pytest: «2 failed, 5 passed in 0.31s».
_PYTEST_TOTALS = re.compile(r"^=+\s.*\b(passed|failed|error|no tests ran)\b.*=+$", re.MULTILINE)
# Строка утверждения pytest: отчёт печатает их с ведущего «E ».
_PYTEST_ASSERT = re.compile(r"^E\s+.+$", re.MULTILINE)
# Последняя строка трассировки — тип исключения и сообщение.
_EXCEPTION = re.compile(r"^(\w*(?:Error|Exception|Warning))\b.*$", re.MULTILINE)


def first_error(text: str, limit: int = 600) -> str:
    """Достаёт из вывода то, по чему можно сделать правку.

    Три источника по убыванию полезности: строки утверждений pytest,
    строка исключения из трассировки, итоговая строка отчёта. Первый,
    который нашёлся, и возвращается.

    Смысл функции не в красоте, а в бюджете: полный отчёт pytest
    вытесняет из контекста 3B-модели весь разговор, а три строки из
    него — не вытесняют.
    """
    if not text.strip():
        return ""

    asserts = _PYTEST_ASSERT.findall(text)
    if asserts:
        return "\n".join(asserts[:6])[:limit]

    exceptions = _EXCEPTION.findall(text)
    if exceptions:
        for line in text.splitlines():
            if line.strip().startswith(exceptions[0]):
                return line.strip()[:limit]

    totals = _PYTEST_TOTALS.findall(text)
    if totals:
        return _PYTEST_TOTALS.search(text).group(0).strip("= ")[:limit]

    return text.strip().splitlines()[-1][:limit]


def suite_passed(run: Run) -> bool:
    """Прошли ли тесты.

    Имя не начинается с «test» намеренно: pytest собирает тесты по этому
    префиксу и попытался бы запустить саму функцию как тест.

    Не то же самое, что `run.ok`. pytest возвращает 5, когда не нашёл
    ни одного теста, и трактовать это как успех нельзя: «тестов нет»
    и «тесты прошли» — разные новости, и вторая из первой не следует.
    """
    if run.timed_out or run.code != 0:
        return False
    return "no tests ran" not in run.text().lower()


# ====================================================================
# ИНСТРУМЕНТЫ
# ====================================================================

@tool
def run_command(command: str) -> str:
    """Выполняет команду из белого списка, показывает вывод и код возврата."""
    return _guarded_run(command, ask=True)


@tool
def run_python(path: str) -> str:
    """Запускает Python-файл из рабочего каталога и возвращает его вывод."""
    try:
        target = guard.resolve_path(path)
    except guard.OutsideWorkspace as exc:
        return str(exc)
    if not target.is_file():
        return f"Нет такого файла: {guard.relative(target)}"
    return _guarded_run(f"python {guard.relative(target)}", ask=True)


@tool
def run_tests(target: str = "") -> str:
    """Прогоняет pytest и возвращает итог вместе с первой ошибкой."""
    where = ""
    if target.strip():
        try:
            where = " " + guard.relative(guard.resolve_path(target))
        except guard.OutsideWorkspace as exc:
            return str(exc)

    # -q убирает шапку и точки, --no-header — заголовок с версиями:
    # это десятки строк, которые модели не говорят ничего, а место
    # в контексте занимают.
    run = execute(f"python -m pytest{where} -q --no-header", timeout=max(guard.get_policy().timeout, 120.0))
    if run.timed_out:
        return run.summary()

    verdict = "тесты прошли" if suite_passed(run) else "тесты НЕ прошли"
    problem = "" if suite_passed(run) else f"\nГлавное:\n{first_error(run.text())}"
    return f"{verdict} (код возврата {run.code}, {run.seconds:.1f} с){problem}\n\n{clip(run.text(), 1200)}"


@tool
def run_lint(path: str = ".") -> str:
    """Проверяет код линтером ruff и возвращает найденные замечания."""
    try:
        where = guard.relative(guard.resolve_path(path))
    except guard.OutsideWorkspace as exc:
        return str(exc)

    run = execute(f"python -m ruff check {where}")
    if run.code == 127 or "No module named" in run.text():
        return "Линтер ruff не установлен. Поставьте: pip install ruff"
    return run.summary()


# Имена инструментов запуска — для выборки специалиста и схемы ответа.
RUN_TOOLS = ["run_command", "run_python", "run_tests", "run_lint"]
