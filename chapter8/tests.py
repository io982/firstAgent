"""
Тесты Главы 8: границы, формы правки, файловые инструменты, запуск, git.

Запуск быстрых (без Ollama и без сети):

    python -m pytest chapter8/tests.py -q

Тесты, которым нужна запущенная Ollama, помечены `integration`; замеры,
из которых берутся числа текста главы, — `slow`. По умолчанию и те,
и другие пропускаются (см. pytest.ini).

Главная особенность этих тестов по сравнению с предыдущими главами:
здесь код пишет на диск и запускает процессы. Поэтому каждый тест
работает в своём временном каталоге (`tmp_path`), а политика доступа
сбрасывается после каждого — иначе тест, включивший сухой прогон,
тихо испортил бы следующий.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

import chapter1.agent as base
import chapter7.agent as chapter7_agent
import chapter8.agent as agent8
from chapter1.agent import request_model
from chapter7.src.agents import SPECIALISTS, Team
from chapter7.src.graph import State
from chapter7.src.models import using_model
from chapter8.agent import handle
from chapter8.src import codemap, env, guard, pipeline, pipeline_lg, review
from chapter8.src import planner as planner_module
from chapter8.src.edits import (
    ANCHOR,
    APPEND,
    EDIT_FORMS,
    FULL,
    LINES,
    REPLACING_FORMS,
    REQUIRED_FIELDS,
    apply_anchor,
    apply_append,
    apply_full,
    apply_lines,
    definitions,
    describe_forms,
    doubled_main,
    edit_schema,
    filled_bodies,
    lost_definitions,
    missing_fields,
    same_code,
    same_tree,
    stray_definitions,
    syntax_ok,
    unified,
    unreachable_code,
    without_docstring,
)
from chapter8.src.env import ENV_TOOLS
from chapter8.src.fs import (
    FS_TOOLS,
    append_to_file,
    edit_file,
    list_dir,
    put_file,
    read_lines,
    replace_lines,
    search_files,
    write_file,
)
from chapter8.src.pipeline import build_pipeline, run_pipeline
from chapter8.src.planner import (
    FROM_FALLBACK,
    FROM_MODEL,
    MAX_STEPS,
    PLAN_ACTIONS,
    Plan,
    Step,
    build_planner_prompt,
    fallback_plan,
    make_plan,
    named_file,
    parse_plan,
    plan_kind,
    plan_schema,
    render_plan,
    split_target,
    validate_plan,
)
from chapter8.src.session import KEYS, SessionMemory
from chapter8.src.shell import (
    RUN_TOOLS,
    Run,
    clip,
    execute,
    first_error,
    interpreter,
    suite_passed,
    undefined_names,
)
from chapter8.src.vcs import GIT_TOOLS, current_branch, git_commit, git_diff, git_log, git_status

# Настоящий выбор места: фикстура `workspace` подменяет его заглушкой,
# чтобы быстрые тесты не ходили к модели, а тестам самого выбора нужен
# он. Ссылка берётся до первой подмены.
REAL_CHOOSE = codemap.choose
REAL_REVIEW = review.review
REAL_FIND = codemap.find

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git не установлен")


# ====================================================================
# ОБЩИЕ ФИКСТУРЫ
# ====================================================================

@pytest.fixture(autouse=True)
def clean_policy():
    """Возвращает политику доступа в исходное состояние после каждого теста.

    autouse — потому что забыть этот сброс легче всего в тесте, который
    его как раз и ломает: включил сухой прогон, проверил, вышел.
    Следующий тест тогда пишет в пустоту и падает по причине,
    не имеющей к нему отношения.
    """
    yield
    guard.forget_changes()
    guard.reset_policy()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Пустой рабочий каталог с политикой «разрешать молча».

    AUTO, а не ASK: подтверждение читает stdin, а у тестов его нет.
    Сами подтверждения проверяются отдельно, подделкой функции confirm.

    Память сессии здесь же уводится во временный файл. Она глобальная:
    планировщик спрашивает её, не получая параметром, — и без подмены
    пути тесты читали бы память разработчика. Тест, который падает
    в зависимости от того, чем человек занимался в агенте вчера,
    не проверяет ничего.
    """
    monkeypatch.setenv("AGENT_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.setenv("AGENT_CODEMAP_FILE", str(tmp_path / "codemap.json"))
    # Выбор места по карте ходит к модели, а быстрые тесты к ней
    # не ходят по определению: они должны идти секунды и работать без
    # запущенной Ollama. Тесты, которым выбор нужен, подменяют его сами.
    monkeypatch.setattr(codemap, "choose", lambda task, path="": None)
    monkeypatch.setattr(review, "review", lambda task, path, model_call=None: (True, []))
    codemap.forget_cache()
    guard.set_policy(root=tmp_path, mode=guard.AUTO, dry_run=False)
    return tmp_path


@pytest.fixture
def sample(workspace):
    """Учебный файл, на котором проверяются правки."""
    path = workspace / "sample.py"
    path.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def mul(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    return path


# ====================================================================
# ГРАНИЦЫ РАБОЧЕГО КАТАЛОГА
# ====================================================================

class TestWorkspace:
    def test_относительный_путь_разворачивается_от_корня(self, workspace):
        assert guard.resolve_path("a/b.txt") == (workspace / "a" / "b.txt").resolve()

    def test_несуществующий_файл_это_не_ошибка(self, workspace):
        # write_file как раз создаёт файлы, которых ещё нет: требовать
        # существования значило бы запретить создание.
        assert guard.resolve_path("new.txt").name == "new.txt"

    @pytest.mark.parametrize(
        "path",
        [
            "../secret.txt",
            "a/../../secret.txt",
            "./../../..",
        ],
    )
    def test_выход_вверх_запрещён(self, workspace, path):
        with pytest.raises(guard.OutsideWorkspace):
            guard.resolve_path(path)

    def test_абсолютный_путь_наружу_запрещён(self, workspace, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "secret.txt"
        outside.write_text("тайна", encoding="utf-8")
        with pytest.raises(guard.OutsideWorkspace):
            guard.resolve_path(str(outside))

    def test_домашний_каталог_запрещён(self, workspace):
        with pytest.raises(guard.OutsideWorkspace):
            guard.resolve_path("~/.ssh/id_rsa")

    def test_пустой_путь_запрещён(self, workspace):
        with pytest.raises(guard.OutsideWorkspace):
            guard.resolve_path("   ")

    def test_ссылка_наружу_разворачивается_и_отвергается(self, workspace, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        link = workspace / "door"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("создание символических ссылок недоступно")
        with pytest.raises(guard.OutsideWorkspace):
            guard.resolve_path("door/secret.txt")

    def test_сам_корень_разрешён(self, workspace):
        assert guard.resolve_path(".") == workspace.resolve()

    def test_относительный_путь_печатается_коротко(self, workspace):
        assert guard.relative(workspace / "src" / "a.py") == "src/a.py"


# ====================================================================
# ПОДТВЕРЖДЕНИЕ И РЕЖИМЫ
# ====================================================================

class TestPolicy:
    def test_режим_auto_разрешает(self, workspace):
        assert guard.check("что-то", "детали") == guard.ALLOW

    def test_режим_deny_запрещает(self, workspace):
        guard.set_policy(mode=guard.DENY)
        assert guard.check("что-то", "детали") == guard.DENIED

    def test_режим_ask_спрашивает_и_слушается(self, workspace):
        asked = []

        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: asked.append(a) or False)
        assert guard.check("удалить всё", "детали") == guard.DENIED
        assert asked == ["удалить всё"]

        guard.set_policy(confirm=lambda a, d: True)
        assert guard.check("удалить всё", "детали") == guard.ALLOW

    def test_сухой_прогон_не_спрашивает(self, workspace):
        """Раз ничего не произойдёт — вопрос человеку был бы шумом."""
        asked = []
        guard.set_policy(mode=guard.ASK, dry_run=True, confirm=lambda a, d: asked.append(a) or True)
        assert guard.check("что-то", "детали") == guard.DRY
        assert asked == []

    def test_запрет_сильнее_сухого_прогона(self, workspace):
        guard.set_policy(mode=guard.DENY, dry_run=True)
        assert guard.check("что-то", "детали") == guard.DENIED

    def test_у_отказа_и_сухого_прогона_разный_текст(self):
        dry = guard.verdict_message(guard.DRY, "записать файл")
        denied = guard.verdict_message(guard.DENIED, "записать файл")
        assert dry != denied
        assert "Сухой прогон" in dry
        assert "Отказано" in denied


# ====================================================================
# БЕЛЫЙ СПИСОК КОМАНД
# ====================================================================

class TestAllowlist:
    @pytest.mark.parametrize(
        "command",
        [
            "python script.py",
            "pytest -q",
            "ruff check .",
            "git status",
            "python -m pytest chapter8/tests.py",
            "python -m ruff check .",
        ],
    )
    def test_разрешённые(self, workspace, command):
        allowed, reason = guard.command_allowed(command)
        assert allowed, reason

    @pytest.mark.parametrize(
        "command",
        [
            "curl http://example.com",
            "rm -rf /",
            "npm install",
            "powershell -c ls",
        ],
    )
    def test_при_узком_списке_чужая_программа_не_пускается(self, workspace, command):
        guard.set_policy(allowed=guard.NARROW_ALLOWED)
        allowed, reason = guard.command_allowed(command)
        assert not allowed
        assert "белом списке" in reason

    @pytest.mark.parametrize("command", ["curl http://example.com", "npm install", "docker ps"])
    def test_по_умолчанию_списка_нет(self, workspace, command):
        """Умолчание главы: белый список выключен, остаётся подтверждение.

        Решение осознанное и с ценой: агент, который создаёт проекты
        и ставит зависимости, заранее не знает нужных ему программ,
        а список из пятидесяти имён защитой уже не является. Последней
        проверкой становится человек, читающий команду перед «y».
        """
        assert guard.get_policy().allowed is guard.ANY_COMMAND
        assert guard.command_allowed(command)[0]

    def test_синтаксис_оболочки_отвергается_и_без_списка(self, workspace):
        """Это проверка на осмысленность, а не на права: оболочки у нас нет."""
        assert not guard.command_allowed("pip install a && pip install b")[0]

    @pytest.mark.parametrize(
        "command",
        [
            "git status && rm -rf .",
            "python a.py | tee out.txt",
            "python a.py > out.txt",
            "python a.py; git push",
        ],
    )
    def test_синтаксис_оболочки_отвергается(self, workspace, command):
        allowed, reason = guard.command_allowed(command)
        assert not allowed
        assert "оболочки" in reason

    def test_пустая_команда(self, workspace):
        allowed, _ = guard.command_allowed("   ")
        assert not allowed

    def test_имя_программы_берётся_без_пути_и_расширения(self, workspace):
        assert guard.program_of("python script.py") == "python"
        assert guard.program_of("python -m pytest -q") == "pytest"
        assert guard.program_of("git commit -m x") == "git"

    def test_список_можно_сузить(self, workspace):
        guard.set_policy(allowed=("git",))
        assert guard.command_allowed("git status")[0]
        assert not guard.command_allowed("python a.py")[0]


# ====================================================================
# ЖУРНАЛ ИЗМЕНЕНИЙ И ОТКАТ
# ====================================================================

class TestJournal:
    def test_откат_возвращает_прежний_текст(self, sample):
        guard.record(sample)
        sample.write_text("сломано", encoding="utf-8")
        guard.rollback()
        assert "def add(a, b):" in sample.read_text(encoding="utf-8")

    def test_откат_удаляет_созданный_файл(self, workspace):
        new = workspace / "new.txt"
        guard.record(new)          # файла ещё нет — в журнал ляжет None
        new.write_text("текст", encoding="utf-8")
        guard.rollback()
        assert not new.exists()

    def test_две_правки_откатываются_к_первому_состоянию(self, sample):
        """Самая тонкая часть отката: журнал разматывается с конца."""
        guard.record(sample)
        sample.write_text("вторая версия", encoding="utf-8")
        guard.record(sample)
        sample.write_text("третья версия", encoding="utf-8")

        guard.rollback()
        assert "def add(a, b):" in sample.read_text(encoding="utf-8")

    def test_список_файлов_без_повторов(self, sample):
        guard.record(sample)
        guard.record(sample)
        assert guard.changed_files() == [sample]

    def test_откат_очищает_журнал(self, sample):
        guard.record(sample)
        guard.rollback()
        assert guard.changed_files() == []

    def test_двоичный_файл_в_журнал_не_попадает(self, workspace):
        binary = workspace / "picture.bin"
        binary.write_bytes(b"\x00\x01\x02\xff")
        guard.record(binary)
        # Восстановить двоичный файл текстом нельзя, и делать вид,
        # что можно, хуже, чем честно его не записать.
        assert guard.changed_files() == []


# ====================================================================
# ТРИ ФОРМЫ ПРАВКИ
# ====================================================================

TEXT = "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n"


class TestAnchor:
    def test_единственное_вхождение_заменяется(self):
        result = apply_anchor(TEXT, "return a + b", "return a + b + 0")
        assert result.ok
        assert result.form == ANCHOR
        assert "return a + b + 0" in result.text

    def test_якорь_не_найден(self):
        result = apply_anchor(TEXT, "return a - b", "x")
        assert not result.ok
        assert "не найден" in result.message

    def test_к_ненайденному_якорю_даётся_похожая_строка(self):
        result = apply_anchor(TEXT, "    return a  + b", "x")
        assert not result.ok
        assert "Похожая строка" in result.message

    def test_несколько_вхождений_это_отказ(self):
        text = "x = 1\ny = 2\nx = 1\n"
        result = apply_anchor(text, "x = 1", "x = 3")
        assert not result.ok
        assert "2 раз" in result.message
        assert result.text == text

    def test_пустой_якорь(self):
        assert not apply_anchor(TEXT, "", "x").ok

    def test_якорь_равен_замене(self):
        result = apply_anchor(TEXT, "return a + b", "return a + b")
        assert not result.ok
        assert "ничего не меняет" in result.message

    def test_перевод_строки_windows_не_ломает_якорь(self):
        """Модель отвечает через \\n, а файл на Windows хранит \\r\\n."""
        windows = TEXT.replace("\n", "\r\n")
        result = apply_anchor(windows, "def add(a, b):\n    return a + b", "def add(a, b):\n    return 0")
        assert result.ok, result.message
        assert "return 0" in result.text
        # Все переводы строк остались windows-овскими: одиноких \n нет.
        assert "\n" not in result.text.replace("\r\n", "")

    def test_при_отказе_текст_остаётся_прежним(self):
        assert apply_anchor(TEXT, "нет такого", "x").text == TEXT


class TestLines:
    def test_замена_одной_строки(self):
        result = apply_lines(TEXT, 2, 2, "    return a - b")
        assert result.ok
        assert result.form == LINES
        assert "return a - b" in result.text
        assert "def mul(a, b):" in result.text

    def test_замена_диапазона(self):
        result = apply_lines(TEXT, 1, 2, "def add(a, b):\n    return 0")
        assert result.ok
        assert "return 0" in result.text
        assert "a + b" not in result.text

    def test_нумерация_с_единицы(self):
        result = apply_lines(TEXT, 0, 1, "x")
        assert not result.ok
        assert "начинается с 1" in result.message

    def test_конец_раньше_начала(self):
        result = apply_lines(TEXT, 3, 2, "x")
        assert not result.ok
        assert "раньше начала" in result.message

    def test_за_пределами_файла(self):
        result = apply_lines(TEXT, 1, 99, "x")
        assert not result.ok
        assert "5 строк" in result.message

    def test_начало_за_пределами_файла(self):
        assert not apply_lines(TEXT, 99, 100, "x").ok

    def test_последняя_строка_сохраняет_перевод(self):
        result = apply_lines(TEXT, 5, 5, "    return 0")
        assert result.ok
        assert result.text.endswith("\n")

    def test_замена_на_то_же_самое_это_отказ(self):
        result = apply_lines(TEXT, 2, 2, "    return a + b")
        assert not result.ok

    def test_переводы_строк_файла_сохраняются(self):
        windows = TEXT.replace("\n", "\r\n")
        result = apply_lines(windows, 2, 2, "    return a - b")
        assert result.ok
        assert "\r\n" in result.text
        assert "\n" not in result.text.replace("\r\n", "")


class TestRestoreIndent:
    """Отступ, потерянный моделью в форме `lines`, — второе, что нашла живая модель."""

    def test_потерянный_отступ_возвращается(self):
        result = apply_lines(TEXT, 2, 2, "return a - b")
        assert result.ok
        assert "    return a - b" in result.text
        assert "Отступ восстановлен (4" in result.message

    def test_файл_после_этого_разбирается(self):
        result = apply_lines(TEXT, 2, 2, "return a - b")
        assert syntax_ok("a.py", result.text) == (True, "")

    def test_свой_отступ_модели_не_трогается(self):
        result = apply_lines(TEXT, 2, 2, "        return a - b")
        assert "        return a - b" in result.text
        assert "Отступ восстановлен" not in result.message

    def test_замена_на_уровне_модуля_не_сдвигается(self):
        """У строки 1 отступа не было — добавлять нечего."""
        result = apply_lines(TEXT, 1, 1, "def add(a, b, c=0):")
        assert result.text.startswith("def add(a, b, c=0):")
        assert "Отступ восстановлен" not in result.message

    def test_многострочная_замена_получает_общий_отступ(self):
        result = apply_lines(TEXT, 2, 2, "value = a + b\nreturn value")
        assert "    value = a + b" in result.text
        assert "    return value" in result.text
        assert syntax_ok("a.py", result.text) == (True, "")

    def test_смешанный_отступ_считается_своим(self):
        """Первая строка без отступа, вторая с ним — модель расставила сама."""
        result = apply_lines(TEXT, 2, 2, "    if a:\n        return a + b")
        assert "Отступ восстановлен" not in result.message

    def test_пустая_замена_ничего_не_ломает(self):
        result = apply_lines(TEXT, 2, 2, "")
        assert result.ok or "ничего не меняет" in result.message


class TestFull:
    def test_перезапись(self):
        result = apply_full(TEXT, "x = 1\n")
        assert result.ok
        assert result.form == FULL
        assert result.text == "x = 1\n"

    def test_тот_же_текст_это_отказ(self):
        assert not apply_full(TEXT, TEXT).ok

    def test_резкое_укорочение_помечается(self):
        long_text = "".join(f"line {n}\n" for n in range(40))
        result = apply_full(long_text, "line 0\n")
        assert result.ok
        assert "ВНИМАНИЕ" in result.message

    def test_короткий_файл_не_вызывает_предупреждения(self):
        result = apply_full(TEXT, "x = 1\n")
        assert "ВНИМАНИЕ" not in result.message


class TestLostDefinitions:
    """Правка не имеет права уносить с собой чужие определения.

    Проверка родилась из замера: на файле из десяти функций форма `full`
    оставляла одну, а форма `lines` промахивалась номером строки на
    единицу и съедала заголовок функции. Снаружи это разные поломки,
    внутри — одна: из файла пропало определение.
    """

    LONG = (
        "def half(x):\n    return x / 2\n\n"
        "def add(a, b):\n    return a - b\n\n"
        "def join(parts):\n    return ', '.join(parts)\n"
    )

    def test_имена_определений_читаются_разбором(self):
        assert definitions("a.py", self.LONG) == {"half", "add", "join"}

    def test_слово_def_в_строке_определением_не_считается(self):
        text = 'ЖУРНАЛ = "def подделка(): pass"\n\ndef настоящая():\n    pass\n'
        assert definitions("a.py", text) == {"настоящая"}

    def test_вложенные_и_классы_тоже_считаются(self):
        text = "class A:\n    def m(self):\n        def inner():\n            pass\n"
        assert definitions("a.py", text) == {"A", "m", "inner"}

    def test_перезапись_одной_функцией_теряет_остальные(self):
        lost = lost_definitions("a.py", self.LONG, "def add(a, b):\n    return a + b\n")
        assert lost == ["half", "join"]

    def test_промах_на_строку_съедает_заголовок(self):
        """Модель называет строку 4 вместо 5 — и заголовок add исчезает."""
        broken = apply_lines(self.LONG, 4, 4, "    return a + b")
        assert broken.ok, broken.message
        assert lost_definitions("a.py", self.LONG, broken.text) == ["add"]

    def test_верная_правка_ничего_не_теряет(self):
        good = apply_lines(self.LONG, 5, 5, "    return a + b")
        assert lost_definitions("a.py", self.LONG, good.text) == []

    def test_не_python_не_проверяется(self):
        assert lost_definitions("readme.md", "def a(): pass", "") == []

    def test_добавление_функции_это_не_потеря(self):
        added = self.LONG + "\ndef новая():\n    pass\n"
        assert lost_definitions("a.py", self.LONG, added) == []

    def test_промах_на_строку_не_доезжает_до_диска(self, workspace):
        path = workspace / "long.py"
        path.write_text(self.LONG, encoding="utf-8")
        out = replace_lines("long.py", "4", "4", "    return a + b")
        assert "пропали определения" in out
        assert "add" in out
        assert path.read_text(encoding="utf-8") == self.LONG


class TestAppend:
    """Четвёртая форма правки — дописать в конец. Пришла из живой задачи.

    Человек попросил «добавить в calc.py интерактивность», и модель
    ответила якорем с ПУСТЫМ `old`: попыталась сказать «просто допиши
    вот это», не имея для такой мысли формы. Правка не применилась,
    круг сгорел.
    """

    def test_дописывает_в_конец(self):
        result = apply_append(TEXT, "def div(a, b):\n    return a / b\n")
        assert result.ok
        assert result.form == APPEND
        assert result.text.startswith(TEXT.rstrip("\n"))
        assert result.text.rstrip().endswith("return a / b")

    def test_ничего_не_удаляет(self):
        result = apply_append(TEXT, "x = 1\n")
        assert lost_definitions("a.py", TEXT, result.text) == []

    def test_файл_остаётся_разбираемым(self):
        result = apply_append(TEXT, "def div(a, b):\n    return a / b\n")
        assert syntax_ok("a.py", result.text) == (True, "")

    def test_пустая_строка_между_старым_и_новым(self):
        """Без неё приписанная функция слипается с последней строкой."""
        result = apply_append("x = 1\n", "def f():\n    pass\n")
        assert "x = 1\n\ndef f():" in result.text

    def test_вторая_пустая_строка_не_добавляется(self):
        result = apply_append("x = 1\n\n", "y = 2\n")
        assert "\n\n\n" not in result.text

    def test_пустой_текст_дописывать_нечего(self):
        assert not apply_append(TEXT, "   ").ok

    def test_в_пустой_файл_тоже_можно(self):
        result = apply_append("", "x = 1\n")
        assert result.ok
        assert result.text == "x = 1\n"

    def test_переводы_строк_файла_сохраняются(self):
        windows = TEXT.replace("\n", "\r\n")
        result = apply_append(windows, "x = 1\n")
        assert "\n" not in result.text.replace("\r\n", "")

    def test_форма_есть_в_схеме_и_в_описании(self):
        forms = [v["properties"]["form"]["enum"][0] for v in edit_schema()["oneOf"]]
        assert APPEND in forms
        assert APPEND in describe_forms()

    def test_замеряются_только_заменяющие_формы(self):
        """Сравнивать между собой имеет смысл те три, что решают одну задачу."""
        assert APPEND not in REPLACING_FORMS
        assert set(EDIT_FORMS) == set(REPLACING_FORMS) | {APPEND}

    def test_инструмент_дописывает(self, sample):
        out = append_to_file("sample.py", "def div(a, b):\n    return a / b\n")
        assert "Дописано" in out
        text = sample.read_text(encoding="utf-8")
        assert "def add" in text and "def mul" in text and "def div" in text

    def test_инструмент_проверяет_синтаксис(self, sample):
        before = sample.read_text(encoding="utf-8")
        out = append_to_file("sample.py", "def broken(:\n")
        assert "перестаёт разбираться" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_конвейер_применяет_форму(self, sample):
        out = pipeline.apply_edit(
            {"form": "append", "path": "sample.py", "content": "def div(a, b):\n    return a / b\n"}
        )
        assert "Дописано" in out
        assert "def div" in sample.read_text(encoding="utf-8")


class TestRunStepFallback:
    """Команда в поле `detail` вместо `target`. Тоже из живого прогона."""

    def test_команда_берётся_из_описания(self, workspace):
        (workspace / "hi.py").write_text("print('ага')\n", encoding="utf-8")
        state = started(Plan("з", [Step("run", "", "python hi.py")]))
        pipeline.node_step(state)
        assert "ага" in state.extra["log"][0]
        assert state.extra["failed_steps"] == []

    def test_путь_важнее_описания(self, workspace):
        (workspace / "hi.py").write_text("print('из target')\n", encoding="utf-8")
        state = started(Plan("з", [Step("run", "python hi.py", "python другое.py")]))
        pipeline.node_step(state)
        assert "из target" in state.extra["log"][0]

    def test_совсем_без_команды_это_провал(self, workspace):
        state = started(Plan("з", [Step("run", "", "")]))
        pipeline.node_step(state)
        assert state.extra["failed_steps"] == ["1. run "]

    def test_проверка_плана_ругается_на_run_без_команды(self, workspace):
        plan = Plan("з", [Step("run", "", ""), Step("test", "", "")])
        assert any("run без команды" in c for c in validate_plan(plan))

    def test_команда_в_описании_претензией_не_считается(self, workspace):
        plan = Plan("з", [Step("run", "", "python app.py"), Step("test", "", "")])
        assert not any("run без команды" in c for c in validate_plan(plan))


class TestSyntax:
    def test_целый_python_проходит(self):
        assert syntax_ok("a.py", TEXT) == (True, "")

    def test_сломанный_python_не_проходит(self):
        ok, problem = syntax_ok("a.py", "def add(:\n")
        assert not ok
        assert "строка" in problem

    def test_не_python_не_проверяется(self):
        assert syntax_ok("readme.md", "### это не код (((") == (True, "")


class TestEditSchema:
    """Схема правки: три варианта через oneOf, а не один общий объект.

    Переписана после замера. Общий объект, где обязательны только `form`
    и `path`, давал сборке qwen2_5coder3b_q5 ноль применимых правок
    из пяти: она заявляла форму `anchor` и заполняла `start` от `lines`.
    `enum` держит значение поля, а не смысл ответа.
    """

    def test_вариантов_столько_же_сколько_форм(self):
        assert len(edit_schema()["oneOf"]) == len(EDIT_FORMS)

    def test_в_каждом_варианте_свои_поля_обязательны(self):
        for variant in edit_schema()["oneOf"]:
            form = variant["properties"]["form"]["enum"][0]
            assert set(variant["required"]) == {"form", "path", *REQUIRED_FIELDS[form]}

    def test_поля_чужой_формы_в_вариант_не_попадают(self):
        """Ровно это и ломало слабую модель: `start` при форме anchor."""
        anchor = next(v for v in edit_schema()["oneOf"]
                      if v["properties"]["form"]["enum"] == ["anchor"])
        assert "start" not in anchor["properties"]
        assert "content" not in anchor["properties"]

    def test_номера_строк_объявлены_числами(self):
        lines = next(v for v in edit_schema()["oneOf"]
                     if v["properties"]["form"]["enum"] == ["lines"])
        assert lines["properties"]["start"]["type"] == "integer"

    def test_одна_форма_даёт_схему_без_oneOf(self):
        narrow = edit_schema((ANCHOR,))
        assert "oneOf" not in narrow
        assert narrow["properties"]["form"]["enum"] == [ANCHOR]

    def test_сужение_сохраняет_порядок_форм(self):
        two = edit_schema((FULL, ANCHOR))
        assert [v["properties"]["form"]["enum"][0] for v in two["oneOf"]] == [ANCHOR, FULL]

    def test_пустое_сужение_это_все_формы(self):
        assert len(edit_schema(())["oneOf"]) == len(EDIT_FORMS)

    def test_проверка_полноты_осталась(self):
        """Схема закрывает ответ модели, а план приходит и от человека."""
        assert missing_fields({"form": "anchor", "path": "a.py", "old": "x"}) == ["new"]
        assert missing_fields({"form": "anchor", "path": "a.py", "old": "x", "new": ""}) == []
        assert "неизвестная форма" in missing_fields({"form": "diff"})[0]


class TestDiffAndForms:
    def test_diff_показывает_обе_стороны(self):
        text = unified(TEXT, TEXT.replace("a + b", "a - b"), "sample.py")
        assert "-    return a + b" in text
        assert "+    return a - b" in text

    def test_описание_форм_перечисляет_все_три(self):
        text = describe_forms()
        for form in (ANCHOR, LINES, FULL):
            assert form in text


# ====================================================================
# ФАЙЛОВЫЕ ИНСТРУМЕНТЫ: ЧТЕНИЕ
# ====================================================================

class TestReadTools:
    def test_листинг_показывает_файлы_и_папки(self, sample, workspace):
        (workspace / "src").mkdir()
        out = list_dir(".")
        assert "sample.py" in out
        assert "src/" in out

    def test_листинг_пропускает_служебные_папки(self, workspace):
        (workspace / "__pycache__").mkdir()
        (workspace / "calc_mod.py").write_text("x = 1\n", encoding="utf-8")
        out = list_dir(".")
        assert "__pycache__" not in out
        assert "calc_mod.py" in out

    def test_листинг_за_пределы_не_ходит(self, workspace):
        assert "пределы" in list_dir("../..")

    def test_чтение_печатает_номера_строк(self, sample):
        out = read_lines("sample.py")
        assert "1| def add(a, b):" in out
        assert "всего строк: 5" in out

    def test_чтение_диапазона(self, sample):
        out = read_lines("sample.py", "4", "5")
        assert "4| def mul(a, b):" in out
        assert "def add" not in out

    def test_запрошена_строка_за_концом_файла(self, sample):
        assert "всего 5 строк" in read_lines("sample.py", "99")

    def test_конец_раньше_начала_это_сказано(self, sample):
        """Прежде такой запрос отвечал заголовком «3-1» и пустотой.

        Молчаливый бессмысленный ответ хуже отказа: из него не следует,
        что делать дальше, и модель чинит не ту беду. `apply_lines`
        про то же говорит прямо, и чтение обязано вести себя так же.
        """
        answer = read_lines("sample.py", "3", "1")
        assert "Конец раньше начала" in answer

    def test_обычный_диапазон_читается(self, sample):
        assert "1|" in read_lines("sample.py", "1", "2")

    def test_чтение_несуществующего(self, workspace):
        assert "Нет такого файла" in read_lines("missing.py")

    def test_чтение_ограничено_потолком_строк(self, workspace):
        from chapter8.src.fs import MAX_LINES

        big = workspace / "big.py"
        big.write_text("".join(f"x{n} = {n}\n" for n in range(MAX_LINES * 2)), encoding="utf-8")
        out = read_lines("big.py")
        assert f"1-{MAX_LINES}" in out

    def test_поиск_отвечает_адресом(self, sample):
        out = search_files("return a * b")
        assert "sample.py:5:" in out

    def test_поиск_не_различает_регистр(self, sample):
        assert "sample.py" in search_files("RETURN A + B")

    def test_поиск_ничего_не_нашёл(self, sample):
        assert "Ничего не найдено" in search_files("такого текста нет")

    def test_поиск_ограничен_маской(self, workspace):
        (workspace / "a.py").write_text("иголка\n", encoding="utf-8")
        (workspace / "b.md").write_text("иголка\n", encoding="utf-8")
        assert "b.md" not in search_files("иголка", "*.py")
        assert "b.md" in search_files("иголка", "*.md")

    def test_поиск_пустого_запроса(self, sample):
        assert "Пустой запрос" in search_files("  ")


# ====================================================================
# ФАЙЛОВЫЕ ИНСТРУМЕНТЫ: ЗАПИСЬ
# ====================================================================

class TestWriteTools:
    def test_создание_файла(self, workspace):
        out = write_file("new.py", "x = 1\n")
        assert "создан" in out
        assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_перезапись_с_сохранением_функций(self, sample):
        write_file("sample.py", "def add(a, b):\n    return 0\n\ndef mul(a, b):\n    return a * b\n")
        assert "return 0" in sample.read_text(encoding="utf-8")

    def test_перезапись_не_имеет_права_терять_функции(self, sample):
        """Главная беда формы «целиком»: модель возвращает одну функцию из двух."""
        before = sample.read_text(encoding="utf-8")
        out = write_file("sample.py", "def add(a, b):\n    return a + b\n")
        assert "пропали определения" in out
        assert "mul" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_правка_по_якорю(self, sample):
        out = edit_file("sample.py", "return a + b", "return a - b")
        assert "Заменено" in out
        assert "return a - b" in sample.read_text(encoding="utf-8")

    def test_правка_по_строкам(self, sample):
        out = replace_lines("sample.py", "2", "2", "    return 0")
        assert "Заменены строки 2-2" in out
        assert "return 0" in sample.read_text(encoding="utf-8")

    def test_номера_строк_приходят_текстом(self, sample):
        """Схема реестра Главы 2 объявляет все параметры строками."""
        assert replace_lines("sample.py", " 2 ", " 2 ", "    return 0").startswith("sample.py")

    def test_сломанный_синтаксис_на_диск_не_попадает(self, sample):
        before = sample.read_text(encoding="utf-8")
        out = replace_lines("sample.py", "1", "1", "def add(:")
        assert "перестаёт разбираться" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_сухой_прогон_ничего_не_пишет(self, sample):
        before = sample.read_text(encoding="utf-8")
        guard.set_policy(dry_run=True)
        out = edit_file("sample.py", "return a + b", "return a - b")
        assert "Сухой прогон" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_отказ_человека_ничего_не_пишет(self, sample):
        before = sample.read_text(encoding="utf-8")
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: False)
        out = edit_file("sample.py", "return a + b", "return a - b")
        assert "Отказано" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_человеку_показывают_diff(self, sample):
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(d) or True)
        edit_file("sample.py", "return a + b", "return a - b")
        assert "-    return a + b" in shown[0]
        assert "+    return a - b" in shown[0]

    def test_запись_за_пределы_запрещена(self, workspace):
        assert "пределы" in write_file("../escape.py", "x = 1\n")

    def test_запись_попадает_в_журнал(self, sample):
        edit_file("sample.py", "return a + b", "return a - b")
        assert guard.changed_files() == [sample]

    def test_откат_после_правки_инструментом(self, sample):
        before = sample.read_text(encoding="utf-8")
        edit_file("sample.py", "return a + b", "return a - b")
        guard.rollback()
        assert sample.read_text(encoding="utf-8") == before

    def test_неудачная_правка_в_журнал_не_попадает(self, sample):
        edit_file("sample.py", "такого текста нет", "x")
        assert guard.changed_files() == []

    def test_новый_файл_со_сломанным_синтаксисом_не_создаётся(self, workspace):
        out = write_file("broken.py", "def f(:\n")
        assert "не разбирается" in out
        assert not (workspace / "broken.py").exists()


# ====================================================================
# ЗАПУСК ПРОЦЕССОВ
# ====================================================================

class TestRunOutput:
    """Разбор вывода — без запуска процессов, на готовых текстах."""

    def test_обрезка_оставляет_начало_и_конец(self):
        text = "НАЧАЛО" + "x" * 5000 + "КОНЕЦ"
        out = clip(text, 1000)
        assert out.startswith("НАЧАЛО")
        assert out.endswith("КОНЕЦ")
        assert "вырезано" in out

    def test_короткий_вывод_не_трогается(self):
        assert clip("две строки", 1000) == "две строки"

    def test_из_отчёта_pytest_достаются_утверждения(self):
        report = (
            "collected 3 items\n"
            "F..\n"
            "E       assert 4 == 5\n"
            "E        +  where 4 = add(2, 2)\n"
            "=== 1 failed, 2 passed in 0.31s ===\n"
        )
        out = first_error(report)
        assert "assert 4 == 5" in out

    def test_из_трассировки_достаётся_исключение(self):
        trace = (
            'Traceback (most recent call last):\n'
            '  File "a.py", line 3, in <module>\n'
            "    add(1)\n"
            "TypeError: add() missing 1 required positional argument: 'b'\n"
        )
        assert first_error(trace).startswith("TypeError")

    def test_пустой_вывод(self):
        assert first_error("   ") == ""

    def test_нет_тестов_это_не_успех(self):
        """pytest возвращает 5, когда не нашёл ни одного теста."""
        assert not suite_passed(Run("pytest", 5, "no tests ran in 0.01s", "", 0.1))

    def test_нулевой_код_это_успех(self):
        assert suite_passed(Run("pytest", 0, "3 passed in 0.2s", "", 0.2))

    def test_обрыв_по_времени_это_не_успех(self):
        assert not suite_passed(Run("pytest", 0, "", "", 5.0, timed_out=True))


class TestExecute:
    """Здесь процессы действительно запускаются — но без Ollama и без сети."""

    def test_успешный_запуск(self, workspace):
        (workspace / "hi.py").write_text("print('привет')\n", encoding="utf-8")
        run = execute("python hi.py")
        assert run.ok
        assert "привет" in run.out

    def test_ненулевой_код_возврата(self, workspace):
        (workspace / "bad.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
        run = execute("python bad.py")
        assert run.code == 3
        assert not run.ok

    def test_ошибка_попадает_в_stderr(self, workspace):
        (workspace / "bad.py").write_text("raise ValueError('всё сломалось')\n", encoding="utf-8")
        run = execute("python bad.py")
        assert "ValueError" in run.text()
        assert "всё сломалось" in run.text()

    def test_предел_по_времени_снимает_процесс(self, workspace):
        (workspace / "sleep.py").write_text("import time; time.sleep(30)\n", encoding="utf-8")
        run = execute("python sleep.py", timeout=1.0)
        assert run.timed_out
        assert not run.ok
        assert "Прервано по времени" in run.summary()

    def test_несуществующая_программа(self, workspace):
        run = execute("такой-программы-нет --версия")
        assert run.code == 127
        assert "не найдена" in run.text()

    def test_команда_запускается_в_рабочем_каталоге(self, workspace):
        (workspace / "где.py").write_text("import os; print(os.getcwd())\n", encoding="utf-8")
        run = execute("python где.py")
        assert str(workspace.resolve()) in run.out

    def test_список_аргументов_не_разбирается_заново(self, workspace):
        run = execute(["python", "-c", "print('а б в')"])
        assert "а б в" in run.out


# ====================================================================
# GIT
# ====================================================================

@pytest.fixture
def repo(workspace):
    """Настоящий репозиторий git во временном каталоге.

    Настоящий, а не подделка: проверяется здесь как раз стыковка
    с программой git, и подделка проверяла бы саму себя.
    """
    if GIT is None:
        pytest.skip("git не установлен")

    def run(*args):
        return subprocess.run(
            [GIT, *args], cwd=workspace, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

    run("init", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Тест")
    run("config", "commit.gpgsign", "false")
    (workspace / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    run("add", "sample.py")
    run("commit", "-m", "первый коммит")
    return workspace


@needs_git
class TestGitRead:
    def test_текущая_ветка(self, repo):
        assert current_branch() == "main"

    def test_статус_чистого_репозитория(self, repo):
        assert "Изменений нет" in git_status()

    def test_статус_видит_правку(self, repo):
        (repo / "sample.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        assert "sample.py" in git_status()

    def test_история(self, repo):
        assert "первый коммит" in git_log("5")

    def test_сводка_изменений_без_пути(self, repo):
        (repo / "sample.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        out = git_diff()
        assert "sample.py" in out
        # Без пути показывается СВОДКА, а не изменения целиком.
        assert "-    return a + b" not in out

    def test_diff_по_пути_показывает_строки(self, repo):
        (repo / "sample.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        out = git_diff("sample.py")
        assert "-    return a + b" in out
        assert "+    return a - b" in out

    def test_не_репозиторий(self, workspace):
        assert "не репозиторий" in git_status()


@needs_git
class TestGitCommit:
    def test_коммитятся_только_файлы_агента(self, repo):
        # Агент правит один файл...
        edit_file("sample.py", "return a + b", "return a - b")
        # ...а человек параллельно другой.
        (repo / "foreign.txt").write_text("не трогать\n", encoding="utf-8")

        out = git_commit("правка агента")
        assert "Коммит создан" in out

        listed = subprocess.run(
            [GIT, "show", "--name-only", "--format=", "HEAD"],
            cwd=repo, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        assert "sample.py" in listed
        assert "foreign.txt" not in listed

    def test_без_правок_коммитить_нечего(self, repo):
        assert "коммитить нечего" in git_commit("пустой коммит")

    def test_пустое_сообщение_отвергается(self, repo):
        edit_file("sample.py", "return a + b", "return a - b")
        assert "Пустое сообщение" in git_commit("   ")

    def test_сухой_прогон_не_коммитит(self, repo):
        edit_file("sample.py", "return a + b", "return a - b")
        guard.set_policy(dry_run=True)
        assert "Сухой прогон" in git_commit("не должно случиться")
        assert "первый коммит" in git_log("5")
        assert "правка" not in git_log("5")

    def test_отказ_человека_не_коммитит(self, repo):
        edit_file("sample.py", "return a + b", "return a - b")
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: False)
        assert "Отказано" in git_commit("не должно случиться")

    def test_человеку_показывают_файлы_и_сообщение(self, repo):
        edit_file("sample.py", "return a + b", "return a - b")
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(d) or True)
        git_commit("правка агента")
        assert "правка агента" in shown[0]
        assert "sample.py" in shown[0]

    def test_после_коммита_журнал_пуст(self, repo):
        edit_file("sample.py", "return a + b", "return a - b")
        git_commit("правка агента")
        assert guard.changed_files() == []


# ====================================================================
# РЕЕСТР
# ====================================================================

class TestRegistry:
    def test_все_инструменты_главы_зарегистрированы(self):
        from chapter2.src.tools import TOOL_REGISTRY

        for name in FS_TOOLS + RUN_TOOLS + GIT_TOOLS + ENV_TOOLS:
            assert name in TOOL_REGISTRY, name

    def test_у_каждого_есть_описание(self):
        from chapter2.src.tools import TOOL_REGISTRY

        for name in FS_TOOLS + RUN_TOOLS + GIT_TOOLS + ENV_TOOLS:
            description = TOOL_REGISTRY[name]["schema"]["function"]["description"]
            assert description and description != "Нет описания", name

    def test_новых_инструментов_двадцать(self):
        assert len(FS_TOOLS) + len(RUN_TOOLS) + len(GIT_TOOLS) + len(ENV_TOOLS) == 20


# ====================================================================
# ОКРУЖЕНИЕ И ЗАВИСИМОСТИ
# ====================================================================

class TestImports:
    """Что импортирует код — считается разбором, а не спрашивается у модели."""

    def test_импорты_собираются_разбором(self):
        text = "import os\nimport requests\nfrom bs4 import BeautifulSoup\nfrom . import local\n"
        assert env.imported_modules(text) == ["os", "requests", "bs4"]

    def test_точечный_импорт_даёт_корень(self):
        assert env.imported_modules("import os.path\nimport a.b.c\n") == ["os", "a"]

    def test_слово_import_в_строке_не_считается(self):
        assert env.imported_modules('Ж = "import requests"\nimport os\n') == ["os"]

    def test_повторы_схлопываются(self):
        assert env.imported_modules("import os\nimport os\nfrom os import path\n") == ["os"]

    def test_сломанный_файл_не_роняет_разбор(self):
        assert env.imported_modules("import (((") == []

    def test_свои_модули_проекта_видны(self, workspace):
        (workspace / "calc.py").write_text("x = 1\n", encoding="utf-8")
        (workspace / "pkg").mkdir()
        (workspace / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        assert {"calc", "pkg"} <= env.local_modules()

    def test_свой_модуль_не_считается_недостающим(self, workspace):
        """Иначе агент пошёл бы ставить из сети пакет с именем своего файла."""
        (workspace / "calc.py").write_text("x = 1\n", encoding="utf-8")
        assert env.missing_imports("import calc\nimport os\n") == []

    def test_имя_пакета_не_всегда_имя_модуля(self):
        assert env.package_for("yaml") == "pyyaml"
        assert env.package_for("bs4") == "beautifulsoup4"
        assert env.package_for("requests") == "requests"


class TestEnvTools:
    def test_отчёт_без_окружения_называет_интерпретатор_агента(self, workspace):
        assert "Окружения проекта нет" in env.env_report()
        assert not env.has_venv()

    def test_проверка_импортов_несуществующего_файла(self, workspace):
        assert "Нет такого файла" in env.check_imports("нетакого.py")

    def test_установка_без_имён(self, workspace):
        assert "Не названо ни одного пакета" in env.install("   ")

    @pytest.mark.parametrize("bad", ["--upgrade", "a;rm", "a b|c"])
    def test_ключ_вместо_имени_пакета_отвергается(self, workspace, bad):
        assert "Недопустимые имена" in env.install(bad)

    def test_установка_спрашивает_и_показывает_пакеты(self, workspace):
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(d) or False)
        out = env.install("requests bs4")
        assert "Отказано" in out
        assert "requests" in shown[0] and "bs4" in shown[0]
        assert "из сети" in shown[0], "человек должен видеть, что это загрузка кода из сети"

    def test_сухой_прогон_не_ставит(self, workspace):
        guard.set_policy(dry_run=True)
        assert "Сухой прогон" in env.install("requests")

    def test_requirements_без_окружения_не_пишется(self, workspace):
        """pip freeze без .venv перечислил бы все пакеты машины."""
        out = env.write_requirements()
        assert "Сначала create_venv" in out
        assert not (workspace / "requirements.txt").exists()


# ====================================================================
# ПЛАНИРОВЩИК
# ====================================================================

class TestPlanSchema:
    def test_действия_схемы_совпадают_со_списком(self):
        """Промпт, схема и проверка обязаны говорить об одних действиях."""
        schema = plan_schema()
        enum = schema["properties"]["steps"]["items"]["properties"]["action"]["enum"]
        assert enum == list(PLAN_ACTIONS)

    def test_установки_пакетов_в_языке_плана_нет(self):
        """Зависимости вычисляются из кода, а не назначаются моделью."""
        assert "deps" not in PLAN_ACTIONS
        assert "install" not in PLAN_ACTIONS

    def test_потолок_шагов_в_схеме(self):
        assert plan_schema()["properties"]["steps"]["maxItems"] == MAX_STEPS

    def test_промпт_перечисляет_все_действия(self, workspace):
        prompt = build_planner_prompt("поправь сложение", files=["a.py"])
        for action in PLAN_ACTIONS:
            assert action in prompt
        assert "a.py" in prompt

    def test_в_пустом_каталоге_промпт_другой(self, workspace):
        """В пустом каталоге правило «бери путь из списка» невыполнимо."""
        empty = build_planner_prompt("напиши приложение", files=[])
        full = build_planner_prompt("напиши приложение", files=["a.py"])
        assert "придумай понятные имена файлов" in empty
        assert "придумай понятные имена файлов" not in full
        assert "Не выдумывай новых путей" in full

    def test_английская_версия_говорит_то_же_самое(self, workspace):
        """Два языка — один промпт. Иначе замер сравнивал бы не язык."""
        ru = build_planner_prompt("поправь сложение", files=["a.py"], language="ru")
        en = build_planner_prompt("поправь сложение", files=["a.py"], language="en")
        assert ru != en
        for prompt in (ru, en):
            for action in PLAN_ACTIONS:
                assert action in prompt
            assert "a.py" in prompt
            assert "поправь сложение" in prompt, "задача идёт человеку как есть, её не переводят"
            assert str(MAX_STEPS) in prompt

    def test_английский_в_пустом_каталоге_тоже_запрещает_файл_на_функцию(self, workspace):
        empty = build_planner_prompt("напиши приложение", files=[], language="en")
        full = build_planner_prompt("напиши приложение", files=["a.py"], language="en")
        assert "ONE PROGRAM — ONE FILE" in empty
        assert "ONE PROGRAM — ONE FILE" not in full
        assert "Do not invent new paths" in full

    def test_язык_по_умолчанию_берётся_из_окружения(self, workspace, monkeypatch):
        monkeypatch.setattr(planner_module, "PLANNER_LANG", "en")
        assert "You plan work on code" in build_planner_prompt("задача", files=["a.py"])
        monkeypatch.setattr(planner_module, "PLANNER_LANG", "ru")
        assert "Ты составляешь план" in build_planner_prompt("задача", files=["a.py"])

    def test_неизвестный_язык_это_русский(self, workspace):
        """Опечатка в переменной окружения не должна ронять планировщик."""
        assert "Ты составляешь план" in build_planner_prompt("з", files=["a.py"], language="de")


class TestParsePlan:
    def test_разбор_годного_ответа(self):
        raw = json.dumps({"steps": [
            {"action": "create", "target": "app.py", "detail": "точка входа", "why": "задача"},
            {"action": "test", "target": "", "detail": "прогнать"},
        ]})
        steps, problems = parse_plan(raw)
        assert [s.action for s in steps] == ["create", "test"]
        assert steps[0].target == "app.py"
        assert problems == []

    def test_не_json(self):
        steps, problems = parse_plan("извольте, вот план:")
        assert steps == []
        assert "JSON" in problems[0]

    def test_нет_списка_шагов(self):
        steps, problems = parse_plan('{"plan": "поправить"}')
        assert steps == []
        assert "steps" in problems[0]

    def test_неизвестное_действие_отбрасывается(self):
        raw = json.dumps({"steps": [{"action": "deploy", "target": "", "detail": ""},
                                    {"action": "test", "target": "", "detail": ""}]})
        steps, problems = parse_plan(raw)
        assert [s.action for s in steps] == ["test"]
        assert "deploy" in problems[0]

    def test_слишком_длинный_план_обрезается(self):
        raw = json.dumps({"steps": [{"action": "read", "target": "a.py", "detail": "x"}] * (MAX_STEPS + 3)})
        steps, problems = parse_plan(raw)
        assert len(steps) == MAX_STEPS
        assert any("оставлены первые" in p for p in problems)

    def test_шаг_не_объект(self):
        steps, problems = parse_plan(json.dumps({"steps": ["почини всё"]}))
        assert steps == []
        assert "не объект" in problems[0]


class TestSplitTarget:
    """Путь, склеенный моделью с объяснением. Нашла живая модель."""

    def test_описание_отрезается_по_существующему_файлу(self, sample):
        assert split_target("sample.py: почему тут ошибка?") == ("sample.py", "почему тут ошибка?")

    def test_чистый_путь_не_трогается(self, sample):
        assert split_target("sample.py") == ("sample.py", "")

    def test_несуществующий_файл_не_режется(self, workspace):
        assert split_target("нет.py: описание") == ("нет.py: описание", "")

    def test_путь_с_двоеточием_диска_не_ломается(self, workspace):
        text = "C:\\work\\a.py"
        assert split_target(text) == (text, "")

    def test_пустой_путь(self, workspace):
        assert split_target("   ") == ("", "")

    def test_разбор_плана_раскладывает_склеенное_поле(self, sample):
        raw = json.dumps({"steps": [
            {"action": "edit", "target": "sample.py: почему тут ошибка?", "detail": ""},
            {"action": "test", "target": "", "detail": "прогнать"},
        ]})
        steps, _ = parse_plan(raw)
        assert steps[0].target == "sample.py"
        assert steps[0].detail == "почему тут ошибка?"

    def test_поиск_не_режется(self, sample):
        raw = json.dumps({"steps": [
            {"action": "search", "target": "sample.py: где сложение", "detail": ""},
            {"action": "test", "target": "", "detail": "x"},
        ]})
        steps, _ = parse_plan(raw)
        assert steps[0].target == "sample.py: где сложение"


class TestValidatePlan:
    def test_годный_план_правки(self, sample):
        plan = Plan("задача", [Step("edit", "sample.py", "поправить"), Step("test", "", "прогнать")])
        assert validate_plan(plan) == []

    def test_годный_план_с_нуля(self, workspace):
        plan = Plan("задача", [Step("create", "app.py", "точка входа"),
                               Step("create", "test_app.py", "тесты"), Step("test", "", "")])
        assert validate_plan(plan) == []

    def test_пустой_план(self):
        assert validate_plan(Plan("задача", [])) == ["план пуст"]

    def test_последний_шаг_не_проверка(self, sample):
        plan = Plan("задача", [Step("edit", "sample.py", "поправить")])
        assert any("не test" in p for p in validate_plan(plan))

    def test_правка_несуществующего_файла(self, workspace):
        plan = Plan("задача", [Step("edit", "нет.py", "поправить"), Step("test", "", "")])
        assert any("нет.py" in p and "create" in p for p in validate_plan(plan))

    def test_создание_существующего_файла(self, sample):
        plan = Plan("задача", [Step("create", "sample.py", "создать"), Step("test", "", "")])
        assert any("уже есть" in p for p in validate_plan(plan))

    def test_правка_сразу_после_записи_это_претензия(self, workspace):
        """Живой прогон так удвоил содержимое батника."""
        plan = Plan("з", [Step("create", "a.bat", "x"), Step("edit", "a.bat", "y"), Step("test", "", "")])
        claims = validate_plan(plan)
        assert any("сразу после его create" in c for c in claims)
        assert not any("для нового нужен create" in c for c in claims), "две претензии на один шаг"

    def test_создание_двух_файлов_претензий_не_вызывает(self, workspace):
        plan = Plan("з", [Step("create", "a.py", "x"), Step("create", "test_a.py", "y"),
                          Step("test", "", "")])
        assert validate_plan(plan) == []

    def test_путь_за_пределами_каталога(self, workspace):
        plan = Plan("задача", [Step("edit", "../чужое.py", "поправить"), Step("test", "", "")])
        assert any("вне рабочего каталога" in p for p in validate_plan(plan))

    def test_поиск_без_запроса(self, workspace):
        plan = Plan("задача", [Step("search", "", "найти"), Step("test", "", "")])
        assert any("search без запроса" in p for p in validate_plan(plan))

    def test_правка_без_пути_после_поиска_законна(self, workspace):
        plan = Plan("з", [Step("search", "add", "найти"), Step("edit", "", "поправить"), Step("test", "", "")])
        assert validate_plan(plan) == []

    def test_правка_без_пути_и_без_поиска_это_претензия(self, workspace):
        plan = Plan("задача", [Step("edit", "", "поправить"), Step("test", "", "")])
        assert any("без пути" in p for p in validate_plan(plan))


class TestFixOrScratch:
    """Развилка планировщика: что строит код, а что остаётся модели.

    К нынешнему виду она пришла тремя живыми прогонами подряд, и каждый
    показывал одно: план от модели портит задачу. Он пересказывает её
    короче, теряет требования, а на задаче про квадратное уравнение
    развалил одну маленькую программу на четыре файла — по файлу
    на функцию. Поэтому кодом строится всё, что выводится однозначно,
    а у модели остаётся ровно один вопрос — как назвать файл.
    """

    def test_создать_названный_новый_файл(self, workspace):
        task = "сделай калькулятор, назови его calc.py, он должен ждать ввода"
        assert plan_kind(task) == ("one_file", "calc.py")

    def test_создать_названный_существующий_это_перезапись(self, sample):
        """Повтор задачи — законная перезапись, а не правка."""
        assert plan_kind("сделай sample.py заново") == ("one_file", "sample.py")

    def test_поправить_существующий(self, sample):
        assert plan_kind("поправь sample.py, сложение неверное") == ("fix", "sample.py")

    def test_правка_без_повелительного_глагола(self, sample):
        """Самая частая форма просьбы, и глагола в ней нет вовсе."""
        assert plan_kind("в sample.py функция add вычитает вместо сложения") == ("fix", "sample.py")

    def test_поправить_несуществующий_это_написать_его(self, workspace):
        """Править нечего — значит надо написать, и имя уже названо."""
        assert plan_kind("поправь нетакого.py") == ("needs_name", "")

    def test_просят_файл_другого_вида(self, sample):
        """«сделай bat для запуска sample.py»: имя есть, но это ссылка."""
        assert plan_kind("сделай bat файл для запуска sample.py") == ("needs_name", "")
        assert plan_kind("сделай батник для запуска sample.py") == ("needs_name", "")

    def test_названный_батник_это_цель(self, sample):
        assert plan_kind("сделай run.bat для запуска sample.py") == ("one_file", "run.bat")

    def test_без_имени_файла_имя_придётся_спросить(self, workspace):
        assert plan_kind("напиши приложение hello world") == ("needs_name", "")

    def test_две_вещи_в_одной_задаче_идут_к_модели(self, workspace):
        """«и затем» означает разные артефакты — вот тут план нужен."""
        assert plan_kind("сделай calc.py и затем сделай bat файл") == ("model", "")
        assert plan_kind("напиши lib.py, а также набросай README") == ("model", "")

    def test_точка_в_конце_предложения_именем_файла_не_является(self, workspace):
        assert named_file("сделай всё как надо. и побыстрее") == ""

    def test_план_правки(self, sample):
        plan = fallback_plan("поправь sample.py, сложение неверное")
        assert [s.action for s in plan.steps] == ["search", "edit", "test"]
        assert plan.steps[0].target == "sample.py"

    def test_план_записи_одного_файла(self, workspace):
        task = "сделай калькулятор, назови его calc.py, он должен ждать ввода"
        plan = fallback_plan(task)
        assert [s.action for s in plan.steps] == ["create", "test"]
        assert plan.steps[0].target == "calc.py"
        assert plan.steps[0].detail == task, "формулировка человека обязана доехать целиком"

    def test_тесты_заводятся_только_если_их_просят(self, workspace):
        """Тест, которого не просили, всё равно нечем проверить на правильность."""
        without = fallback_plan("сделай calc.py — калькулятор")
        assert [s.action for s in without.steps] == ["create", "test"]

        with_tests = fallback_plan("сделай calc.py — калькулятор, и тесты к нему")
        assert [s.action for s in with_tests.steps] == ["create", "create", "test"]
        assert with_tests.steps[1].target == "test_calc.py"

    def test_без_имени_запасной_план_берёт_main(self, workspace):
        """Без модели план всё равно должен получиться: Ollama может быть не запущена."""
        plan = fallback_plan("напиши приложение hello world")
        assert [s.action for s in plan.steps] == ["create", "test"]
        assert plan.steps[0].target == "main.py"

    def test_несколько_вещей_кодом_не_планируются(self, workspace):
        plan = fallback_plan("сделай calc.py и затем bat файл")
        assert plan.steps == []
        assert "больше одной вещи" in plan.problems[0]


class TestEditWithoutName:
    """Правка без названного файла — это правка, а не «напиши новое».

    Живой прогон: человек написал «добавь после print(f'd/dx = {df_dx}')
    ожидание ввода», имея в виду main.py, написанный пятью минутами
    раньше. План вышел «create quadratic.py» — новый файл с именем
    от модели, рядом с тем, который просили поправить. Глагол в задаче
    был «добавь», то есть правка, а код читал его как «напиши что-нибудь».
    """

    @pytest.fixture
    def project_pair(self, workspace):
        """Модуль и батник рядом. Батник моложе — он и есть «последний»."""
        (workspace / "main.py").write_text(
            "def main():\n    df_dx = 2\n    print(f'd/dx = {df_dx}')\n", encoding="utf-8")
        time.sleep(0.05)
        (workspace / "run.bat").write_text("@echo off\npython main.py\n", encoding="utf-8")
        return workspace

    def test_цитата_из_кода_адресует_файл(self, project_pair):
        """`df_dx` есть ровно в одном файле — по нему и найдётся."""
        task = "добавь после print(f'd/dx = {df_dx}') ожидание ввода любой кнопки"
        kind, target = plan_kind(task)
        assert kind == "fix"
        assert target == "df_dx", "адресом может быть и слово: искать умеет шаг search"
        assert [s.action for s in fallback_plan(task).steps] == ["search", "edit", "test"]

    def test_без_цитаты_берётся_последний_изменённый(self, project_pair):
        task = "исправь: оно не должно закрываться сразу после вывода результата"
        assert plan_kind(task) == ("fix", "run.bat")

    def test_частые_слова_адресом_не_считаются(self, project_pair):
        """`print` есть в каждом втором файле и не адресует ничего."""
        assert planner_module.code_word("добавь print в конец") == ""

    def test_самое_длинное_слово_а_не_первое(self):
        assert planner_module.code_word("поправь calc_total в модуле") == "calc_total"

    def test_слова_которых_в_проекте_нет_не_адресуют(self, project_pair):
        """Выдуманное имя не должно уводить поиск в никуда — берём последний файл."""
        assert plan_kind("поправь незнакомое_имя_которого_нет") == ("fix", "run.bat")

    def test_память_важнее_времени_изменения(self, project_pair, monkeypatch):
        """«Последний по времени» — это и файл, который человек открыл сам."""
        monkeypatch.setattr(planner_module, "remembered_file", lambda: "main.py")
        assert plan_kind("исправь: оно не должно закрываться сразу") == ("fix", "main.py")

    def test_чужая_память_не_адресует(self, project_pair, monkeypatch):
        """Память живёт дольше каталога: файла из прошлого проекта здесь нет."""
        monkeypatch.setattr(planner_module, "remembered_file", lambda: "")
        assert plan_kind("исправь: оно не должно закрываться сразу") == ("fix", "run.bat")

    def test_глагол_создания_по_прежнему_создаёт(self, project_pair):
        assert plan_kind("напиши приложение которое считает факториал") == ("needs_name", "")

    def test_названный_файл_важнее_догадки(self, project_pair):
        assert plan_kind("поправь main.py, сложение неверное") == ("fix", "main.py")

    def test_названный_несуществующий_остаётся_созданием(self, project_pair):
        """Регрессия, которую первая версия ветки внесла в обратную сторону.

        «Поправь нетакого.py» — файл НАЗВАН, и его нет; это «напиши его».
        Без проверки `not named` ветка «правка без названия» срабатывала
        и здесь, адресом становился последний изменённый файл — и агент
        шёл править посторонние заметки вместо того, чтобы создать
        названный файл.
        """
        assert plan_kind("поправь нетакого.py") == ("needs_name", "")
        assert plan_kind("исправь ошибки в new.py") == ("needs_name", "")

    def test_адрес_ищется_по_всему_дереву(self, workspace):
        """Проект из одного файла бывает только в учебнике."""
        (workspace / "src").mkdir()
        (workspace / "src" / "calc.py").write_text("def total():\n    return 0\n", encoding="utf-8")
        kind, target = plan_kind("поправь total, считает неверно")
        assert (kind, target) == ("fix", "total")

    def test_регистр_в_цитате_не_мешает(self, workspace):
        (workspace / "calc.py").write_text("df_dx = 1\n", encoding="utf-8")
        assert plan_kind("поправь DF_DX") == ("fix", "DF_DX")

    def test_заметки_адресом_не_становятся(self, workspace):
        """Свежий README тронут по другому поводу, чем «оно не запускается»."""
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        time.sleep(0.05)
        (workspace / "README.md").write_text("# проект\n", encoding="utf-8")
        assert plan_kind("исправь: оно не должно закрываться") == ("fix", "app.py")

    def test_в_пустом_каталоге_правка_остаётся_без_адреса(self, workspace):
        """Править нечего — значит, надо писать, и это прежняя ветка."""
        assert plan_kind("добавь ожидание ввода") == ("needs_name", "")


class TestAskFilename:
    """Единственное, что кодом не выводится, — имя файла."""

    def test_имя_от_модели(self, workspace, monkeypatch):
        monkeypatch.setattr(planner_module, "request_model",
                            lambda *a, **k: json.dumps({"filename": "quadratic.py"}))
        assert planner_module.ask_filename("реши квадратное уравнение") == "quadratic.py"

    def test_модель_не_ответила(self, workspace, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("Ollama молчит")

        monkeypatch.setattr(planner_module, "request_model", boom)
        assert planner_module.ask_filename("задача") == "main.py"

    @pytest.mark.parametrize("bad", ["", "приложение", "app", "два слова.py", "calc.exe"])
    def test_негодное_имя_заменяется_запасным(self, workspace, monkeypatch, bad):
        monkeypatch.setattr(planner_module, "request_model",
                            lambda *a, **k: json.dumps({"filename": bad}))
        assert planner_module.ask_filename("задача") == "main.py"

    @pytest.mark.parametrize("given", ["src/calc.py", r"..\calc.py", "../calc.py"])
    def test_путь_срезается_до_имени(self, workspace, monkeypatch, given):
        """Каталог выбираем не мы и не модель: файл ложится в рабочий."""
        monkeypatch.setattr(planner_module, "request_model",
                            lambda *a, **k: json.dumps({"filename": given}))
        assert planner_module.ask_filename("задача") == "calc.py"

    def test_схема_просит_ровно_одно_поле(self):
        schema = planner_module.file_name_schema()
        assert list(schema["properties"]) == ["filename"]
        assert schema["required"] == ["filename"]

    def test_план_собирается_вокруг_полученного_имени(self, workspace, monkeypatch):
        monkeypatch.setattr(planner_module, "request_model",
                            lambda *a, **k: json.dumps({"filename": "quadratic.py"}))
        task = "напиши приложение, решающее квадратное уравнение"
        plan = make_plan(task)
        assert plan.source == FROM_FALLBACK, "план строит код, модель дала только имя"
        assert [s.action for s in plan.steps] == ["create", "test"]
        assert plan.steps[0].target == "quadratic.py"
        assert plan.steps[0].detail == task


class TestOverwrite:
    """Явный `create` перезаписывает файл, инструмент `write_file` — нет."""

    def test_шаг_создания_переписывает_существующий(self, sample, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")]))
        state = started(Plan("з", [Step("create", "sample.py", "напиши заново")]))
        pipeline.node_step(state)
        assert sample.read_text(encoding="utf-8") == "x = 1\n"
        assert state.extra["failed_steps"] == []

    def test_инструмент_терять_определения_не_даёт(self, sample):
        """У модели в руках не должно быть флага «можно терять код»."""
        before = sample.read_text(encoding="utf-8")
        out = write_file("sample.py", "def add(a, b):\n    return a + b\n")
        assert "пропали определения" in out
        assert sample.read_text(encoding="utf-8") == before

    def test_человеку_показывают_что_пропадёт(self, sample):
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(d) or False)
        put_file("sample.py", "def add(a, b):\n    return a + b\n", replace=True)
        assert "пропадут определения" in shown[0]
        assert "mul" in shown[0]

    def test_перезапись_откатывается(self, sample):
        before = sample.read_text(encoding="utf-8")
        put_file("sample.py", "x = 1\n", replace=True)
        guard.rollback()
        assert sample.read_text(encoding="utf-8") == before

    def test_сломанный_файл_не_пишется_и_при_перезаписи(self, sample):
        before = sample.read_text(encoding="utf-8")
        out = put_file("sample.py", "def f(:\n", replace=True)
        assert "перестаёт разбираться" in out
        assert sample.read_text(encoding="utf-8") == before


class TestMakePlan:
    def test_план_от_модели(self, workspace, monkeypatch):
        answer = json.dumps({"steps": [
            {"action": "create", "target": "app.py", "detail": "точка входа"},
            {"action": "test", "target": "", "detail": "прогнать"},
        ]})
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: answer)
        plan = make_plan("напиши приложение", model="qwen2.5:3b")
        assert plan.source == FROM_MODEL
        assert plan.model == "qwen2.5:3b"
        assert len(plan.steps) == 2

    def test_модель_не_ответила_даёт_запасной_план(self, sample, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("Ollama не запущена")

        monkeypatch.setattr(planner_module, "request_model", boom)
        plan = make_plan("поправь sample.py", use_model=True)
        assert plan.source == FROM_FALLBACK
        assert "Ollama не запущена" in plan.problems[0]
        assert plan.steps, "для задачи правки запасной план не должен быть пустым"

    def test_мусор_от_модели_даёт_запасной_план(self, sample, monkeypatch):
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: "не буду")
        plan = make_plan("поправь sample.py", use_model=True)
        assert plan.source == FROM_FALLBACK
        assert plan.problems

    def test_пустая_задача(self, workspace):
        plan = make_plan("   ")
        assert plan.steps == []
        assert "пустая задача" in plan.problems

    def test_схема_уезжает_в_модель(self, workspace, monkeypatch):
        seen = {}

        def capture(messages, response_format=None):
            seen["format"] = response_format
            return json.dumps({"steps": [{"action": "test", "target": "", "detail": "x"}]})

        monkeypatch.setattr(planner_module, "request_model", capture)
        make_plan("задача", use_model=True)
        assert seen["format"] == plan_schema()


class TestPlannerDefault:
    """Кто составляет план по умолчанию — вывод замера, а не вкус."""

    def test_правка_названного_файла_идёт_без_модели(self, sample, monkeypatch):
        asked = []
        monkeypatch.setattr(planner_module, "PLANNER", "fallback")
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: asked.append(1) or "{}")
        plan = make_plan("поправь sample.py, сложение неверное")
        assert plan.source == FROM_FALLBACK
        assert asked == [], "на задаче правки модель не должна вызываться"

    def test_задача_с_нуля_планируется_кодом(self, workspace, monkeypatch):
        """У модели спрашивают имя файла, а не план."""
        monkeypatch.setattr(planner_module, "PLANNER", "fallback")
        monkeypatch.setattr(planner_module, "request_model",
                            lambda *a, **k: json.dumps({"filename": "hello.py"}))
        plan = make_plan("напиши приложение hello world")
        assert plan.source == FROM_FALLBACK
        assert plan.steps[0].target == "hello.py"

    def test_несколько_вещей_идут_к_планировщику(self, workspace, monkeypatch):
        answer = json.dumps({"steps": [{"action": "create", "target": "app.py", "detail": "x"},
                                       {"action": "test", "target": "", "detail": "x"}]})
        monkeypatch.setattr(planner_module, "PLANNER", "fallback")
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: answer)
        assert make_plan("напиши app.py и затем README").source == FROM_MODEL

    def test_переменной_окружения_модель_включается_везде(self, sample, monkeypatch):
        answer = json.dumps({"steps": [{"action": "test", "target": "", "detail": "x"}]})
        monkeypatch.setattr(planner_module, "PLANNER", "model")
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: answer)
        assert make_plan("поправь sample.py").source == FROM_MODEL

    def test_явно_названная_модель_включает_её_сама(self, sample, monkeypatch):
        answer = json.dumps({"steps": [{"action": "test", "target": "", "detail": "x"}]})
        monkeypatch.setattr(planner_module, "PLANNER", "fallback")
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: answer)
        assert make_plan("поправь sample.py", model="qwen2.5:3b").source == FROM_MODEL


class TestPlanRendering:
    def test_отчёт_называет_источник(self, sample):
        assert "запасной план без модели" in render_plan(fallback_plan("поправь sample.py"))

    def test_отчёт_называет_модель(self, sample):
        plan = Plan("з", [Step("edit", "sample.py", "поправить"), Step("test", "", "")],
                    source=FROM_MODEL, model="qwen2.5:3b", seconds=1.5)
        text = render_plan(plan)
        assert "qwen2.5:3b" in text
        assert "1.5 с" in text

    def test_претензии_показываются_вместе_с_планом(self, workspace):
        text = render_plan(Plan("з", [Step("edit", "нет.py", "поправить")]))
        assert "Претензии к плану" in text
        assert "нет.py" in text

    def test_пустой_план(self):
        assert render_plan(Plan("з", [])) == "План пуст."

    def test_план_переживает_словарь(self, workspace):
        plan = Plan("з", [Step("create", "a.py", "написать", "затем")], source=FROM_MODEL, model="m")
        again = Plan.from_dict(plan.to_dict())
        assert again.steps[0].to_dict() == plan.steps[0].to_dict()
        assert again.source == FROM_MODEL and again.model == "m"

    def test_незнакомые_поля_словаря_отбрасываются(self):
        plan = Plan.from_dict({"task": "з", "steps": [{"action": "test", "лишнее": 1}], "версия": 9})
        assert plan.steps[0].action == "test"


# ====================================================================
# КОНВЕЙЕР: ПОДДЕЛКИ
# ====================================================================

def fake_model(answers):
    """Подделка модели: отдаёт заготовленные ответы по очереди.

    По очереди, а не один и тот же — иначе не проверить главное свойство
    конвейера: второй круг должен отличаться от первого.
    """
    queue = list(answers)
    source = list(answers)

    def call(messages, response_format=None, **kwargs):
        call.seen.append(messages)
        call.options.append(kwargs.get("options"))
        return queue.pop(0) if queue else (source[-1] if source else "{}")

    call.seen = []
    call.options = []
    return call


def edit_answer(path, start, end, content, form="lines"):
    """Ответ модели с правкой в форме «строки»."""
    return json.dumps({"form": form, "path": path, "start": start, "end": end, "content": content})


def anchor_answer(path, old, new):
    """Ответ модели с правкой по якорю."""
    return json.dumps({"form": "anchor", "path": path, "old": old, "new": new})


def file_answer(content):
    """Ответ модели на «напиши файл целиком»."""
    return json.dumps({"content": content})


def fake_run(green, output=""):
    """Подделка запуска процесса: удачный или нет."""
    def run(command, timeout=None, feed=""):
        return Run(command, 0 if green else 1, output or ("2 passed" if green else "E  assert 4 == 5"), "", 0.2)
    return run


@pytest.fixture
def project(workspace):
    """Крошечный проект: модуль со сломанным сложением и тест к нему."""
    (workspace / "calc_mod.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (workspace / "test_calc_mod.py").write_text(
        "from calc_mod import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n", encoding="utf-8"
    )
    return workspace


@pytest.fixture
def fix_plan():
    """План на два шага: поправить calc_mod.py и прогнать тесты."""
    return Plan(
        "почини сложение",
        [Step("edit", "calc_mod.py", "вернуть a + b вместо a - b"), Step("test", "", "прогнать тесты")],
    )


def started(plan, **extra):
    """Состояние так, как его оставляет узел плана. Для тестов узлов поодиночке."""
    state = State(user_input=plan.task)
    state.extra["plan"] = plan.to_dict()
    state.extra.update(extra)
    return pipeline.node_plan(state)


# ====================================================================
# КОНВЕЙЕР: ФОРМА ГРАФА
# ====================================================================

class TestPipelineShape:
    def test_граф_собирается_без_претензий(self):
        assert build_pipeline().validate() == []

    def test_девять_узлов(self):
        assert set(build_pipeline().nodes) == {
            pipeline.PLAN, pipeline.CONFIRM, pipeline.STEP, pipeline.DEPS, pipeline.VERIFY,
            pipeline.READ_NODE, pipeline.EDIT_NODE, pipeline.ROLLBACK, pipeline.DONE,
        }

    def test_шаг_возвращается_сам_в_себя(self):
        """Первая стрелка назад: цикл по шагам плана."""
        assert pipeline.STEP in build_pipeline().targets[pipeline.STEP]

    def test_из_проверки_можно_вернуться_к_чтению(self):
        """Вторая стрелка назад — цикл починки.

        Ведёт она к чтению, а не сразу к правке: после неудачной попытки
        файл на диске другой, и модель должна увидеть его новым.
        """
        targets = build_pipeline().targets[pipeline.VERIFY]
        assert pipeline.READ_NODE in targets
        assert pipeline.EDIT_NODE not in targets

    def test_зависимости_стоят_после_шагов_и_до_проверки(self):
        graph = build_pipeline()
        assert pipeline.DEPS in graph.targets[pipeline.STEP]
        assert graph.edges[pipeline.DEPS] == pipeline.VERIFY


# ====================================================================
# КОНВЕЙЕР: ОТДЕЛЬНЫЕ ШАГИ
# ====================================================================

class TestStepLoop:
    def test_шаги_идут_по_очереди(self, project, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")] * 3))
        plan = Plan("з", [Step("create", "a.py", "раз"), Step("create", "b.py", "два"),
                          Step("test", "", "три")])
        state = started(plan)
        for _ in range(3):
            pipeline.node_step(state)
        assert state.extra["cursor"] == 3
        assert len(state.extra["log"]) == 3
        assert state.extra["log"][0].startswith("1. create a.py")

    def test_ребро_гоняет_по_кругу_пока_шаги_не_кончатся(self, project):
        plan = Plan("з", [Step("read", "calc_mod.py", ""), Step("test", "", "")])
        state = started(plan)
        assert pipeline.edge_after_step(state) == pipeline.STEP
        state.extra["cursor"] = 2
        assert pipeline.edge_after_step(state) == pipeline.DEPS

    def test_курсор_за_концом_плана_ничего_не_делает(self, project):
        state = started(Plan("з", [Step("test", "", "")]), cursor=5)
        pipeline.node_step(state)
        assert state.extra["log"] == []


class TestStepSearch:
    def test_имя_файла_берётся_как_путь(self, project):
        state = started(Plan("з", [Step("search", "calc_mod.py", "")]))
        pipeline.node_step(state)
        assert state.extra["path"] == "calc_mod.py"

    def test_не_файл_ищется_текстом(self, project):
        state = started(Plan("з", [Step("search", "def add", "")]))
        pipeline.node_step(state)
        assert state.extra["path"] == "calc_mod.py"

    def test_ничего_не_нашлось(self, project):
        state = started(Plan("з", [Step("search", "такого текста нет нигде", "")]))
        pipeline.node_step(state)
        assert "ничего не нашлось" in state.extra["log"][0]


class TestStepCreate:
    def test_файл_пишется_целиком(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("print('привет')\n")]))
        state = started(Plan("з", [Step("create", "app.py", "печатает привет")]))
        pipeline.node_step(state)
        assert (workspace / "app.py").read_text(encoding="utf-8") == "print('привет')\n"
        assert state.extra["touched"] == ["app.py"]

    def test_путь_берётся_из_плана_а_не_у_модели(self, workspace, monkeypatch):
        """В схеме ответа поля path нет: писать не туда модели нечем."""
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")]))
        assert "path" not in pipeline.file_schema()["properties"]
        state = started(Plan("з", [Step("create", "нужный.py", "")]))
        pipeline.node_step(state)
        assert (workspace / "нужный.py").exists()

    def test_обёртка_из_обратных_кавычек_снимается(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([file_answer("```python\nx = 1\n```")]))
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert (workspace / "app.py").read_text(encoding="utf-8").strip() == "x = 1"

    def test_пустой_ответ_модели(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("   ")]))
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert "пустой файл" in state.extra["log"][0]
        assert not (workspace / "app.py").exists()

    def test_модель_не_ответила(self, workspace, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("Ollama молчит")

        monkeypatch.setattr(pipeline, "request_model", boom)
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert "не написала файл" in state.extra["log"][0]

    def test_имя_нужного_файла_идёт_последним(self, workspace, monkeypatch):
        """Модель пишет то, что прочитала последним.

        Живая проба показала: соседний файл, поставленный в конец
        запроса, она возвращает вместо запрошенного.
        """
        model = fake_model([file_answer("x = 1\n"), file_answer("y = 2\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("create", "lib.py", "раз"), Step("create", "test_lib.py", "два")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        second = model.seen[1][-1]["content"]
        # Порядок: справка о соседях — что нужно сейчас — ЗАДАЧА ЦЕЛИКОМ.
        # Последнее, что модель прочитала, — это то, что она напишет.
        assert second.index("СПРАВКА") < second.index("СЕЙЧАС НУЖЕН ОДИН ФАЙЛ")
        assert second.index("СЕЙЧАС НУЖЕН ОДИН ФАЙЛ") < second.index("ЗАДАЧА ЦЕЛИКОМ")
        assert "test_lib.py целиком" in second

    def test_соседние_файлы_едут_в_контекст(self, workspace, monkeypatch):
        """Тест, который агент пишет вторым, должен знать имена из первого."""
        model = fake_model([file_answer("def add(a, b):\n    return a + b\n"), file_answer("x = 1\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("create", "lib.py", "функция add"),
                                   Step("create", "test_lib.py", "тест")]))
        pipeline.node_step(state)
        pipeline.node_step(state)
        second = model.seen[1][-1]["content"]
        assert "СПРАВКА, файлы проекта" in second
        assert "def add(a, b):" in second

    def test_проект_виден_даже_когда_прогон_ещё_ничего_не_писал(self, workspace, monkeypatch):
        """Живой прогон: «напиши батник для запуска» дал `python test_run.py`.

        Каталог был не пуст — в нём лежал готовый quadratic.py, — но
        прогон начинался с нуля, справка отдавала пустоту, и модель
        взяла имя из ниоткуда. Спрашивать «что запускать» было не у кого:
        в задаче не было ни одного имени файла.
        """
        (workspace / "quadratic.py").write_text(
            "def solve(a):" + chr(10) + "    return a" + chr(10), encoding="utf-8")
        model = fake_model([file_answer("@echo off" + chr(10) + "python quadratic.py" + chr(10))])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("напиши батник", [Step("create", "run.bat", "батник")]))
        pipeline.node_step(state)

        asked = model.seen[0][-1]["content"]
        assert "quadratic.py" in asked, "имя файла надо откуда-то взять"
        assert "Главный файл программы: quadratic.py" in asked


class TestSkeleton:
    """Скелет собирает КОД, модели на этом шаге нет.

    Приём из наблюдения: базовые Coder-модели обучены на ДОПОЛНЕНИЕ
    кода, а не на исполнение инструкций. «Напиши файл по описанию» —
    инструкция, «допиши тела в готовый скелет» — дополнение. Между
    двумя запросами к модели стоит код: он собирает скелет из короткого
    структурированного ответа, и форма файла оказывается не на совести
    модели.
    """

    SPEC = {
        "imports": ["math", "math"],
        "entry": "main",
        "functions": [
            {"name": "parse", "args": "text", "purpose": "разобрать выражение"},
            {"name": "main", "args": "", "purpose": "цикл ввода-вывода"},
        ],
    }

    def test_скелет_разбирается_как_python(self):
        assert syntax_ok("a.py", pipeline.render_skeleton(self.SPEC, "задача")) == (True, "")

    def test_определения_на_месте(self):
        text = pipeline.render_skeleton(self.SPEC, "задача")
        assert definitions("a.py", text) == {"parse", "main"}

    def test_описания_попадают_в_кавычки(self):
        assert '"""разобрать выражение"""' in pipeline.render_skeleton(self.SPEC, "задача")

    def test_задача_становится_описанием_файла(self):
        assert pipeline.render_skeleton(self.SPEC, "калькулятор").startswith('"""калькулятор"""')

    def test_повторы_импортов_схлопываются(self):
        text = pipeline.render_skeleton(self.SPEC, "з")
        assert text.count("import math") == 1

    def test_точка_входа_из_ответа_модели(self):
        assert 'if __name__ == "__main__":\n    main()' in pipeline.render_skeleton(self.SPEC, "з")

    def test_точка_входа_по_умолчанию_это_main(self):
        """Поле entry модель заполняет через раз, и файл оставался без запуска."""
        spec = {"functions": [{"name": "main", "args": "", "purpose": "цикл"}]}
        assert "__main__" in pipeline.render_skeleton(spec, "з")

    def test_без_main_точки_входа_нет(self):
        spec = {"functions": [{"name": "add", "args": "a, b", "purpose": "сумма"}]}
        assert "__main__" not in pipeline.render_skeleton(spec, "з")

    def test_негодное_имя_функции_пропускается(self):
        """Модель иногда отвечает «функция сложения» вместо имени."""
        spec = {"functions": [{"name": "функция сложения", "args": "", "purpose": "x"},
                              {"name": "add", "args": "a", "purpose": "y"}]}
        assert definitions("a.py", pipeline.render_skeleton(spec, "з")) == {"add"}

    def test_кавычки_в_описании_не_ломают_файл(self):
        spec = {"functions": [{"name": "f", "args": "", "purpose": 'делает """нечто"""'}]}
        assert syntax_ok("a.py", pipeline.render_skeleton(spec, "з")) == (True, "")

    def test_негодное_имя_модуля_пропускается(self):
        """`import assert` не разбирается, а именно это модель и предложила.

        Замер режимов письма упирался в это на задаче про самое длинное
        слово: одно слово в списке `imports` — и весь прогон кончался
        ничем. Модель отвечает про смысл, синтаксис — наша забота.
        """
        spec = {"imports": ["assert", "os.path", "два слова", "", "json"],
                "functions": [{"name": "f", "args": "", "purpose": "x"}]}
        text = pipeline.render_skeleton(spec, "з")
        assert syntax_ok("a.py", text) == (True, "")
        assert "import os.path" in text
        assert "import json" in text
        assert "assert" not in text

    @pytest.mark.parametrize("name,ok", [
        ("os", True), ("os.path", True), ("assert", False), ("import", False),
        ("два слова", False), ("", False), ("3json", False),
    ])
    def test_годность_имени_модуля(self, name, ok):
        assert pipeline.importable(name) is ok

    def test_схема_состава_не_просит_кода(self):
        props = pipeline.file_plan_schema()["properties"]
        assert set(props["functions"]["items"]["required"]) == {"name", "args", "purpose"}
        assert "content" not in props and "code" not in props


class TestWriteModes:
    """Два способа написать файл: один запрос или скелет плюс дописывание."""

    def test_по_умолчанию_один_запрос(self):
        assert pipeline.WRITE_MODE == "direct"

    def test_прямой_режим_делает_один_запрос(self, workspace, monkeypatch):
        model = fake_model([file_answer("x = 1\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "WRITE_MODE", "direct")
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert len(model.seen) == 1
        assert (workspace / "app.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_режим_скелета_делает_два(self, workspace, monkeypatch):
        spec = json.dumps({"functions": [{"name": "add", "args": "a, b", "purpose": "сумма"}]})
        filled = file_answer('def add(a, b):\n    """сумма"""\n    return a + b\n')
        model = fake_model([spec, filled])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "WRITE_MODE", "skeleton")

        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)

        assert len(model.seen) == 2, "состав файла и дописывание — два запроса"
        assert "return a + b" in (workspace / "app.py").read_text(encoding="utf-8")
        assert "..." in state.extra["skeleton"], "в скелете тела пустые"

    def test_скелет_едет_в_запрос_на_дописывание(self, workspace, monkeypatch):
        spec = json.dumps({"functions": [{"name": "add", "args": "a, b", "purpose": "сумма"}]})
        model = fake_model([spec, file_answer("def add(a, b):\n    return a + b\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "WRITE_MODE", "skeleton")
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)

        second = model.seen[1][-1]["content"]
        assert "Скелет файла app.py" in second
        assert "def add(a, b):" in second

    def test_потерянная_при_дописывании_функция_это_провал(self, workspace, monkeypatch):
        """Скелет — договор о составе, и дописывание не имеет права его нарушить."""
        spec = json.dumps({"functions": [
            {"name": "add", "args": "a, b", "purpose": "сумма"},
            {"name": "mul", "args": "a, b", "purpose": "произведение"},
        ]})
        # Пара ответов дважды: шаг создания переспрашивает один раз,
        # и одной неудачи мало, чтобы его провалить.
        bad = file_answer("def add(a, b):\n    return a + b\n")
        model = fake_model([spec, bad, spec, bad])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "WRITE_MODE", "skeleton")

        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert "пропали функции: mul" in state.extra["log"][0]
        assert not (workspace / "app.py").exists()

    def test_оборванный_ответ_называют_оборванным(self, workspace, monkeypatch):
        """Разбор проверяется РАНЬШЕ состава, и это не мелочь отчёта.

        Файл, который не разбирается, не имеет определений вовсе —
        и сверка состава честно объявляет пропавшими все до одной
        функции. Замер режимов письма из-за этого показывал «пропали
        функции» там, где модель ушла в разгон: писала assert за
        assert, упиралась в предел длины и обрывалась на полуслове.
        Беда была в длине, а искали её в составе.
        """
        spec = json.dumps({"functions": [{"name": "add", "args": "a, b", "purpose": "сумма"}]})
        torn = file_answer("def add(a, b):\n    assert add(1, 1) == 2, 'сум")
        monkeypatch.setattr(pipeline, "request_model", fake_model([spec, torn, spec, torn]))
        monkeypatch.setattr(pipeline, "WRITE_MODE", "skeleton")

        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert "не разбирается" in state.extra["log"][0]
        assert "пропали функции" not in state.extra["log"][0]

    def test_разгон_дописывания_запрещён_правилами(self):
        """Живой замер: тестовая функция писала assert, пока хватало длины."""
        assert "две-три" in pipeline.FILL_RULES
        assert "не удлиняй" in pipeline.FILL_RULES

    def test_не_python_пишется_прямо_даже_в_режиме_скелета(self, workspace, monkeypatch):
        """Скелет из функций для батника бессмыслен."""
        model = fake_model([file_answer("@echo off\npython calc.py\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "WRITE_MODE", "skeleton")
        state = started(Plan("з", [Step("create", "run.bat", "")]))
        pipeline.node_step(state)
        assert len(model.seen) == 1
        assert (workspace / "run.bat").exists()


class TestStepOther:
    def test_чтение_кладёт_файл_с_номерами_строк(self, project):
        state = started(Plan("з", [Step("read", "calc_mod.py", "")]))
        pipeline.node_step(state)
        assert "1| def add(a, b):" in state.retrieved

    def test_шаг_проверки_с_существующим_файлом(self, project):
        state = started(Plan("з", [Step("test", "test_calc_mod.py", "")]))
        pipeline.node_step(state)
        assert state.extra["tests"] == "test_calc_mod.py"

    def test_шаг_проверки_с_выдуманным_файлом(self, project):
        """Модель регулярно велит прогнать тест, который сама не создала."""
        state = started(Plan("з", [Step("test", "test_нетакого.py", "")]))
        pipeline.node_step(state)
        assert "tests" not in state.extra
        assert "нет — проверим тем, что есть" in state.extra["log"][0]

    def test_шаг_команды_спрашивает(self, project):
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(a) or False)
        state = started(Plan("з", [Step("run", "python calc_mod.py", "")]))
        pipeline.node_step(state)
        assert "python calc_mod.py" in shown[0]
        assert "Отказано" in state.extra["log"][0]

    def test_шаг_команды_выполняется(self, workspace):
        (workspace / "hi.py").write_text("print('ага')\n", encoding="utf-8")
        state = started(Plan("з", [Step("run", "python hi.py", "")]))
        pipeline.node_step(state)
        assert "ага" in state.extra["log"][0]


# ====================================================================
# КОНВЕЙЕР: ЗАВИСИМОСТИ И ПРОВЕРКА
# ====================================================================

class TestScaffoldAsked:
    """Каркас, о котором просит ЧЕЛОВЕК: определения с описаниями и без кода.

    Появилось из живого прогона. На «напиши каркас приложения-переводчика»
    агент написал работающее приложение: второе правило CREATE_RULES
    («без заглушек, без TODO») запрещает ровно то, о чём просили.
    Домашнее правило переспорило задачу — а такого права у него нет.

    Отличать каркас-просьбу от режима письма `skeleton` важно: там
    скелет — промежуточная форма, которую человек не видит, здесь он
    и есть результат, и второго прохода не будет.
    """

    @pytest.mark.parametrize("task", [
        "напиши каркас приложения на python",
        "сделай скелет модуля корзины",
        "набросай заготовку парсера",
        "напиши функции без реализации",
        "сделай app.py, только сигнатуры функций",
    ])
    def test_просьбу_о_каркасе_видно(self, task):
        assert planner_module.wants_scaffold(task)

    @pytest.mark.parametrize("task", [
        "напиши приложение на python",
        "сделай калькулятор с тестами",
        "поправь add в calc.py",
    ])
    def test_обычную_задачу_за_каркас_не_принимают(self, task):
        assert not planner_module.wants_scaffold(task)

    def test_без_заглушек_это_не_просьба_о_каркасе(self):
        """Ровно противоположная просьба, и корень у неё совпадающий."""
        assert not planner_module.wants_scaffold("напиши рабочий код без заглушек")

    def test_каркас_собирается_кодом_за_один_запрос(self, workspace, monkeypatch):
        """У модели спрашивают СОСТАВ, форму файла строит render_skeleton.

        Свободной генерацией это не делается, и проверено живьём:
        правила «описание в тройных кавычках и многоточие» модель
        исполняет наполовину — тела оставляет пустыми, описания
        не пишет. Два захода подряд вернули `pass` без единой фразы.
        """
        spec = json.dumps({
            "functions": [
                {"name": "translate", "args": "word", "purpose": "переводит слово на английский"},
                {"name": "main", "args": "", "purpose": "спрашивает слово и печатает перевод"},
            ],
            "entry": "main",
        })
        model = fake_model([spec])
        monkeypatch.setattr(pipeline, "request_model", model)

        state = started(Plan("напиши каркас переводчика", [Step("create", "app.py", "")]))
        pipeline.node_step(state)

        assert len(model.seen) == 1, "каркас — это ПЕРВАЯ половина скелета, дописывания не будет"
        written = (workspace / "app.py").read_text(encoding="utf-8")
        assert filled_bodies("app.py", written) == [], "в каркасе не должно быть кода"
        assert without_docstring("app.py", written) == [], "каркас без описаний бесполезен"
        assert "переводит слово на английский" in written
        assert 'if __name__ == "__main__":' in written

    def test_обычная_задача_каркасом_не_пишется(self, workspace, monkeypatch):
        model = fake_model([file_answer("def add(a, b):\n    return a + b\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("напиши сложение", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert (workspace / "app.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    def test_каркас_без_определений_это_беда(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([json.dumps({"functions": []})] * 2))
        state = started(Plan("напиши каркас чего-нибудь", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert state.extra["failed_steps"], "пустой каркас — не работа"
        assert not (workspace / "app.py").exists()

    def test_зависимости_по_каркасу_не_ставятся(self, workspace, monkeypatch):
        """Импорты каркаса — догадка о коде, которого ещё нет.

        Живой прогон: в каркасе переводчика оказалось `import translate`,
        и агент пошёл искать такой пакет. Менять окружение человека
        ради догадки нельзя.
        """
        (workspace / "app.py").write_text("import такогонет\n", encoding="utf-8")
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        state = started(Plan("напиши каркас приложения", []), touched=["app.py"])
        pipeline.node_deps(state)
        assert called == []
        assert state.extra["missing_packages"] == []

    def test_каркас_проверяется_разбором(self, workspace, monkeypatch):
        """Ни запуском, ни импортом — и то и другое здесь врёт.

        Запуск каркаса отработает с кодом 0 всегда: в теле многоточие.
        Импорт упадёт на пакете, которого ещё нет. Правду про каркас
        говорит только разбор.
        """
        (workspace / "app.py").write_text(
            'def main():\n    """делает дело"""\n    ...\n', encoding="utf-8"
        )
        seen = []

        def run(command, timeout=None, feed=""):
            seen.append(command)
            return Run(command, 0, "", "", 0.1)

        monkeypatch.setattr(pipeline, "execute", run)
        state = started(Plan("напиши каркас приложения", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"] is True
        assert "разбор" in state.extra["verified_by"]
        assert any("py_compile" in " ".join(command) for command in seen)


class TestStrayDefinitions:
    """Определение, уехавшее внутрь чужого блока, — работа без работы.

    Живой прогон: на «добавь вывод производной» модель ответила формой
    `append`, а тело начала с четырёх пробелов — и `def` приехал внутрь
    блока `except`, стоявшего в конце файла. Файл разбирается, проверка
    импортом проходит, агент рапортует «Готово». А функция объявляется
    только при ошибке ввода и не вызывается никогда.
    """

    BEFORE = (
        "def solve(a):\n"
        "    return a\n"
        "\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        print(solve(1))\n"
        "    except ValueError:\n"
        "        print('ошибка')\n"
    )

    def test_определение_внутри_except_видно(self):
        after = self.BEFORE + "\n    def derivative(a, b):\n        return 2*a\n"
        assert stray_definitions("a.py", self.BEFORE, after) == ["derivative"]

    def test_на_своём_уровне_претензий_нет(self):
        after = self.BEFORE + "\n\ndef derivative(a, b):\n    return 2*a\n"
        assert stray_definitions("a.py", self.BEFORE, after) == []

    def test_замыкание_это_законно(self):
        after = self.BEFORE + "\n\ndef outer():\n    def inner():\n        return 1\n    return inner\n"
        assert stray_definitions("a.py", self.BEFORE, after) == []

    def test_метод_класса_это_законно(self):
        after = self.BEFORE + "\n\nclass Solver:\n    def run(self):\n        return 1\n"
        assert stray_definitions("a.py", self.BEFORE, after) == []

    def test_чужое_прошлое_не_наша_вина(self):
        """Такое могло лежать в файле и до нас: запрещать его — запрещать правку."""
        before = "if True:\n    def f():\n        return 1\n"
        assert stray_definitions("a.py", before, before + "\nx = 1\n") == []

    def test_новый_файл_не_проверяется(self):
        """У нового файла всё новое, и `try/except ImportError` там законен."""
        after = "try:\n    import json\nexcept ImportError:\n    def json():\n        return None\n"
        assert stray_definitions("a.py", "", after) == []

    def test_дописывание_снимает_сбитый_отступ(self, workspace):
        """Механическую беду чинят механически, а не третьей просьбой.

        Три захода подряд давали один и тот же отступ. Снять его кодом
        и сказать об этом — честнее и дешевле, чем спорить промптом.
        Приём тот же, что у `normalize_edit`.
        """
        (workspace / "app.py").write_text(self.BEFORE, encoding="utf-8")
        answer = append_to_file("app.py", "    def derivative(a, b):" + chr(10) + "        return 2*a" + chr(10))
        assert "снят лишний отступ" in answer
        written = (workspace / "app.py").read_text(encoding="utf-8")
        assert stray_definitions("app.py", self.BEFORE, written) == []
        assert "def derivative" in written

    def test_перезапись_целиком_со_сдвигом_чинится(self, workspace):
        """Живой прогон: «исправь все ошибки» вернуло файл, сдвинутый вправо.

        Правка отменялась с «строка 1: unexpected indent» — а отменять
        то, что чинится одним dedent, значит терять работу
        на форматировании.
        """
        before = "def f():" + chr(10) + "    return 1" + chr(10)
        (workspace / "app.py").write_text(before, encoding="utf-8")
        answer = write_file("app.py", "    def f():" + chr(10) + "        return 2" + chr(10))
        assert "снят лишний отступ" in answer
        assert (workspace / "app.py").read_text(encoding="utf-8") == "def f():" + chr(10) + "    return 2" + chr(10)

    def test_законный_отступ_не_трогают(self, workspace):
        """Дописать строку ВНУТРЬ блока `__main__` — обычное дело."""
        (workspace / "app.py").write_text(self.BEFORE, encoding="utf-8")
        append_to_file("app.py", "        print('готово')" + chr(10))
        assert "        print('готово')" in (workspace / "app.py").read_text(encoding="utf-8")

    def test_вложенное_определение_через_номера_строк_отменяется(self, workspace):
        """Что нельзя починить снятием отступа — отменяется с объяснением."""
        before = "if True:" + chr(10) + "    pass" + chr(10)
        (workspace / "app.py").write_text(before, encoding="utf-8")
        answer = replace_lines("app.py", "2", "2", "    def f():" + chr(10) + "        return 1")
        assert "ВНУТРИ чужого блока" in answer
        assert (workspace / "app.py").read_text(encoding="utf-8") == before


class TestImportsOnTop:
    """Импорт, дописанный в конец файла, бесполезен — его место наверху.

    Живой прогон показал это дословно. Человек принёс traceback «name
    'math' is not defined», модель ответила `import math` — и он приехал
    ПОСЛЕ блока `if __name__`, то есть выполнился уже после вызова,
    который падал. Файл разбирается, ошибка на месте, а человеку
    пришлось чинить руками.
    """

    APP = (
        "def solve(a):\n"
        "    return math.sqrt(a)\n"
        "\n\n"
        "if __name__ == '__main__':\n"
        "    print(solve(4))\n"
    )

    def test_импорт_уходит_в_начало(self, workspace):
        (workspace / "app.py").write_text(self.APP, encoding="utf-8")
        answer = append_to_file("app.py", "import math\n")
        assert "в начало файла" in answer
        written = (workspace / "app.py").read_text(encoding="utf-8")
        assert written.startswith("import math")

    def test_докстринг_модуля_остаётся_первым(self, workspace):
        (workspace / "app.py").write_text('"""модуль"""\nimport os\n\n\nx = os\n', encoding="utf-8")
        append_to_file("app.py", "import math\n")
        written = (workspace / "app.py").read_text(encoding="utf-8")
        assert written.startswith('"""модуль"""')
        assert written.splitlines()[1:3] == ["import os", "import math"]

    def test_не_импорт_дописывается_как_прежде(self, workspace):
        (workspace / "app.py").write_text(self.APP, encoding="utf-8")
        answer = append_to_file("app.py", "print('конец')\n")
        assert "в конец файла" in answer
        assert (workspace / "app.py").read_text(encoding="utf-8").rstrip().endswith("print('конец')")


class TestUnreachable:
    """Код после return не выполнится никогда, как бы он ни выглядел.

    Живой прогон: на «добавь, чтобы выводилась ещё производная» модель
    дописала три строки в конец функции — после `return`. Файл
    разбирается, запускается, все проверки зелёные, агент отчитывается
    «Готово». А производная не печатается и не напечатается никогда.
    """

    BEFORE = ("def solve(a):" + chr(10) + "    if a > 0:" + chr(10) +
              "        return 1" + chr(10) + "    return None" + chr(10))

    def test_код_после_return_виден(self):
        after = self.BEFORE + "    print(2)" + chr(10)
        assert unreachable_code("a.py", self.BEFORE, after) == ["print(2)"]

    def test_код_перед_return_законен(self):
        after = "def solve(a):" + chr(10) + "    print(2)" + chr(10) + "    return 1" + chr(10)
        assert unreachable_code("a.py", self.BEFORE, after) == []

    def test_чужой_мёртвый_код_не_наша_вина(self):
        """Он мог лежать в файле и до нас: запрещать из-за него правку нельзя."""
        old = "def f():" + chr(10) + "    return 1" + chr(10) + "    print(0)" + chr(10)
        assert unreachable_code("a.py", old, old + chr(10) + "x = 1" + chr(10)) == []

    def test_сдвиг_строк_ложной_тревоги_не_даёт(self):
        """Сравниваются тексты строк, а не номера: номера двигает любая вставка."""
        old = "def f():" + chr(10) + "    return 1" + chr(10) + "    print(0)" + chr(10)
        new = "import os" + chr(10) * 2 + old
        assert unreachable_code("a.py", old, new) == []

    def test_правка_с_мёртвым_кодом_отменяется(self, workspace):
        (workspace / "app.py").write_text(self.BEFORE, encoding="utf-8")
        answer = append_to_file("app.py", "    print(2)" + chr(10))
        assert "не дойдёт никогда" in answer
        assert (workspace / "app.py").read_text(encoding="utf-8") == self.BEFORE

    def test_не_python_не_проверяется(self):
        assert unreachable_code("run.bat", "echo 1", "echo 2") == []


class TestDoubledMain:
    """Блок входа в программу бывает ровно один.

    Живой прогон: модель, которой показали одну функцию, вернула её
    вместе со всем блоком `__main__`, а замена по строкам положила это
    рядом со старым. Программа стала спрашивать коэффициенты дважды,
    и все проверки при этом зелёные: файл разбирается, запускается,
    подставленного ввода хватает на оба круга.
    """

    ONE = ("def f():" + chr(10) + "    return 1" + chr(10) * 2 +
           'if __name__ == "__main__":' + chr(10) + "    f()" + chr(10))

    def test_второй_блок_виден(self):
        assert doubled_main("a.py", self.ONE, self.ONE + chr(10) + self.ONE)

    def test_один_блок_это_норма(self):
        assert not doubled_main("a.py", self.ONE, self.ONE + "x = 1" + chr(10))

    def test_чужие_два_блока_не_наша_вина(self):
        """Так мог выглядеть файл и до нас."""
        both = self.ONE + self.ONE
        assert not doubled_main("a.py", both, both + "x = 1" + chr(10))

    def test_правка_со_вторым_блоком_отменяется(self, workspace):
        (workspace / "app.py").write_text(self.ONE, encoding="utf-8")
        answer = append_to_file("app.py", self.ONE)
        assert "второй блок" in answer
        assert (workspace / "app.py").read_text(encoding="utf-8") == self.ONE


class TestSameTree:
    """Комментарий не меняет поведения программы — значит, не чинит её."""

    def test_комментарий_дерево_не_меняет(self):
        before = "x = 1\n"
        assert same_tree("a.py", before, before + "# пояснение\n")

    def test_настоящая_правка_меняет(self):
        assert not same_tree("a.py", "x = 1\n", "x = 2\n")

    def test_не_python_сравнивать_нечем(self):
        assert not same_tree("run.bat", "echo 1\n", "echo 1\nrem пояснение\n")

    def test_сломанный_файл_считается_другим(self):
        assert not same_tree("a.py", "x = 1\n", "x = (\n")


class TestErrorWithoutText:
    """«Исправь ошибку» без текста ошибки: агент идёт и смотрит сам.

    Раньше он правил наугад — живой прогон дал `# Дополнительные
    комментарии`, дописанные в конец файла. Модель и не могла ответить
    иначе: текста ошибки в запросе не было. А узнать его можно —
    программа лежит на диске, запускать её мы умеем.
    """

    @pytest.mark.parametrize("task", [
        "исправь ошибку в нашем приложении",
        "приложение падает",
        "не работает вывод корней",
    ])
    def test_просят_починить_не_говоря_что(self, task):
        assert pipeline.asks_about_error(task)

    @pytest.mark.parametrize("task", [
        "исправь ошибку: Traceback (most recent call last): File app.py, line 4",
        "добавь функцию деления",
        "напиши приложение hello world",
    ])
    def test_текст_ошибки_принесли_или_её_нет(self, task):
        assert not pipeline.asks_about_error(task)

    def test_ошибка_запуска_попадает_в_запрос(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(math.pi)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", lambda c, timeout=None, feed="": Run(
            str(c), 1, "", "NameError: name 'math' is not defined", 0.1))
        model = fake_model([anchor_answer("app.py", "print(math.pi)", "import math\nprint(math.pi)")])
        monkeypatch.setattr(pipeline, "request_model", model)

        state = started(Plan("исправь ошибку в приложении",
                             [Step("edit", "app.py", "")]))
        pipeline.node_step(state)

        asked = model.seen[0][-1]["content"]
        assert "NameError" in asked, "модель должна увидеть настоящую ошибку"

    def test_рабочая_программа_ошибки_не_придумывает(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = started(Plan("исправь ошибку в приложении", []))
        assert pipeline._seen_error(state, "app.py") == ""

    def test_комментарий_вместо_починки_не_засчитывается(self, workspace, monkeypatch):
        """Текст файла изменился, дерево — нет, ошибка на месте."""
        (workspace / "app.py").write_text("print(math.pi)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", lambda c, timeout=None, feed="": Run(
            str(c), 1, "", "NameError: name 'math' is not defined", 0.1))
        model = fake_model([
            anchor_answer("app.py", "", "# просто комментарий"),
            anchor_answer("app.py", "print(math.pi)", "import math\nprint(math.pi)"),
        ])
        monkeypatch.setattr(pipeline, "request_model", model)

        state = started(Plan("исправь ошибку в приложении", [Step("edit", "app.py", "")]))
        pipeline.node_step(state)

        assert len(model.seen) == 2, "первая правка не засчитана — спросили ещё раз"
        assert "только комментарий" in model.seen[1][-1]["content"]
        assert "import math" in (workspace / "app.py").read_text(encoding="utf-8")


class TestSameCode:
    """Отличать правку от её видимости — и не спутать с настоящей правкой отступа."""

    def test_пустые_строки_разницей_не_считаются(self):
        assert same_code("a = 1\n", "a = 1\n\n\n")

    def test_хвостовые_пробелы_тоже(self):
        assert same_code("a = 1\n", "a = 1   \n")

    def test_изменённый_код_виден(self):
        assert not same_code("a = 1\n", "a = 2\n")

    def test_сбитый_отступ_это_настоящая_правка(self):
        """Слева пробелы не трогаем: в Python это ошибка, а не оформление."""
        assert not same_code("def f():\n    return 1\n", "def f():\n        return 1\n")

    def test_дописанная_строка_видна(self):
        assert not same_code("a = 1\n", "a = 1\nb = 2\n")


class TestFilledBodies:
    """Разбор кода вместо поиска по словам: `...` бывает и внутри описания."""

    def test_пустые_тела(self):
        text = 'def a():\n    """что делает"""\n    ...\n\n\ndef b():\n    pass\n'
        assert filled_bodies("app.py", text) == []

    def test_настоящий_код_виден(self):
        text = 'def a():\n    """что делает"""\n    return 1\n'
        assert filled_bodies("app.py", text) == ["a"]

    def test_многоточие_в_описании_телом_не_считается(self):
        text = 'def a():\n    """шаг первый, ... , шаг третий"""\n    ...\n'
        assert filled_bodies("app.py", text) == []

    def test_описание_обязательно(self):
        assert without_docstring("app.py", "def a():\n    ...\n") == ["a"]
        assert without_docstring("app.py", 'def a():\n    """есть"""\n    ...\n') == []

    def test_сломанный_файл_не_роняет_разбор(self):
        assert filled_bodies("app.py", "def a(:\n") == []
        assert without_docstring("app.py", "def a(:\n") == []

    def test_не_питон_не_разбирают(self):
        assert filled_bodies("run.bat", "echo hi\n") == []


class TestDepsNode:
    def test_ничего_не_ставится_если_всё_на_месте(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("import os\n", encoding="utf-8")
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_deps(state)
        assert state.extra["missing_packages"] == []
        assert called == []

    def test_недостающий_пакет_ставится(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("import такогонет\n", encoding="utf-8")
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "поставлено")
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_deps(state)
        assert state.extra["missing_packages"] == ["такогонет"]
        assert called == ["такогонет"]

    def test_имя_пакета_а_не_имя_модуля(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline.env, "missing_imports", lambda text: ["yaml", "bs4"])
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        (workspace / "app.py").write_text("import yaml\n", encoding="utf-8")
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_deps(state)
        assert called == ["pyyaml beautifulsoup4"]

    def test_повторы_не_ставятся_дважды(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline.env, "missing_imports", lambda text: ["requests"])
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        for name in ("a.py", "b.py"):
            (workspace / name).write_text("import requests\n", encoding="utf-8")
        state = started(Plan("з", []), touched=["a.py", "b.py"])
        pipeline.node_deps(state)
        assert called == ["requests"]


class TestWaitsForInput:
    """Ждёт ли программа ввода — и, значит, надо ли подставлять строки.

    Раньше здесь была развилка потоньше: ввод под `if __name__`
    проверялся импортом, ввод на верхнем уровне — разбором, потому что
    импорт такой модуль ВЫПОЛНЯЕТ и падает тем же EOFError, от которого
    импорт и должен был спасти. Запуск с подставленным вводом сделал
    различие ненужным: он одинаково выполняет оба вида и одинаково
    ловит обе беды. Развилка убрана, вопрос остался один.
    """

    GUARDED = (
        "def solve(a):" + chr(10) +
        '    """корни"""' + chr(10) +
        "    return a" + chr(10) * 3 +
        'if __name__ == "__main__":' + chr(10) +
        "    a = float(input('a: '))" + chr(10) +
        "    print(solve(a))" + chr(10)
    )
    BARE = (
        "def solve(a):" + chr(10) +
        '    """корни"""' + chr(10) +
        "    return a" + chr(10) * 3 +
        "a = float(input('a: '))" + chr(10) +
        "print(solve(a))" + chr(10)
    )

    @pytest.mark.parametrize("source", ["GUARDED", "BARE"])
    def test_ввод_виден_в_обоих_видах(self, workspace, source):
        (workspace / "app.py").write_text(getattr(self, source), encoding="utf-8")
        assert pipeline._waits_for_input("app.py")

    def test_ввод_внутри_функции_тоже_считается(self, workspace):
        (workspace / "app.py").write_text(
            "def ask():" + chr(10) + "    return input('a: ')" + chr(10), encoding="utf-8")
        assert pipeline._waits_for_input("app.py")

    def test_имя_переменной_вводом_не_считается(self, workspace):
        """`user_input` — это не вызов `input`, а слово, похожее на него."""
        (workspace / "app.py").write_text("user_input = 1" + chr(10), encoding="utf-8")
        assert not pipeline._waits_for_input("app.py")

    def test_сломанный_файл_вводом_не_считается(self, workspace):
        (workspace / "app.py").write_text("def f(:" + chr(10), encoding="utf-8")
        assert not pipeline._waits_for_input("app.py")

    def test_ввод_подставляется_и_программа_запускается(self, workspace, monkeypatch):
        """Запуск с подставленным вводом заменил и импорт, и разбор.

        Оба молчали про то, ради чего программу пишут: файл разбирается,
        импортируется — а запуск падает NameError на первой же строке.
        Ввод подставляется строками, и программа выполняется всерьёз.
        """
        (workspace / "app.py").write_text(self.BARE, encoding="utf-8")
        seen = []

        def run(command, timeout=None, feed=""):
            seen.append((command, feed))
            return Run(str(command), 0, "", "", 0.1)

        monkeypatch.setattr(pipeline, "execute", run)
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"]
        assert "с подставленным вводом" in state.extra["verified_by"]
        assert seen[0][1] == pipeline.SCRIPTED_INPUT

    def test_ввод_под_main_тоже_подставляется(self, workspace, monkeypatch):
        """Разница «на верхнем уровне или под __main__» для запуска не важна."""
        (workspace / "app.py").write_text(self.GUARDED, encoding="utf-8")
        seen = []

        def run(command, timeout=None, feed=""):
            seen.append((command, feed))
            return Run(str(command), 0, "", "", 0.1)

        monkeypatch.setattr(pipeline, "execute", run)
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert "с подставленным вводом" in state.extra["verified_by"]
        assert seen[0][1] == pipeline.SCRIPTED_INPUT


class TestPlannedModules:
    """Свой модуль не ищется в сети, даже если файл не написался."""

    def test_несозданный_модуль_плана_остаётся_своим(self, workspace, monkeypatch):
        """Живой прогон: create calculator.py провалился, и агент пошёл
        ставить чужой пакет `calculator` с PyPI. Человек согласился.
        """
        (workspace / "test_calculator.py").write_text("import calculator\n", encoding="utf-8")
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        plan = Plan("з", [Step("create", "calculator.py", ""),
                          Step("create", "test_calculator.py", "")])
        state = started(plan, touched=["test_calculator.py"])
        pipeline.node_deps(state)

        assert called == [], "модуль, который план собирался создать, из сети не ставят"
        assert state.extra["missing_packages"] == []

    def test_чужой_пакет_по_прежнему_предлагается(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("import такогонет\n", encoding="utf-8")
        called = []
        monkeypatch.setattr(pipeline.env, "install", lambda p: called.append(p) or "ok")
        state = started(Plan("з", [Step("create", "app.py", "")]), touched=["app.py"])
        pipeline.node_deps(state)
        assert called == ["такогонет"]


class TestUndefinedNames:
    """Программу, которую нельзя запустить, проверяет линтер вместо запуска.

    Живой прогон: правка вернула `return root1, root2, derivative`, где
    `derivative` не существует. Файл разбирается, импортируется, агент
    рапортует «Готово» — а запуск падает NameError на первой же строке.
    Импорт и разбор молчат ровно про то, ради чего программу пишут.
    """

    BROKEN = (
        "def solve(a, b, c):\n"
        "    return a, derivative\n"
        "\n\n"
        "def main():\n"
        "    a = float(input('a: '))\n"
        "    print(solve(a, 1, 1))\n"
        "\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    def test_имя_из_ниоткуда_ловится(self, workspace):
        (workspace / "app.py").write_text(self.BROKEN, encoding="utf-8")
        assert undefined_names("app.py") == ["derivative"]

    def test_целый_файл_претензий_не_вызывает(self, workspace):
        (workspace / "app.py").write_text(self.BROKEN.replace(", derivative", ""), encoding="utf-8")
        assert undefined_names("app.py") == []

    def test_не_python_не_проверяется(self, workspace):
        (workspace / "run.bat").write_text("@echo off\npython app.py\n", encoding="utf-8")
        assert undefined_names("run.bat") == []

    def test_проверка_ловит_ненайденное_имя_в_конвейере(self, workspace):
        (workspace / "app.py").write_text(self.BROKEN, encoding="utf-8")
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"] is False
        assert "derivative" in state.extra["failure"]

    def test_слово_запускать_проверку_не_отменяет(self, workspace):
        """Проверка стояла на `"запуск" not in verified_by` — и молча ломалась.

        Строка «программа ждёт ввода, ЗАПУСКАТЬ её нечем» содержит
        подстроку «запуск», а смысл у неё обратный: программу как раз
        НЕ запускали. Признаком служит флаг, а не поиск слова в тексте,
        написанном для человека.
        """
        (workspace / "app.py").write_text(self.BROKEN, encoding="utf-8")
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)
        # Программа запускается по-настоящему и падает по-настоящему:
        # NameError вместо нашей формулировки про неопределённые имена.
        assert state.extra["tests_green"] is False
        assert "derivative" in state.extra["failure"]

    def test_линтер_спрашивается_и_у_запущенной(self, workspace, monkeypatch):
        """Запуск проходит ОДНОЙ веткой, а имя не определено во всём файле.

        Живой прогон: `math.sqrt` без `import math`, подставленные
        числа дали отрицательный дискриминант, ветка с корнями
        не выполнилась — запуск зелёный, программа падает на первом же
        уравнении с корнями. Линтер видел это с самого начала, а его
        не спрашивали: «запуск сильнее линтера» оказалось неверным.
        """
        (workspace / "app.py").write_text("print(1)" + chr(10), encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        monkeypatch.setattr(pipeline, "undefined_names", lambda p: ["math"])
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"] is False
        assert "math" in state.extra["failure"]

    def test_запуск_идёт_несколькими_наборами_ввода(self, workspace, monkeypatch):
        """Один набор проходит одной веткой, беда в соседней остаётся невидимой."""
        (workspace / "app.py").write_text("print(input())" + chr(10), encoding="utf-8")
        feeds = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": feeds.append(feed) or Run(str(c), 0, "", "", 0.1))
        monkeypatch.setattr(pipeline, "undefined_names", lambda p: [])
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert feeds == list(pipeline.SCRIPTED_INPUTS)

    def test_первый_же_провал_прекращает_запуски(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(input())" + chr(10), encoding="utf-8")
        feeds = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": feeds.append(feed) or Run(str(c), 1, "", "ZeroDivisionError", 0.1))
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert len(feeds) == 1, "сломано — дальше проверять нечего"
        assert state.extra["tests_green"] is False


class TestVerifyModes:
    def test_есть_тесты_значит_pytest(self, project, monkeypatch):
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = started(Plan("з", []), touched=["calc_mod.py"])
        pipeline.node_verify(state)
        assert state.extra["verified_by"] == "pytest"
        assert state.extra["tests_green"]

    def test_тестов_нет_значит_запуск_файла(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        seen = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": seen.append(c) or Run(c, 0, "1", "", 0.1))
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)
        assert state.extra["verified_by"] == "запуск app.py"
        assert seen[0] == [interpreter(), "app.py"], "путь отдельным аргументом: в нём бывает пробел"
        assert state.extra["tests_green"]

    def test_проверять_нечем(self, workspace):
        """«Нечем проверить» — отдельный исход, а не провал."""
        state = started(Plan("з", []), touched=[])
        pipeline.node_verify(state)
        assert state.extra["verified_by"] == "нечем"
        assert state.extra["unverifiable"]
        assert not state.extra["tests_green"]
        assert state.extra["failure"] == "", "чинить нечего — ошибки нет"

    def test_нечем_проверить_ведёт_к_итогу_а_не_к_починке(self, workspace):
        """Чинить нечего, если не сломано.

        Живой прогон: агент написал батник — правильно написал, — а `.bat`
        этой главе проверять нечем. Прогон считался провалившимся, цикл
        починки два круга бился о пустоту, и работа откатывалась.
        """
        state = started(Plan("з", []), touched=[])
        pipeline.node_verify(state)
        assert pipeline.edge_after_verify(state) == pipeline.DONE

    def test_красная_проверка_по_прежнему_ведёт_к_починке(self, project, monkeypatch):
        monkeypatch.setattr(pipeline, "execute", fake_run(green=False))
        state = started(Plan("з", []), touched=["calc_mod.py"])
        pipeline.node_verify(state)
        assert pipeline.edge_after_verify(state) == pipeline.READ_NODE

    def test_запускается_не_тестовый_файл(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        (workspace / "helper.py").write_text("print(2)\n", encoding="utf-8")
        seen = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": seen.append(c) or Run(c, 0, "", "", 0.1))
        state = started(Plan("з", []), touched=["app.py", "helper.py"])
        pipeline.node_verify(state)
        assert "helper.py" in seen[0], "запускается последний написанный"


class TestInteractiveProgram:
    """Программу, которая ждёт ввода, запуском не проверить.

    Нашёл живой прогон, и цена ошибки была высокой: агент написал
    работающий калькулятор, проверка запустила его без stdin, тот честно
    упал с `EOFError`, и конвейер откатил готовую работу как сломанную.
    У проверки нет пользователя, который нажмёт клавиши.
    """

    def test_вызов_input_виден(self, workspace):
        (workspace / "a.py").write_text("x = input()\n", encoding="utf-8")
        assert pipeline._waits_for_input("a.py")

    def test_чтение_stdin_видно(self, workspace):
        (workspace / "a.py").write_text("import sys\nprint(sys.stdin.read())\n", encoding="utf-8")
        assert pipeline._waits_for_input("a.py")

    def test_слово_input_в_имени_переменной_не_считается(self, workspace):
        """Разбор, а не поиск подстроки: `user_input` — не вызов."""
        (workspace / "a.py").write_text("user_input = 1  # input() тут в комментарии\n", encoding="utf-8")
        assert not pipeline._waits_for_input("a.py")

    def test_сломанный_файл_не_роняет_проверку(self, workspace):
        (workspace / "a.py").write_text("def f(:\n", encoding="utf-8")
        assert not pipeline._waits_for_input("a.py")

    def test_интерактивная_программа_запускается_с_вводом(self, workspace, monkeypatch):
        """Раньше её импортировали — теперь запускают, подставив строки.

        Импорт отвечал на вопрос «файл разбирается и подключается»,
        а человеку нужен ответ на другой: «работает ли программа».
        """
        (workspace / "app.py").write_text(
            "def main():" + chr(10) + "    print(input())" + chr(10) * 2 +
            "if __name__ == '__main__':" + chr(10) + "    main()" + chr(10),
            encoding="utf-8",
        )
        seen = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": seen.append((c, feed)) or Run(str(c), 0, "", "", 0.1))
        state = started(Plan("з", [Step("create", "app.py", "")]), touched=["app.py"])
        pipeline.node_verify(state)

        assert "с подставленным вводом" in state.extra["verified_by"]
        assert seen[0][0][-1] == "app.py", "запускается файл, а не импортируется модуль"
        assert state.extra["tests_green"]

    def test_не_хватило_ввода_это_не_провал(self, workspace, monkeypatch):
        """Программа спросила больше, чем мы дали, — это наша беда, не её."""
        (workspace / "app.py").write_text(
            "print(input())" + chr(10) + "print(input())" + chr(10), encoding="utf-8")
        answers = iter([Run("запуск", 1, "", "EOFError: EOF when reading a line", 0.1),
                        Run("разбор", 0, "", "", 0.1)])
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": next(answers))
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"], "разбор прошёл — значит, не сломано"
        assert "не хватило" in state.extra["verified_by"]

    def test_обычная_программа_по_прежнему_запускается(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print('привет')\n", encoding="utf-8")
        seen = []
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": seen.append(c) or Run(str(c), 0, "", "", 0.1))
        state = started(Plan("з", [Step("create", "app.py", "")]), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["verified_by"] == "запуск app.py"
        assert seen[0] == [interpreter(), "app.py"], "путь отдельным аргументом: в нём бывает пробел"

    def test_дочернему_процессу_не_отдаётся_наш_терминал(self, workspace):
        """Иначе программа агента заберёт ввод себе, и агент повиснет."""
        (workspace / "a.py").write_text("print(input())\n", encoding="utf-8")
        run = execute("python a.py", timeout=30)
        assert not run.ok
        assert "EOF" in run.text(), "ввод должен быть закрыт, а не унаследован"


class TestFileToFix:
    def test_чинится_файл_из_текста_ошибки(self, workspace):
        state = started(Plan("з", []), touched=["lib.py", "test_lib.py"],
                        failure="test_lib.py:12: UnboundLocalError", verify_output="")
        assert pipeline._file_to_fix(state) == "test_lib.py"

    def test_без_подсказки_чинится_не_тест(self, workspace):
        state = started(Plan("з", []), touched=["lib.py", "test_lib.py"],
                        failure="что-то пошло не так", verify_output="")
        assert pipeline._file_to_fix(state) == "lib.py"

    def test_чужой_файл_не_чинится(self, workspace):
        """Тест, который агент не писал, — это провал задачи, а не приглашение его переписать."""
        state = started(Plan("з", []), touched=["lib.py"],
                        failure="test_чужой.py:3: AssertionError", verify_output="")
        assert pipeline._file_to_fix(state) == "lib.py"

    def test_нечего_чинить(self, workspace):
        state = started(Plan("з", []), touched=[])
        pipeline.node_read(state)
        assert "не тронул ни одного файла" in state.error
        assert pipeline.edge_after_read(state) == pipeline.DONE

    def test_чинимый_файл_читается_с_парой(self, workspace):
        (workspace / "lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (workspace / "test_lib.py").write_text("from lib import add\n", encoding="utf-8")
        state = started(Plan("з", []), touched=["lib.py", "test_lib.py"],
                        failure="test_lib.py:1: ошибка", verify_output="")
        pipeline.node_read(state)
        assert "Файл test_lib.py" in state.retrieved
        assert "Парный файл lib.py" in state.retrieved
        assert "return a - b" in state.retrieved


# ====================================================================
# КОНВЕЙЕР: ПРАВКА
# ====================================================================

class TestApplyEdit:
    def test_форма_строк(self, sample):
        out = pipeline.apply_edit({"form": "lines", "path": "sample.py", "start": 2, "end": 2,
                                   "content": "    return 0"})
        assert "Заменены строки" in out
        assert "return 0" in sample.read_text(encoding="utf-8")

    def test_форма_якоря(self, sample):
        out = pipeline.apply_edit({"form": "anchor", "path": "sample.py",
                                   "old": "return a + b", "new": "return 0"})
        assert "Заменено" in out

    def test_форма_целиком(self, sample):
        out = pipeline.apply_edit({"form": "full", "path": "sample.py",
                                   "content": "def add(a, b):\n    return 0\n\ndef mul(a, b):\n    return a * b\n"})
        assert "перезаписан" in out

    def test_нет_обязательных_полей(self, sample):
        out = pipeline.apply_edit({"form": "lines", "path": "sample.py"})
        assert "нет полей" in out and "start" in out

    def test_неизвестная_форма(self, sample):
        assert "нет полей" in pipeline.apply_edit({"form": "diff", "path": "sample.py"})

    def test_путь_берётся_запасной(self, sample):
        out = pipeline.apply_edit({"form": "anchor", "old": "return a + b", "new": "return 0"},
                                  default_path="sample.py")
        assert "Заменено" in out

    def test_без_пути_вообще(self, sample):
        assert "не назвала файл" in pipeline.apply_edit({"form": "anchor", "old": "x", "new": "y"})

    def test_правка_идёт_через_инструменты_и_попадает_в_журнал(self, sample):
        pipeline.apply_edit({"form": "anchor", "path": "sample.py", "old": "return a + b", "new": "return 0"})
        assert guard.changed_files() == [sample]

    def test_правка_за_пределы_каталога_не_проходит(self, sample):
        out = pipeline.apply_edit({"form": "full", "path": "../взлом.py", "content": "x = 1\n"})
        assert "пределы" in out


class TestNormalizeEdit:
    """Ответ модели приводится к той форме, которую она имела в виду.

    Правило из живого прогона: на задаче «добавь в calc.py
    интерактивность» модель раз за разом отвечала формой `anchor`
    с ПУСТЫМ якорем и непустой заменой. Пустой якорь не адресует ничего,
    и как замена ответ бессмыслен; смысл в нём один — «допиши вот это».

    Вывод шире одного случая: новая форма в промпте не делает модель
    её использующей. Правило про `append` стоит в EDIT_RULES ПЕРВЫМ
    пунктом, и модель всё равно выбирала anchor три раза из трёх.
    """

    def test_пустой_якорь_это_дописывание(self):
        fixed, hint = pipeline.normalize_edit(
            {"form": "anchor", "path": "a.py", "old": "", "new": "x = 1"}
        )
        assert fixed["form"] == "append"
        assert fixed["content"] == "x = 1"
        assert "пустой якорь" in hint

    def test_якорь_из_пробелов_тоже(self):
        fixed, _ = pipeline.normalize_edit(
            {"form": "anchor", "path": "a.py", "old": "   \n ", "new": "x = 1"}
        )
        assert fixed["form"] == "append"

    def test_нормальный_якорь_не_трогается(self):
        data = {"form": "anchor", "path": "a.py", "old": "y = 1", "new": "y = 2"}
        fixed, hint = pipeline.normalize_edit(data)
        assert fixed is data
        assert hint == ""

    def test_пустая_замена_дописыванием_не_считается(self):
        """Пустые и якорь, и замена — это не «допиши», а мусор."""
        data = {"form": "anchor", "path": "a.py", "old": "", "new": ""}
        fixed, hint = pipeline.normalize_edit(data)
        assert fixed["form"] == "anchor"
        assert hint == ""

    def test_другие_формы_не_трогаются(self):
        for form in ("lines", "full", "append"):
            data = {"form": form, "path": "a.py", "content": "x"}
            assert pipeline.normalize_edit(data)[0] is data

    def test_подмена_доходит_до_диска_и_до_отчёта(self, sample):
        out = pipeline.apply_edit(
            {"form": "anchor", "path": "sample.py", "old": "",
             "new": "def div(a, b):\n    return a / b\n"}
        )
        assert "пустой якорь" in out
        assert "Дописано" in out
        text = sample.read_text(encoding="utf-8")
        assert "def add" in text and "def div" in text


class TestWrongPathFromModel:
    """Модель называет файл, которого в проекте нет. Правится кодом, не промптом."""

    def test_несуществующий_путь_подменяется_найденным(self, project):
        result = pipeline.apply_edit(
            {"form": "lines", "path": "main.py", "start": 2, "end": 2, "content": "    return a + b"},
            default_path="calc_mod.py",
        )
        assert "модель назвала файл main.py" in result
        assert "return a + b" in (project / "calc_mod.py").read_text(encoding="utf-8")

    def test_существующий_путь_не_трогается(self, project):
        (project / "second.py").write_text("x = 1\n", encoding="utf-8")
        result = pipeline.apply_edit(
            {"form": "full", "path": "second.py", "content": "x = 2\n"}, default_path="calc_mod.py"
        )
        assert "модель назвала" not in result
        assert (project / "second.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_подмена_не_выводит_за_пределы_каталога(self, project):
        result = pipeline.apply_edit(
            {"form": "full", "path": "../взлом.py", "content": "x = 1\n"}, default_path="calc_mod.py"
        )
        assert "пределы" in result

    def test_без_запасного_пути_подменять_нечем(self, project):
        result = pipeline.apply_edit({"form": "lines", "path": "main.py", "start": 1, "end": 1, "content": "x"})
        assert "Нет такого файла" in result


class TestEditSuccessByJournal:
    """Успех правки определяется журналом, а не текстом сообщения."""

    def test_повторная_правка_того_же_файла_считается(self, sample):
        edit_file("sample.py", "return a + b", "return 0")
        first = guard.change_count()
        edit_file("sample.py", "return 0", "return a + b")
        assert guard.changed_files() == [sample], "файл тот же, путь один"
        assert guard.change_count() == first + 1, "а правок стало на одну больше"

    def test_правка_несуществующего_файла_журнал_не_растит(self, sample):
        before = guard.change_count()
        pipeline.apply_edit({"form": "lines", "path": "нет.py", "start": 1, "end": 1, "content": "x"})
        assert guard.change_count() == before

    def test_конвейер_не_верит_бодрому_сообщению(self, project, monkeypatch):
        """Инструмент ответил текстом без слова «не применена» — а файла не тронул."""
        monkeypatch.setattr(pipeline, "apply_edit", lambda data, default_path="": "Всё отлично, готово!")
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        state = started(Plan("з", []), touched=["calc_mod.py"], path="calc_mod.py")
        pipeline.node_edit(state)
        assert not state.extra["edit_ok"]
        assert pipeline.edge_after_edit(state) == pipeline.READ_NODE


# ====================================================================
# КОНВЕЙЕР: ПРОГОН ЦЕЛИКОМ
# ====================================================================

class TestPipelineRun:
    def test_правка_с_первой_попытки(self, project, monkeypatch, fix_plan):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.extra["tests_green"]
        assert state.extra["attempt"] == 0, "чинить не пришлось"
        assert "return a + b" in (project / "calc_mod.py").read_text(encoding="utf-8")
        assert state.steps == ["plan", "confirm", "step", "step", "deps", "verify", "done"]

    def test_проект_с_нуля(self, workspace, monkeypatch):
        model = fake_model([
            file_answer("def add(a, b):\n    return a + b\n"),
            file_answer("from lib import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"),
        ])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        plan = Plan("з", [Step("create", "lib.py", "функция add"),
                          Step("create", "test_lib.py", "тест"), Step("test", "", "")])

        state = pipeline.run_pipeline("напиши модуль сложения с тестом", plan=plan)
        assert state.extra["tests_green"]
        assert (workspace / "lib.py").exists()
        assert (workspace / "test_lib.py").exists()
        assert state.steps.count("step") == 3

    def test_провал_возвращает_к_чтению_и_правке(self, project, monkeypatch, fix_plan):
        monkeypatch.setattr(pipeline, "request_model", fake_model([
            edit_answer("calc_mod.py", 2, 2, "    return a * b"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ]))
        results = iter([False, True])
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": Run(c, 0 if next(results) else 1, "E  assert 0 == 4", "", 0.2))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.extra["tests_green"]
        assert state.extra["attempt"] == 1
        assert state.steps.count("read") == 1
        assert "return a + b" in (project / "calc_mod.py").read_text(encoding="utf-8")

    def test_второй_круг_показывает_модели_изменённый_файл(self, project, monkeypatch, fix_plan):
        """Ровно та ошибка, из-за которой стрелку пришлось перевести на чтение."""
        model = fake_model([
            edit_answer("calc_mod.py", 2, 2, "    return a * b"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ])
        monkeypatch.setattr(pipeline, "request_model", model)
        results = iter([False, True])
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": Run(c, 0 if next(results) else 1, "E  assert 0 == 4", "", 0.2))
        pipeline.run_pipeline("почини сложение", plan=fix_plan)

        second = model.seen[1][-1]["content"]
        assert "return a * b" in second, "на втором круге модель должна видеть свою же правку"
        assert "return a - b" not in second, "а не текст файла до неё"
        assert "assert 0 == 4" in second, "и текст ошибки"

    def test_исчерпание_попыток_откатывает(self, project, monkeypatch, fix_plan):
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a * b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=False))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert not state.extra["tests_green"]
        assert state.extra["attempt"] == pipeline.MAX_ATTEMPTS
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before
        assert "откачено" in state.answer

    def test_откат_убирает_созданные_файлы(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=False))
        plan = Plan("з", [Step("create", "app.py", "что-то"), Step("test", "", "")])

        state = pipeline.run_pipeline("напиши что-нибудь", plan=plan)
        assert not state.extra["tests_green"]
        assert not (workspace / "app.py").exists(), "наполовину сделанная работа хуже несделанной"

    def test_неприменимая_правка_переспрашивается_в_том_же_шаге(self, project, monkeypatch, fix_plan):
        """Осечка правки не должна останавливать прогон.

        Живой прогон: три задачи подряд кончились «якорь в файле
        не найден» и остановкой — модель придумала строку, которой
        в файле нет, и второго шанса ей никто не дал. Цикл починки эту
        дыру не закрывает: он чинит красную проверку, а провалившийся
        шаг уводит прогон в «проверить нечем», мимо починки.
        """
        monkeypatch.setattr(pipeline, "request_model", fake_model([
            anchor_answer("calc_mod.py", "такого текста нет", "x"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.extra["tests_green"]
        assert not state.extra.get("failed_steps"), "вторая попытка легла, шаг сделан"
        assert state.steps.count("edit") == 0, "чинить нечего: проверка зелёная"
        assert "return a + b" in (project / "calc_mod.py").read_text(encoding="utf-8")

    def test_модель_видит_чем_не_годилась_прошлая_правка(self, project, monkeypatch, fix_plan):
        """Отказ инструмента подробен, и передать его обратно — весь смысл."""
        model = fake_model([
            anchor_answer("calc_mod.py", "такого текста нет", "x"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        pipeline.run_pipeline("почини сложение", plan=fix_plan)

        second = model.seen[1][-1]["content"]
        assert "Якорь в файле не найден" in second
        assert "return a - b" in second, "и похожая строка из файла, которую инструмент нашёл сам"

    def test_пустая_правка_работой_не_считается(self, project, monkeypatch, fix_plan):
        """Якорь, заменённый сам на себя, — не правка, а её видимость.

        Живой прогон на задаче «исправь NameError» три раза подряд дал
        ответ, в котором `new` повторял исходную строку. Инструмент
        честно писал «заменено одно вхождение», журнал считал изменение,
        шаг засчитывался — и агент рапортовал «Готово» об ошибке,
        которая никуда не делась. Пустая правка хуже отказа: отказ виден.
        """
        same = anchor_answer("calc_mod.py", "return a - b", "return a - b\n")
        model = fake_model([same, edit_answer("calc_mod.py", 2, 2, "    return a + b")])
        monkeypatch.setattr(pipeline, "request_model", model)
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert not state.extra.get("failed_steps"), "вторая попытка была настоящей правкой"
        assert "return a + b" in (project / "calc_mod.py").read_text(encoding="utf-8")
        assert "ничего не изменила" in model.seen[1][-1]["content"]

    def test_обе_попытки_мимо_это_провал_шага(self, project, monkeypatch, fix_plan):
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([anchor_answer("calc_mod.py", "такого текста нет", "x")] * 2))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.extra["failed_steps"]
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before

    def test_красная_проверка_после_правки_даёт_ещё_круг(self, project, monkeypatch, fix_plan):
        """А вот это уже работа цикла починки: правка легла, тесты красные."""
        ran = []
        monkeypatch.setattr(pipeline, "request_model", fake_model([
            edit_answer("calc_mod.py", 2, 2, "    return a * b"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ]))
        results = iter([False, True])
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": (ran.append(c),
                                                     Run(c, 0 if next(results) else 1, "E fail", "", 0.2))[1])
        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.extra["tests_green"]
        assert state.steps.count("edit") == 1, "один круг починки"
        assert len(ran) == 2, "первая проверка после шагов, вторая после починки"

    def test_отказ_человека_останавливает_до_первого_действия(self, project, monkeypatch, fix_plan):
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: False)
        monkeypatch.setattr(pipeline, "request_model", fake_model(["{}"]))

        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert state.steps == ["plan", "confirm"]
        assert "Отказано" in state.answer
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before

    def test_человеку_показывают_весь_план(self, project, monkeypatch, fix_plan):
        shown = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: shown.append(d) or False)
        pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert "вернуть a + b" in shown[0]
        assert "1." in shown[0] and "2." in shown[0]

    def test_пустой_план_ничего_не_делает(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model(["{}"]))
        state = pipeline.run_pipeline("непонятная задача", plan=Plan("з", []))
        assert state.steps == ["plan", "confirm"]
        assert "План пуст" in state.answer

    def test_сухой_прогон_не_трогает_файлы(self, project, monkeypatch, fix_plan):
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        guard.set_policy(dry_run=True)
        monkeypatch.setattr(pipeline, "request_model", fake_model(["{}"]))
        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert "Сухой прогон" in state.answer
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before

    def test_состояние_прогона_сериализуемо(self, project, monkeypatch, fix_plan):
        """Из этого следует чекпоинт: снимок конвейера — обычный JSON."""
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        json.dumps(state.to_dict(), ensure_ascii=False)

    def test_отчёт_перечисляет_сделанное(self, project, monkeypatch, fix_plan):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert "1. edit calc_mod.py" in state.answer
        assert "Проверено: pytest" in state.answer


class TestHonestOutcome:
    """«Готово» говорится, только когда сделано всё. Нашёл живой прогон.

    Модель написала модуль с заглушками `pass`, файл тестов потеряла
    по таймауту, `python cart.py` отработал с кодом 0 — и агент
    отчитался «Готово». Отчёт, которому нельзя верить, хуже отсутствия
    отчёта, и здесь закрыты обе дыры сразу: провалившийся шаг и подмена
    проверки запуском.
    """

    def test_шаг_возвращает_пару_а_не_текст(self, workspace):
        """Успех шага — факт, а не слово в сообщении."""
        state = started(Plan("з", [Step("search", "", "")]))
        ok, message = pipeline._run_step(Step("search", "", ""), state)
        assert ok is False and isinstance(message, str)

    def test_провалившийся_шаг_попадает_в_список(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("   ")]))
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert state.extra["failed_steps"] == ["1. create app.py"]
        assert "ПРОВАЛ" in state.extra["log"][0]

    def test_удавшийся_шаг_в_список_не_попадает(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")]))
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert state.extra["failed_steps"] == []

    def test_прогон_продолжается_после_провала(self, workspace, monkeypatch):
        """Следующий шаг может и получиться — обрывать прогон незачем."""
        # Два пустых ответа подряд: шаг создания переспрашивает один раз,
        # и одного плохого ответа теперь мало, чтобы его провалить.
        monkeypatch.setattr(pipeline, "request_model", fake_model(
            [file_answer("   "), file_answer("   "), file_answer("x = 1\n")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        plan = Plan("з", [Step("create", "плохой.py", ""), Step("create", "хороший.py", "")])
        state = pipeline.run_pipeline("з", plan=plan)
        assert (workspace / "хороший.py").exists()
        assert len(state.extra["failed_steps"]) == 1

    def test_зелёная_проверка_не_отменяет_провала_шага(self, workspace, monkeypatch):
        # Два пустых ответа подряд: шаг создания переспрашивает один раз,
        # и одного плохого ответа теперь мало, чтобы его провалить.
        monkeypatch.setattr(pipeline, "request_model", fake_model(
            [file_answer("   "), file_answer("   "), file_answer("x = 1\n")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        plan = Plan("з", [Step("create", "плохой.py", ""), Step("create", "хороший.py", "")])
        state = pipeline.run_pipeline("з", plan=plan)
        assert state.extra["tests_green"]
        assert "Готово" not in state.answer
        assert "Сделано НЕ ВСЁ" in state.answer
        assert "1. create плохой.py" in state.answer
    def test_шаг_создания_переспрашивает_один_раз(self, workspace, monkeypatch):
        """Файл, который не записался, потерян насовсем: чинить нечего."""
        model = fake_model([file_answer("def f(:\n"), file_answer("x = 1\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)

        assert (workspace / "app.py").read_text(encoding="utf-8") == "x = 1\n"
        assert state.extra["failed_steps"] == []
        assert len(model.seen) == 2, "должен быть ровно один переспрос"
        assert "Прошлая попытка не годится" in model.seen[1][-1]["content"]

    def test_две_неудачи_подряд_это_провал_шага(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([file_answer("def f(:\n")] * 3))
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert state.extra["failed_steps"] == ["1. create app.py"]
        assert not (workspace / "app.py").exists()


    def test_всё_сделано_и_проверено(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = pipeline.run_pipeline("з", plan=Plan("з", [Step("create", "app.py", "")]))
        assert state.extra["failed_steps"] == []
        assert state.answer.startswith("Готово")


class TestPromisedTests:
    """План обещал НАПИСАТЬ тесты — проверять запуском файла нельзя.

    Обещанием считается шаг `create test_*.py`, а не шаг `test`.
    Разница стоила одного упавшего прогона: `test` означает «проверь»,
    и промпт планировщика требует ставить его последним ВСЕГДА. Считай
    мы его обещанием — задача «напиши hello world» объявлялась бы
    провалившейся за отсутствие тестов, которых никто не просил.
    """

    def test_обещание_видно_по_шагу_create(self, workspace):
        state = started(Plan("з", [Step("create", "test_app.py", "")]))
        assert state.extra["wants_tests"]

    def test_шаг_test_обещанием_не_является(self, workspace):
        state = started(Plan("з", [Step("create", "app.py", ""), Step("test", "", "")]))
        assert not state.extra["wants_tests"]

    def test_без_обещания_запуск_файла_законен(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = started(Plan("з", [Step("create", "app.py", ""), Step("test", "", "")]),
                        touched=["app.py"])
        pipeline.node_verify(state)
        assert state.extra["verified_by"] == "запуск app.py"
        assert state.extra["tests_green"]

    def test_обещали_тесты_а_их_нет_это_провал(self, workspace, monkeypatch):
        """Ровно то, на чём агент отчитался «Готово» по заглушкам."""
        (workspace / "app.py").write_text("def add(a, b):\n    pass\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = started(
            Plan("з", [Step("create", "app.py", ""), Step("create", "test_app.py", "")]),
            touched=["app.py"],
        )
        pipeline.node_verify(state)
        assert not state.extra["tests_green"]
        assert state.extra["verified_by"] == "нечем"
        assert "обещал тесты" in state.extra["failure"]

    def test_обещали_и_написали_значит_pytest(self, project, monkeypatch):
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = started(Plan("з", [Step("create", "test_calc_mod.py", "")]),
                        touched=["calc_mod.py", "test_calc_mod.py"])
        pipeline.node_verify(state)
        assert state.extra["verified_by"] == "pytest"


class TestWriteTimeout:
    """Написать файл целиком — генерация длиннее реплики, и ждать её надо дольше."""

    def test_на_время_записи_предел_поднимается(self, workspace, monkeypatch):
        seen = []

        def capture(messages, response_format=None, **kwargs):
            seen.append(base.REQUEST_TIMEOUT)
            return file_answer("x = 1\n")

        monkeypatch.setattr(pipeline, "request_model", capture)
        was = base.REQUEST_TIMEOUT
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert seen == [pipeline.WRITE_TIMEOUT]
        assert base.REQUEST_TIMEOUT == was, "предел обязан вернуться на место"

    def test_длина_ответа_ограничена(self, workspace, monkeypatch):
        """Генерация без предела уходит в разгон и теряет весь шаг."""
        model = fake_model([file_answer("x = 1\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert model.options == [{"num_predict": pipeline.WRITE_MAX_TOKENS}]

    def test_предел_возвращается_и_после_исключения(self, workspace, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("обрыв")

        monkeypatch.setattr(pipeline, "request_model", boom)
        was = base.REQUEST_TIMEOUT
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert base.REQUEST_TIMEOUT == was


class TestUnverifiableOutcome:
    """Работу, которую нечем проверить, нельзя объявлять провалом.

    Живой прогон: человек попросил батник для запуска calc.py, агент
    его написал — правильно написал, — а конвейер откатил, потому что
    `.bat` ему проверять нечем. «Не смогли проверить» и «проверили,
    и оно сломано» — разные исходы, и второй из первого не следует.
    """

    def test_батник_остаётся_на_диске(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([file_answer("@echo off\npython calc.py\n")]))
        plan = Plan("з", [Step("create", "run.bat", "запускать calc.py")])
        state = pipeline.run_pipeline("сделай батник", plan=plan)

        assert (workspace / "run.bat").exists(), "правильно сделанную работу не откатывают"
        assert state.extra["unverifiable"]
        assert state.extra.get("rolled_back") is None

    def test_отчёт_говорит_что_проверить_нечем(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("@echo off\n")]))
        state = pipeline.run_pipeline("сделай батник", plan=Plan("з", [Step("create", "run.bat", "")]))
        assert "ПРОВЕРИТЬ НЕЧЕМ" in state.answer
        assert "Готово" not in state.answer, "непроверенное не выдают за проверенное"

    def test_провалившийся_шаг_виден_и_здесь(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([file_answer("   "), file_answer("   "), file_answer("@echo off\n")]))
        plan = Plan("з", [Step("create", "плохой.bat", ""), Step("create", "run.bat", "")])
        state = pipeline.run_pipeline("сделай батники", plan=plan)
        assert "Сделано НЕ ВСЁ" in state.answer
        assert "проверить нечем" in state.answer

    def test_красный_прогон_по_прежнему_откатывается(self, project, monkeypatch, fix_plan):
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a * b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=False))
        state = pipeline.run_pipeline("почини сложение", plan=fix_plan)
        assert "откачено" in state.answer
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before


class TestShellBuiltins:
    """`cd` — не программа, а команда оболочки, которой у нас нет."""

    @pytest.mark.parametrize("command", ["cd path_to_your_directory", "echo привет", "set X=1", "dir"])
    def test_встроенные_команды_отвергаются(self, workspace, command):
        allowed, reason = guard.command_allowed(command)
        assert not allowed
        assert "встроенная команда оболочки" in reason

    def test_отказ_объясняет_про_рабочий_каталог(self, workspace):
        _, reason = guard.command_allowed("cd ..")
        assert "каталог у агента один" in reason

    def test_обычные_программы_не_задеты(self, workspace):
        for command in ("python app.py", "pip install requests", "git status"):
            assert guard.command_allowed(command)[0], command

    def test_план_с_такой_командой_получает_претензию(self, workspace):
        plan = Plan("з", [Step("run", "cd path_to_your_directory", ""), Step("test", "", "")])
        assert any("встроенная команда" in c for c in validate_plan(plan))

    def test_шаг_проваливается_внятно(self, workspace):
        state = started(Plan("з", [Step("run", "cd куда-то", "")]))
        pipeline.node_step(state)
        assert "встроенная команда оболочки" in state.extra["log"][0]
        assert state.extra["failed_steps"] == ["1. run cd куда-то"]


class TestCoderModelSwitch:
    def test_по_умолчанию_модель_курса(self):
        assert pipeline.coder_model() == base.MODEL

    def test_переменная_меняет_модель(self, monkeypatch):
        monkeypatch.setattr(pipeline, "CODER_MODEL", "qwen2.5-coder:7b")
        assert pipeline.coder_model() == "qwen2.5-coder:7b"

    def test_запрос_идёт_под_выбранной_моделью(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "CODER_MODEL", "qwen2.5-coder:7b")
        seen = []

        def capture(messages, response_format=None, **kwargs):
            seen.append(base.MODEL)
            return file_answer("x = 1\n")

        monkeypatch.setattr(pipeline, "request_model", capture)
        state = started(Plan("з", [Step("create", "app.py", "")]))
        pipeline.node_step(state)
        assert seen == ["qwen2.5-coder:7b"]
        assert base.MODEL != "qwen2.5-coder:7b", "модель курса должна вернуться на место"


# ====================================================================
# ВТОРАЯ СБОРКА КОНВЕЙЕРА: LANGGRAPH
# ====================================================================

needs_langgraph = pytest.mark.skipif(
    not pipeline_lg.LANGGRAPH_AVAILABLE, reason="LangGraph не установлен (pip install langgraph)"
)


@needs_langgraph
class TestLangGraphPipeline:
    def test_собирается(self):
        assert pipeline_lg.build_langgraph_pipeline() is not None

    def test_даёт_тот_же_результат_что_свой_граф(self, project, monkeypatch, fix_plan):
        """Главное свойство второй сборки: она делает ровно то же самое."""
        def run_with(runner):
            (project / "calc_mod.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            guard.forget_changes()
            monkeypatch.setattr(pipeline, "request_model",
                                fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
            monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
            return runner("почини сложение", plan=fix_plan)

        mine = run_with(pipeline.run_pipeline)
        theirs = run_with(pipeline_lg.run_langgraph_pipeline)

        assert mine.steps == theirs.steps
        assert mine.extra["tests_green"] == theirs.extra["tests_green"]
        assert mine.extra["attempt"] == theirs.extra["attempt"]

    def test_трейс_прогона_заполняется(self, project, monkeypatch, fix_plan):
        """LangGraph ведёт свой трейс и в наш объект не пишет — пишет обёртка."""
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        state = pipeline_lg.run_langgraph_pipeline("почини сложение", plan=fix_plan)
        assert state.trace() == "plan -> confirm -> step -> step -> deps -> verify -> done"

    def test_цикл_по_шагам_работает_и_здесь(self, workspace, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")] * 3))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        plan = Plan("з", [Step("create", "a.py", ""), Step("create", "b.py", ""),
                          Step("create", "c.py", ""), Step("test", "", "")])
        state = pipeline_lg.run_langgraph_pipeline("з", plan=plan)
        assert state.steps.count("step") == 4

    def test_цикл_починки_работает_и_здесь(self, project, monkeypatch, fix_plan):
        monkeypatch.setattr(pipeline, "request_model", fake_model([
            edit_answer("calc_mod.py", 2, 2, "    return a * b"),
            edit_answer("calc_mod.py", 2, 2, "    return a + b"),
        ]))
        results = iter([False, True])
        monkeypatch.setattr(pipeline, "execute",
                            lambda c, timeout=None, feed="": Run(c, 0 if next(results) else 1, "E fail", "", 0.2))
        state = pipeline_lg.run_langgraph_pipeline("почини сложение", plan=fix_plan)
        assert state.steps.count("edit") == 1
        assert state.extra["tests_green"]

    def test_отказ_человека_кончает_прогон(self, project, monkeypatch, fix_plan):
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: False)
        monkeypatch.setattr(pipeline, "request_model", fake_model(["{}"]))
        state = pipeline_lg.run_langgraph_pipeline("почини сложение", plan=fix_plan)
        assert state.steps == ["plan", "confirm"]


class TestLangGraphAbsence:
    def test_без_библиотеки_модуль_импортируется(self):
        """Необязательная зависимость не должна ронять импорт главы."""
        assert isinstance(pipeline_lg.LANGGRAPH_AVAILABLE, bool)

    def test_сборка_без_библиотеки_объясняет_причину(self, monkeypatch):
        monkeypatch.setattr(pipeline_lg, "LANGGRAPH_AVAILABLE", False)
        monkeypatch.setattr(pipeline_lg, "IMPORT_PROBLEM", "No module named 'langgraph'")
        with pytest.raises(RuntimeError, match="pip install langgraph"):
            pipeline_lg.build_langgraph_pipeline()


# ====================================================================
# КАРТА КОДА: ЧТО ЕСТЬ В ПРОЕКТЕ И ЧТО ОНО ДЕЛАЕТ
# ====================================================================

SAMPLE_MODULE = (
    "import math\n"
    "\n\n"
    "def solve(a, b, c):\n"
    '    """Возвращает корни уравнения.\n'
    "\n"
    "    Вторая строка описания в карту не идёт.\n"
    '    """\n'
    "    return a\n"
    "\n\n"
    "class Cart:\n"
    '    """Корзина покупок."""\n'
    "\n"
    "    def add(self, item):\n"
    "        return item\n"
    "\n\n"
    "def main():\n"
    "    def inner():\n"
    "        return 1\n"
    "    return inner()\n"
)


@pytest.fixture
def mapped(workspace, monkeypatch):
    """Проект с картой во временном файле: настоящая лежит рядом с главой."""
    monkeypatch.setenv("AGENT_CODEMAP_FILE", str(workspace / "codemap.json"))
    (workspace / "quadratic.py").write_text(SAMPLE_MODULE, encoding="utf-8")
    codemap.forget_cache()
    yield workspace
    codemap.forget_cache()


class TestCodeMap:
    """Карта проекта: имена, границы и назначение каждой функции.

    Заводится ради выбора МЕСТА правки. Пока место ищется текстовым
    поиском, оно находится только если человек процитировал свой код.
    Стоит ему сказать по-человечески — «сделай чтобы приложение выводило
    ещё и производную», — и искать нечего: агент угадывает файл или
    заводит новый рядом с тем, который просили поправить.
    """

    def test_функции_и_классы_видны(self, mapped):
        assert codemap.names() == ["solve", "Cart", "Cart.add", "main"]

    def test_метод_называется_вместе_с_классом(self, mapped):
        """`add` в проекте бывает не один, а выбирать модель будет по имени."""
        assert codemap.find("Cart.add").kind == "метод"

    def test_вложенная_функция_в_карту_не_идёт(self, mapped):
        """Заменять замыкание отдельно от его функции бессмысленно."""
        assert "inner" not in codemap.names()

    def test_назначение_берётся_из_докстринга(self, mapped):
        assert codemap.find("solve").purpose == "Возвращает корни уравнения."

    def test_границы_включают_всё_определение(self, mapped):
        item = codemap.find("solve")
        source = codemap.source_of(item)
        assert source.startswith("def solve(a, b, c):")
        assert source.rstrip().endswith("return a")

    def test_декоратор_входит_в_границы(self, workspace, monkeypatch):
        """Замена по строкам без декоратора оставила бы `@tool` над чужим определением."""
        monkeypatch.setenv("AGENT_CODEMAP_FILE", str(workspace / "codemap.json"))
        (workspace / "tools.py").write_text(
            "def tool(f):\n    return f\n\n\n@tool\ndef run():\n    return 1\n", encoding="utf-8")
        codemap.forget_cache()
        assert codemap.source_of(codemap.find("run")).startswith("@tool")

    def test_метод_ищется_по_короткому_имени(self, mapped):
        assert codemap.find("add").name == "Cart.add"

    def test_сломанный_файл_карту_не_роняет(self, workspace, monkeypatch):
        monkeypatch.setenv("AGENT_CODEMAP_FILE", str(workspace / "codemap.json"))
        (workspace / "broken.py").write_text("def f(:\n", encoding="utf-8")
        (workspace / "fine.py").write_text("def g():\n    return 1\n", encoding="utf-8")
        codemap.forget_cache()
        assert codemap.names() == ["g"]

    def test_подкаталоги_тоже_читаются(self, mapped):
        (mapped / "src").mkdir()
        (mapped / "src" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
        codemap.forget_cache()
        assert "helper" in codemap.names()

    def test_карта_пересобирается_после_правки(self, mapped):
        assert "extra" not in codemap.names()
        path = mapped / "quadratic.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n\ndef extra():\n    return 2\n",
                        encoding="utf-8")
        os.utime(path, (time.time() + 1, time.time() + 1))
        assert "extra" in codemap.names()

    def test_пустой_каталог_даёт_пустую_карту(self, workspace, monkeypatch):
        monkeypatch.setenv("AGENT_CODEMAP_FILE", str(workspace / "codemap.json"))
        codemap.forget_cache()
        assert codemap.names() == []


class TestPurposes:
    """Назначение: докстринг, потом хранилище, и только потом модель."""

    def test_модель_зовётся_только_для_безымянных(self, mapped, monkeypatch):
        model = fake_model([json.dumps({"purposes": [{"name": "Cart.add", "purpose": "перебирает корзину"},
                                                  {"name": "main", "purpose": "точка входа"}]})])
        monkeypatch.setattr(codemap, "request_model", model)
        described = codemap.describe(codemap.scan())

        assert described == 2, "у solve и Cart описания уже есть, спрашивать нечего"
        asked = model.seen[0][-1]["content"]
        assert "Cart.add" in asked and "main" in asked
        assert "solve" not in asked.split("Исходный код:")[0]

    def test_описание_переживает_перезапуск(self, mapped, monkeypatch):
        monkeypatch.setattr(codemap, "request_model",
                            fake_model([json.dumps({"purposes": [{"name": "Cart.add", "purpose": "перебирает корзину"},
                                                  {"name": "main", "purpose": "точка входа"}]})]))
        codemap.describe(codemap.scan())
        codemap.forget_cache()
        assert codemap.find("main").purpose == "Точка входа"

    def test_правка_функции_обесценивает_описание(self, mapped, monkeypatch):
        """Описание живёт дольше кода, и без отпечатка тела оно врало бы."""
        monkeypatch.setattr(codemap, "request_model",
                            fake_model([json.dumps({"purposes": [{"name": "Cart.add", "purpose": "перебирает корзину"},
                                                  {"name": "main", "purpose": "точка входа"}]})]))
        codemap.describe(codemap.scan())

        path = mapped / "quadratic.py"
        path.write_text(path.read_text(encoding="utf-8").replace("return inner()", "return 42"),
                        encoding="utf-8")
        os.utime(path, (time.time() + 1, time.time() + 1))
        assert codemap.find("main").purpose == "", "тело изменилось — описание больше не про него"

    def test_описание_едет_к_своей_функции(self, mapped, monkeypatch):
        """Живой прогон: на паре get_roots/main модель описала первую дважды.

        Порядок ответа проверить нечем — перепутанный выглядит как
        правильный. Имя из `enum` проверяется грамматикой, а повтор
        отбрасывается: описание, приехавшее дважды, означает, что модель
        спутала функции, и второе такое же неверно, как первое.
        """
        monkeypatch.setattr(codemap, "request_model", fake_model([json.dumps({"purposes": [
            {"name": "main", "purpose": "точка входа"},
            {"name": "main", "purpose": "и снова точка входа"},
        ]})]))
        assert codemap.describe(codemap.scan()) == 1
        assert codemap.find("main").purpose == "Точка входа"
        assert codemap.find("Cart.add").purpose == "", "своего описания не приехало"

    def test_молчание_модели_карту_не_роняет(self, mapped, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("Ollama молчит")

        monkeypatch.setattr(codemap, "request_model", boom)
        assert codemap.describe(codemap.scan()) == 0
        assert codemap.names(), "карта остаётся, просто без описаний"

    def test_имя_в_схеме_это_перечисление(self):
        """Назвать функцию, которой нет, модель не сможет физически."""
        schema = codemap.purpose_schema(["solve", "main"])
        item = schema["properties"]["purposes"]["items"]
        assert item["properties"]["name"]["enum"] == ["solve", "main"]
        assert schema["properties"]["purposes"]["maxItems"] == 2

    @pytest.mark.parametrize("name,answer,expected", [
        ("solve", "Функция solve(a, b, c) решает уравнение", "Решает уравнение"),
        ("main", "Функция main() запрашивает коэффициенты", "Запрашивает коэффициенты"),
        ("Cart.add", "добавляет товар", "Добавляет товар"),
        ("derivative", "derivative(a, b) — считает производную", "Считает производную"),
        # Имя снимается только целым словом: без проверки границы
        # описание метода `add` со словом «Adds» превращалось
        # в «S an item to the cart».
        ("Cart.add", "Adds an item to the cart", "Adds an item to the cart"),
        ("total", "Totals everything", "Totals everything"),
    ])
    def test_начало_ответа_подчищается(self, name, answer, expected):
        """«Без слова функция в начале» модель нарушает: так написано в её данных."""
        assert codemap.tidy_purpose(name, answer) == expected

    def test_длинное_описание_режется_по_слову(self):
        long = "Решает " + "очень " * 40 + "длинно"
        tidied = codemap.tidy_purpose("f", long)
        assert len(tidied) <= codemap.PURPOSE_LIMIT + 1
        assert tidied.endswith("…")


class TestChoosePlace:
    """Выбор места правки по карте: одно имя из перечисления.

    Ради этого карта и заводилась. Придумать имя модель не может — его
    нет в грамматике; перепутать файл не может — имя указывает и на файл,
    и на строки. Остаётся один способ ошибиться: выбрать не ту функцию,
    и это человек видит в плане до того, как что-то изменится.
    """

    @pytest.fixture(autouse=True)
    def _described(self, workspace, monkeypatch):
        """Настоящий выбор вместо заглушки, и без похода за описаниями.

        Фикстура `workspace` подменяет `choose` пустышкой, чтобы быстрые
        тесты не ходили к модели. Здесь проверяется он сам, поэтому
        возвращается настоящий, а модель подделывается ответом.

        `workspace` в аргументах — не ради каталога, а ради ПОРЯДКА:
        без него автоматическая фикстура отработает раньше, и подмена
        из `workspace` перекроет нашу.
        """
        monkeypatch.setattr(codemap, "choose", REAL_CHOOSE)
        monkeypatch.setattr(codemap, "describe", lambda items, model=None: 0)

    def test_имя_в_схеме_это_перечисление(self):
        schema = codemap.choose_schema(["solve", "main"])
        assert schema["properties"]["name"]["enum"] == ["solve", "main", codemap.NOTHING_FITS]

    def test_ответ_ни_одна_значит_новая_функция(self, mapped, monkeypatch):
        monkeypatch.setattr(codemap, "request_model",
                            fake_model([json.dumps({"name": codemap.NOTHING_FITS})]))
        assert codemap.choose("добавь совсем другое") is None

    def test_выбранная_функция_возвращается_с_границами(self, mapped, monkeypatch):
        monkeypatch.setattr(codemap, "request_model", fake_model([json.dumps({"name": "solve"})]))
        place = codemap.choose("поправь корни")
        assert (place.name, place.path, place.start) == ("solve", "quadratic.py", 4)

    def test_список_можно_сузить_до_файла(self, mapped, monkeypatch):
        (mapped / "other.py").write_text("def elsewhere():\n    return 1\n", encoding="utf-8")
        codemap.forget_cache()
        model = fake_model([json.dumps({"name": "solve"})])
        monkeypatch.setattr(codemap, "request_model", model)
        codemap.choose("поправь корни", "quadratic.py")

        asked = model.seen[0][-1]["content"]
        assert "elsewhere" not in asked, "короткий список модель читает внимательнее"

    def test_классы_в_выбор_не_идут(self, mapped, monkeypatch):
        """Заменить класс целиком — это не правка, а перезапись файла."""
        model = fake_model([json.dumps({"name": "solve"})])
        monkeypatch.setattr(codemap, "request_model", model)
        codemap.choose("поправь корзину")
        asked = model.seen[0][-1]["content"]
        assert "Cart.add" in asked
        assert "\nquadratic.py:9 Cart " not in asked

    def test_молчание_модели_выбор_не_роняет(self, mapped, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("Ollama молчит")

        monkeypatch.setattr(codemap, "request_model", boom)
        assert codemap.choose("поправь корни") is None

    def test_пустой_проект_модель_не_беспокоит(self, workspace, monkeypatch):
        called = []
        monkeypatch.setattr(codemap, "request_model", lambda *a, **k: called.append(1) or "{}")
        codemap.forget_cache()
        assert codemap.choose("поправь что-нибудь") is None
        assert called == [], "выбирать не из чего — спрашивать не о чем"


class TestReview:
    """Разбор написанного: сделано ли то, о чём просили.

    Единственная проверка конвейера, где спрашивают модель. Все
    остальные механические и отвечают на вопрос «работает ли»;
    ни одна не отвечает на вопрос «то ли это». А проваливается агент
    чаще всего именно там: функция объявлена, файл импортируется, тесты
    зелёные — и производная, которую просили печатать, не печатается.
    """

    MODULE = "def solve(a):\n    return a * 2\n\n\nprint(solve(2))\n"

    @pytest.fixture
    def written(self, workspace, monkeypatch):
        """Настоящий разбор вместо заглушки из `workspace`, и файл под него.

        `workspace` в аргументах не только ради каталога, но и ради
        порядка: он подменяет разбор пустышкой, чтобы быстрые тесты
        не ходили к модели, а здесь проверяется он сам.
        """
        monkeypatch.setattr(review, "review", REAL_REVIEW)
        (workspace / "app.py").write_text(self.MODULE, encoding="utf-8")
        return workspace

    def answer(self, done, problems):
        return lambda messages, response_format=None: json.dumps(
            {"done": done, "problems": problems})

    def test_всё_сделано_значит_претензий_нет(self, written):
        done, problems = review.review("посчитай удвоение", "app.py",
                                       self.answer(True, []))
        assert (done, problems) == (True, [])

    def test_претензия_с_настоящей_цитатой_принимается(self, written):
        done, problems = review.review("выведи ещё и квадрат", "app.py", self.answer(
            False, [{"quote": "print(solve(2))", "what": "квадрат не печатается",
                     "how": "добавить второй print"}]))
        assert done is False
        assert problems[0].what == "квадрат не печатается"

    def test_выдуманная_цитата_выбрасывается(self, written):
        """Модель на 3B охотно выдумывает проблемы, если её о них спросить."""
        done, problems = review.review("выведи квадрат", "app.py", self.answer(
            False, [{"quote": "print(квадрат)", "what": "нет квадрата"}]))
        assert (done, problems) == (True, []), "нечего предъявить — значит, нечего чинить"

    def test_отступ_и_пробелы_цитате_не_мешают(self, written):
        done, problems = review.review("з", "app.py", self.answer(
            False, [{"quote": "  return a * 2  ", "what": "удвоение вместо квадрата"}]))
        assert len(problems) == 1

    def test_претензий_больше_трёх_не_берут(self, written):
        raw = [{"quote": "return a * 2", "what": f"беда {n}"} for n in range(6)]
        _, problems = review.review("з", "app.py", self.answer(False, raw))
        assert len(problems) == review.MAX_PROBLEMS

    def test_молчание_модели_прогон_не_краснит(self, written):
        def boom(*a, **k):
            raise ConnectionError("Ollama молчит")

        assert review.review("з", "app.py", boom) == (True, [])

    def test_пустой_файл_разбирать_нечего(self, workspace):
        (workspace / "empty.py").write_text("", encoding="utf-8")
        assert review.review("з", "empty.py", self.answer(False, [])) == (True, [])

    def test_схема_требует_оба_поля(self):
        schema = review.review_schema()
        assert set(schema["required"]) == {"done", "problems"}
        assert schema["properties"]["problems"]["maxItems"] == review.MAX_PROBLEMS


class TestDoubt:
    """Что конвейер делает с мнением модели: один круг починки и отчёт.

    Механическая проверка говорит о факте, разбор — о мнении, и цена
    ошибки у них разная. Красные тесты — повод вернуть файлы как было;
    «мне кажется, задача не выполнена» — повод сказать об этом человеку.
    Работу, прошедшую все механические проверки, из-за мнения не удаляют.
    """

    def test_сомнение_даёт_круг_починки(self, workspace):
        state = started(Plan("з", []), tests_green=True, doubt="- квадрат не печатается")
        assert pipeline.edge_after_verify(state) == "read"
        assert state.extra["failure"] == "- квадрат не печатается"

    def test_второго_круга_по_тому_же_поводу_нет(self, workspace):
        state = started(Plan("з", []), tests_green=True,
                        doubt="- квадрат не печатается", doubted=True)
        assert pipeline.edge_after_verify(state) == "done"

    def test_сомнение_не_откатывает(self, workspace, monkeypatch):
        """Зелёная проверка плюс мнение — это не повод удалять файлы."""
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        monkeypatch.setattr(review, "REVIEW", "on")
        monkeypatch.setattr(review, "review",
                            lambda task, path, model_call=None: (False, [review.Problem("print(1)", "не то")]))
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)

        assert state.extra["tests_green"] is True, "механические проверки прошли"
        assert "не то" in state.extra["doubt"]

    def test_итог_говорит_о_сомнении(self, workspace):
        state = started(Plan("з", []), tests_green=True, verified_by="запуск app.py",
                        doubt="- квадрат не печатается", doubted=True)
        pipeline.node_done(state)
        assert "СДЕЛАНО, ПОХОЖЕ, НЕ ТО" in state.answer
        assert "квадрат не печатается" in state.answer
        assert "Файлы на месте" in state.answer

    def test_снятое_сомнение_в_итог_не_попадает(self, workspace):
        state = started(Plan("з", []), tests_green=True, verified_by="запуск app.py", doubt="")
        pipeline.node_done(state)
        assert state.answer.startswith("Готово.")

    def test_по_умолчанию_разбор_выключен(self):
        """Решение замера 9: на 3B он не ловит ничего, а время тратит."""
        assert review.REVIEW == "off"

    def test_разбор_выключается_переключателем(self, workspace, monkeypatch):
        (workspace / "app.py").write_text("print(1)\n", encoding="utf-8")
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        monkeypatch.setattr(review, "REVIEW", "off")
        called = []
        monkeypatch.setattr(review, "review", lambda *a, **k: called.append(1) or (True, []))
        state = started(Plan("з", []), touched=["app.py"])
        pipeline.node_verify(state)
        assert called == []


class TestEditFunction:
    """Правка ОДНОЙ функции по её границам — форма, у которой нет адреса.

    Обычная правка требует от модели адреса: процитировать якорь
    (промахивается), назвать номера строк (считает неверно) или
    перепечатать файл целиком (теряет чужой код). Здесь адрес не нужен:
    границы известны из разбора, и модели остаётся написать тело.
    """

    MODULE = (
        "import math\n"
        "\n\n"
        "def get_roots(a, b, c):\n"
        "    return a\n"
        "\n\n"
        "def main():\n"
        "    print(get_roots(1, 2, 3))\n"
    )

    @pytest.fixture
    def project_map(self, workspace, monkeypatch):
        (workspace / "app.py").write_text(self.MODULE, encoding="utf-8")
        codemap.forget_cache()
        # Границы берём из карты, а не пишем руками: рукописные
        # разъезжаются с файлом при первой же его правке, и тест
        # начинает проверять не то, что написано в его названии.
        place = REAL_FIND("main")
        monkeypatch.setattr(codemap, "choose", lambda task, path="": place)
        return workspace

    def test_модели_показывают_одну_функцию(self, project_map, monkeypatch):
        model = fake_model([file_answer("def main():\n    print('готово')\n")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        asked = model.seen[0][-1]["content"]
        assert "СЕЙЧАС НУЖНА ОДНА ФУНКЦИЯ: main" in asked
        assert "def main():" in asked
        assert "def get_roots" not in asked.split("Вот она целиком:")[1], "соседняя функция не нужна"

    def test_функция_заменяется_по_своим_границам(self, project_map, monkeypatch):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([file_answer("def main():\n    print('готово')\n")]))
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        written = (project_map / "app.py").read_text(encoding="utf-8")
        assert "print('готово')" in written
        assert "def get_roots(a, b, c):" in written, "соседняя функция на месте"
        assert state.extra["edit_form"] == "function"

    def test_ответ_без_нужной_функции_отвергается(self, project_map, monkeypatch):
        """Модель, которой показали одну функцию, иногда возвращает соседнюю."""
        wrong = file_answer("def get_roots(a, b, c):\n    return 42\n")
        monkeypatch.setattr(pipeline, "request_model", fake_model([wrong, wrong]))
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        assert (project_map / "app.py").read_text(encoding="utf-8") == self.MODULE
        assert "нет функции main" in " ".join(state.extra["log"])

    def test_новое_рядом_с_правкой_разрешено(self, project_map, monkeypatch):
        """Чтобы `main` печатала производную, её надо ещё и написать."""
        answer = file_answer("def main():" + chr(10) + "    print(helper())" + chr(10) * 2 +
                             "def helper():" + chr(10) + "    return 1" + chr(10))
        monkeypatch.setattr(pipeline, "request_model", fake_model([answer]))
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        written = (project_map / "app.py").read_text(encoding="utf-8")
        assert "def helper():" in written
        assert written.count("def get_roots") == 1

    def test_повтор_существующего_определения_отвергается(self, project_map, monkeypatch):
        """Живой прогон: в файле оказались два `main` и три `get_roots`.

        `replace_lines` кладёт ответ на место ОДНОЙ функции, поэтому
        соседи, приехавшие заодно, не заменяют старых, а добавляются
        к ним. Файл запускается, проверки зелёные, а читать его нельзя.
        """
        answer = file_answer("def main():" + chr(10) + "    print(1)" + chr(10) * 2 +
                             "def get_roots(a, b, c):" + chr(10) + "    return 0" + chr(10))
        monkeypatch.setattr(pipeline, "request_model", fake_model([answer, answer, answer]))
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        assert "заново определены get_roots" in " ".join(state.extra["log"])
        assert (project_map / "app.py").read_text(encoding="utf-8").count("def get_roots") == 1

    def test_не_вышло_с_функцией_значит_обычная_правка(self, project_map, monkeypatch):
        """Место могло быть выбрано неверно — тогда правка нужна не там."""
        wrong = file_answer("x = 1\n")
        model = fake_model([wrong, wrong,
                            anchor_answer("app.py", "    return a", "    return a * 2")])
        monkeypatch.setattr(pipeline, "request_model", model)
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)

        assert "return a * 2" in (project_map / "app.py").read_text(encoding="utf-8")
        assert len(model.seen) == 3, "две попытки функцией, потом обычная форма"

    def test_исчезнувшая_функция_не_роняет_шаг(self, project_map, monkeypatch):
        """Между выбором места и правкой файл могли переписать."""
        monkeypatch.setattr(codemap, "choose",
                            lambda task, path="": codemap.Definition("app.py", "main", "", 500, 501, "функция"))
        monkeypatch.setattr(pipeline, "request_model", fake_model([file_answer("x = 1\n")] * 3))
        state = started(Plan("з", [Step("search", "app.py", ""), Step("edit", "", "")]))
        pipeline.node_step(state)
        pipeline.node_step(state)
        assert "больше не находится" in " ".join(state.extra["log"])


class TestSearchByMap:
    """Шаг поиска: путь, потом текст, потом карта."""

    def test_карта_подключается_когда_текста_не_нашлось(self, project, monkeypatch):
        place = codemap.Definition("calc_mod.py", "add", "a, b", 1, 2, "функция")
        monkeypatch.setattr(codemap, "choose", lambda task, path="": place)
        state = started(Plan("з", [Step("search", "такого текста нет нигде", "")]))
        pipeline.node_step(state)

        assert state.extra["path"] == "calc_mod.py"
        assert state.extra["place"]["name"] == "add"
        assert "по карте кода" in state.extra["log"][0]

    def test_карта_молчит_значит_не_нашлось(self, project, monkeypatch):
        monkeypatch.setattr(codemap, "choose", lambda task, path="": None)
        state = started(Plan("з", [Step("search", "такого текста нет нигде", "")]))
        pipeline.node_step(state)
        assert "ничего не нашлось" in state.extra["log"][0]
        assert "place" not in state.extra

    def test_найденный_текстом_файл_тоже_уточняется_картой(self, project, monkeypatch):
        place = codemap.Definition("calc_mod.py", "add", "a, b", 1, 2, "функция")
        asked = []
        monkeypatch.setattr(codemap, "choose",
                            lambda task, path="": asked.append(path) or place)
        state = started(Plan("з", [Step("search", "def add", "")]))
        pipeline.node_step(state)

        assert asked == ["calc_mod.py"], "файл известен — сужаем список до него"
        assert state.extra["place"]["name"] == "add"

    def test_известный_файл_сужается_до_функции(self, project, monkeypatch):
        """Знание файла не отвечает на вопрос, какую функцию править.

        Первая версия здесь выходила сразу — «файл назван, вопрос
        решён», — и правка функции не включалась никогда: план правки
        всегда ставит в шаг `search` путь, а не цитату.
        """
        place = codemap.Definition("calc_mod.py", "add", "a, b", 1, 2, "функция")
        asked = []
        monkeypatch.setattr(codemap, "choose",
                            lambda task, path="": asked.append(path) or place)
        state = started(Plan("з", [Step("search", "calc_mod.py", "")]))
        pipeline.node_step(state)

        assert state.extra["path"] == "calc_mod.py"
        assert asked == ["calc_mod.py"]
        assert state.extra["place"]["name"] == "add"


# ====================================================================
# АГЕНТ: ВХОД, ШЕСТОЙ СПЕЦИАЛИСТ, ОТЧЁТ
# ====================================================================

class TestSessionMemory:
    """Что агент помнит между запусками — на хранилище Главы 3.

    Живые прогоны начинались одинаково: человек набирал `каталог
    E:\\progects\\CodeAgentTests`, потому что агент про свой каталог
    не помнил ничего. А потом писал «добавь ожидание ввода» — без
    имени файла, потому что имел в виду тот, который они только что
    писали вместе, и считал это очевидным. Оно и очевидно — для того,
    у кого есть память.
    """

    @pytest.fixture
    def memory(self, tmp_path):
        return SessionMemory(tmp_path / "session.json")

    def test_пустая_память_не_врёт(self, memory):
        assert memory.get("workspace") == ""
        assert "пуста" in memory.report()

    def test_запомнил_и_вспомнил(self, memory):
        memory.set("workspace", "E:/work")
        assert memory.get("workspace") == "E:/work"

    def test_память_переживает_перезапуск(self, tmp_path):
        SessionMemory(tmp_path / "session.json").set("current_file", "main.py")
        assert SessionMemory(tmp_path / "session.json").get("current_file") == "main.py"

    def test_пустое_значение_стирает_а_не_пишет_пустоту(self, memory):
        memory.set("current_file", "main.py")
        memory.set("current_file", "   ")
        assert memory.get("current_file") == ""

    def test_каталог_запоминается(self, memory, workspace):
        memory.note_workspace()
        assert Path(memory.get("workspace")) == workspace

    def test_каталог_восстанавливается(self, memory, workspace, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_WORKSPACE", raising=False)
        memory.note_workspace()
        other = tmp_path / "другой"
        other.mkdir()
        guard.set_workspace(other)
        assert Path(memory.restore_workspace()) == workspace
        assert guard.get_workspace() == workspace

    def test_исчезнувший_каталог_не_восстанавливают(self, memory, tmp_path, monkeypatch):
        """Каталог могли удалить между запусками, а агент без корня —
        это агент без единственной границы, внутри которой ему можно писать.
        """
        monkeypatch.delenv("AGENT_WORKSPACE", raising=False)
        memory.set("workspace", str(tmp_path / "которого-нет"))
        before = guard.get_workspace()
        assert memory.restore_workspace() == ""
        assert guard.get_workspace() == before

    def test_переменная_окружения_сильнее_памяти(self, memory, tmp_path, monkeypatch):
        """Кто её выставил, сделал это только что; память — про прошлый раз."""
        monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
        memory.set("workspace", str(tmp_path))
        assert memory.restore_workspace() == ""

    def test_запоминается_файл_и_задача(self, memory):
        memory.note_work("поправь сложение", ["calc.py"])
        assert memory.get("current_file") == "calc.py"
        assert memory.get("last_task") == "поправь сложение"

    def test_файл_тестов_рабочим_не_считается(self, memory):
        """Работали над модулем, тест к нему — следствие, а не адрес правки."""
        memory.note_work("напиши модуль", ["calc.py", "test_calc.py"])
        assert memory.get("current_file") == "calc.py"

    def test_неудачный_прогон_не_стирает_адрес(self, memory):
        """«Ну тогда сделай иначе» — про тот же файл, что и прошлая попытка."""
        memory.note_work("напиши модуль", ["calc.py"])
        memory.note_work("сделай иначе", [])
        assert memory.get("current_file") == "calc.py"
        assert memory.get("last_task") == "сделай иначе"

    def test_забыть_можно_всё(self, memory):
        memory.note_work("задача", ["calc.py"])
        memory.note_workspace()
        memory.forget_all()
        assert all(memory.get(key) == "" for key in KEYS)
        assert "пуста" in memory.report()

    def test_отчёт_показывает_только_известное(self, memory):
        memory.set("current_file", "calc.py")
        report = memory.report()
        assert "calc.py" in report
        assert "Последняя задача" not in report


class TestAgentEntry:
    """Главная развилка агента: задача идёт в конвейер, вопрос — к специалистам."""

    @pytest.mark.parametrize(
        "text",
        ["напиши приложение на python которое выводит hello world",
         "создай модуль корзины покупок",
         "сделай парсер csv с тестами",
         "поправь функцию add в calc.py",
         "почини сложение",
         "добавь функцию деления",
         "запусти тесты",
         "реализуй сортировку пузырьком"],
    )
    def test_это_задача(self, text):
        assert agent8.looks_like_task(text)

    @pytest.mark.parametrize(
        "text",
        ["где реализован калькулятор",
         "как устроен реестр инструментов",
         "что такое эмбеддинг",
         "меня зовут io982",
         "покажи git log",
         "расскажи про свои инструменты",
         "объясни, как работает граф"],
    )
    def test_это_вопрос(self, text):
        assert not agent8.looks_like_task(text)

    def test_показать_важнее_чем_сделать(self, ):
        """«Покажи diff» начинается с глагола, но задачей не является."""
        assert not agent8.looks_like_task("покажи изменения")

    @pytest.mark.parametrize("task", [
        "поправь функцию там, где вычисляется сумма",
        "исправь место, где падает тест",
        "добавь проверку туда, где ввод",
        "почини то, где считается скидка",
    ])
    def test_слово_где_внутри_задачи_её_не_отменяет(self, task):
        """Самая обычная форма просьбы о правке, и она уезжала в ответ текстом.

        Вопросные слова ищутся только в НАЧАЛЕ реплики. Пока проверка
        шла вхождением куда угодно, «где» ловилось в середине задачи —
        и агент отвечал текстом вместо того, чтобы править файл. Ошибка
        того же рода, что уже описана у TASK_MARKERS: реплику определяет
        её главный глагол, а он стоит первым.
        """
        assert agent8.looks_like_task(task)

    @pytest.mark.parametrize("question", [
        "где реализован калькулятор?",
        "покажи git log",
        "что изменилось в calc.py",
        "как устроен реестр инструментов",
        "объясни, что делает guard",
    ])
    def test_вопрос_в_начале_остаётся_вопросом(self, question):
        assert not agent8.looks_like_task(question)

    def test_опечатка_человека_ловится(self):
        """«испарвь» стоит в маркерах намеренно: цена промаха несимметрична."""
        assert agent8.looks_like_task("испарвь сложение в calc.py")


class TestHandle:
    """Разбор реплики — и главная развилка внутри него.

    Тесты появились после ошибки, которую нечем было поймать: развилка
    «задача или вопрос» жила прямо в цикле `input()`, при правке цикла
    выпала, и агент отправлял «напиши приложение» специалисту
    по документам. Ошибка молчаливая: ничего не падает, просто делается
    не то. Поэтому разбор вынесен из цикла в `handle()`.
    """

    @pytest.fixture
    def session(self, monkeypatch, tmp_path):
        """Сессия без диалогов: они тянут модель, а разбор её не трогает."""
        made = agent8.Session.__new__(agent8.Session)
        made.team = Team()
        made.conversations = {}
        # Память во временном файле: настоящая лежит рядом с главой
        # и пережила бы прогон тестов, унеся в следующий чужой каталог.
        made.memory = SessionMemory(tmp_path / "session.json")
        monkeypatch.setattr(agent8, "work", lambda task, **kw: f"[конвейер] {task}")
        monkeypatch.setattr(agent8, "_ask", lambda question, s: f"[вопрос] {question}")
        return made

    @pytest.mark.parametrize(
        "text",
        ["напиши приложение hello world", "поправь функцию add в calc.py",
         "сделай модуль корзины с тестами", "добавь функцию деления"],
    )
    def test_задача_уходит_в_конвейер(self, workspace, session, text):
        assert handle(text, session).startswith("[конвейер]")

    @pytest.mark.parametrize(
        "text",
        ["где реализован калькулятор", "что такое эмбеддинг", "покажи git log"],
    )
    def test_вопрос_уходит_специалистам(self, workspace, session, text):
        assert handle(text, session).startswith("[вопрос]")

    def test_спроси_заставляет_считать_реплику_вопросом(self, workspace, session):
        """Иначе «спроси, как написать парсер» ушло бы делать парсер."""
        assert handle("спроси напиши ли ты парсер", session) == "[вопрос] напиши ли ты парсер"

    def test_выход_возвращает_none(self, workspace, session):
        assert handle("выход", session) is None

    def test_пустая_реплика_ничего_не_делает(self, workspace, session):
        assert handle("   ", session) == ""

    def test_помощь(self, workspace, session):
        assert "Просто напишите, что нужно сделать" in handle("помощь", session)

    def test_каталог_показывает_отчёт(self, workspace, session):
        assert str(workspace) in handle("каталог", session)

    def test_смена_каталога(self, workspace, session, tmp_path_factory):
        other = tmp_path_factory.mktemp("другой")
        handle(f"каталог {other}", session)
        assert guard.get_workspace() == other.resolve()

    def test_смена_каталога_сбрасывает_карту(self, workspace, session, tmp_path_factory):
        """Ключи карты — относительные пути, и в новом каталоге они чужие."""
        (workspace / "app.py").write_text("def тут():" + chr(10) + "    return 1" + chr(10),
                                          encoding="utf-8")
        assert "тут" in codemap.names()
        handle(f"каталог {tmp_path_factory.mktemp('другой')}", session)
        assert codemap.names() == []

    def test_несуществующий_каталог_не_меняет_рабочий(self, workspace, session):
        assert "Нет такого каталога" in handle("каталог такогопутинету", session)
        assert guard.get_workspace() == workspace.resolve()

    def test_режимы_переключаются(self, workspace, session):
        handle("сухо", session)
        assert guard.get_policy().dry_run
        handle("молча", session)
        assert guard.get_policy().mode == guard.AUTO and not guard.get_policy().dry_run
        handle("спрашивать", session)
        assert guard.get_policy().mode == guard.ASK

    def test_откат_возвращает_файлы(self, sample, session):
        before = sample.read_text(encoding="utf-8")
        guard.set_policy(mode=guard.AUTO)
        edit_file("sample.py", "return a + b", "return 0")
        assert "восстановлен" in handle("откат", session)
        assert sample.read_text(encoding="utf-8") == before

    def test_откат_когда_нечего(self, workspace, session):
        assert "Менять было нечего" in handle("откат", session)

    def test_окружение(self, workspace, session):
        assert "Окружения проекта нет" in handle("окружение", session)

    def test_маршрут_объясняет_задачу(self, workspace, session):
        assert "конвейер" in handle("маршрут напиши приложение", session)

    def test_план_ничего_не_делает(self, sample, session, monkeypatch):
        monkeypatch.setattr(agent8, "show_plan", lambda task, use_model=None: f"[план:{use_model}] {task}")
        assert handle("план поправь sample.py", session) == "[план:None] поправь sample.py"
        assert handle("план моделью поправь sample.py", session) == "[план:True] поправь sample.py"

    def test_обычная_задача_запоминается(self, workspace, session, monkeypatch):
        """Память бесполезна, если главный путь мимо неё.

        Первая версия прокинула память только в ветку «langgraph …»,
        и вся затея оказалась мертва: «Работаем над файлом» в обычном
        диалоге не наполнялось, а планировщик спрашивал пустоту.
        """
        seen = {}
        monkeypatch.setattr(agent8, "work", lambda task, **kw: seen.update(kw) or "ok")
        handle("напиши приложение hello world", session)
        assert seen.get("memory") is session.memory

    def test_langgraph_идёт_второй_сборкой(self, workspace, session, monkeypatch):
        seen = {}
        monkeypatch.setattr(agent8, "work", lambda task, **kw: seen.update(kw) or "ok")
        handle("langgraph напиши приложение", session)
        assert seen["langgraph"] is True
        assert seen["memory"] is session.memory, "прогон второй сборкой тоже запоминается"

    def test_инструменты_перечисляются(self, workspace, session):
        text = handle("инструменты", session)
        assert "edit_file" in text and "search_code" in text
        assert "get_weather" not in text

    def test_забудь_пересобирает_диалоги(self, workspace, session, monkeypatch):
        monkeypatch.setattr(agent8, "new_conversations", lambda: {"новые": 1})
        assert "очищена" in handle("забудь", session)
        assert session.conversations == {"новые": 1}


class TestCoderSpecialist:
    """Команда Главы 8 — один исполнитель, и в ней нет ничего лишнего."""

    def test_исполнитель_один(self):
        assert agent8.TEAM.names() == ["код"]

    def test_у_него_все_инструменты_главы(self):
        assert set(agent8.CODER.tools) >= set(FS_TOOLS + RUN_TOOLS + GIT_TOOLS + ENV_TOOLS)

    def test_и_поиск_по_коду_из_главы_5(self):
        """Нужен ровно для того, о чём глава: найти, ГДЕ баг."""
        assert {"search_code", "find_symbol", "project_map"} <= set(agent8.CODER.tools)

    @pytest.mark.parametrize("tool", ["get_weather", "calculator", "read_file",
                                      "remember", "recall", "search_docs"])
    def test_чужих_инструментов_нет(self, tool):
        """`enum` схемы — это список того, что агент может назвать.

        Каждое лишнее имя в нём — ошибка, которую агент способен
        совершить. Погода кодинг-агенту не нужна, а назвать её он смог бы.
        """
        assert tool not in agent8.CODER.tools

    def test_все_его_инструменты_есть_в_реестре(self):
        assert agent8.CODER.unknown_tools() == []

    def test_схема_ответа_сужена_до_его_инструментов(self):
        names = agent8.CODER.response_schema()["properties"]["name"]["enum"]
        assert set(names) == set(agent8.CODER.tools)

    def test_глобальный_реестр_специалистов_не_тронут(self):
        """Глава 8 собирает команду объектом, а не декоратором.

        Декоратор `@specialist` положил бы исполнителя в общий реестр
        Главы 7, и её пятеро получили бы шестого. Здесь команда своя
        и живёт только в этой главе.
        """
        assert "код" in SPECIALISTS
        assert SPECIALISTS["код"] is not agent8.CODER

    def test_в_контексте_лежит_состояние_каталога(self, workspace):
        text, found = agent8.retrieve_workspace("поправь код", 500)
        assert found
        assert str(workspace) in text
        assert "Режим:" in text
        assert "Окружения проекта нет" in text

    def test_контекст_называет_уже_изменённые_файлы(self, sample):
        edit_file("sample.py", "return a + b", "return 0")
        text, _ = agent8.retrieve_workspace("что дальше", 500)
        assert "sample.py" in text

    def test_граф_главы_собирается(self):
        assert agent8.build_graph8().validate() == []

    def test_в_графе_заменён_ровно_один_узел(self):
        """Остальные три — те же объекты, что в Главе 7."""
        mine = agent8.build_graph8()
        theirs = chapter7_agent.build_graph()
        assert mine.nodes["route"] is not theirs.nodes["route"]
        for name in ("retrieve", "handoff", "generate"):
            assert mine.nodes[name] is theirs.nodes[name]

    def test_маршрутизации_нет(self):
        """Выбирать не между кем: узел просто называет единственного."""
        state = State(user_input="что угодно")
        agent8.node_route_single(state)
        assert state.agent == "код"
        assert state.tried == ["код"]

    def test_передавать_реплику_некому(self):
        """Откат Главы 7 на команде из одного не срабатывает."""
        assert agent8.TEAM.next_untried(["код"]) == ""


class TestAgentReport:
    def test_отчёт_о_каталоге_называет_главное(self, workspace):
        text = agent8.workspace_report()
        assert str(workspace) in text
        assert "Режим подтверждения" in text
        assert "Модель для кода" in text

    def test_отчёт_говорит_что_белого_списка_нет(self, workspace):
        assert "ЛЮБЫЕ (белый список выключен)" in agent8.workspace_report()

    def test_отчёт_показывает_узкий_список_если_он_включён(self, workspace):
        guard.set_policy(allowed=guard.NARROW_ALLOWED)
        assert "python, pytest" in agent8.workspace_report()

    def test_отчёт_видит_сухой_прогон(self, workspace):
        guard.set_policy(dry_run=True)
        assert "сухой прогон" in agent8.workspace_report()

    def test_план_показывается_без_правок(self, project, monkeypatch):
        monkeypatch.setattr(planner_module, "request_model", lambda *a, **k: json.dumps(
            {"steps": [{"action": "edit", "target": "calc_mod.py", "detail": "поправить"},
                       {"action": "test", "target": "", "detail": "прогнать"}]}))
        monkeypatch.setattr(planner_module, "PLANNER", "model")
        before = (project / "calc_mod.py").read_text(encoding="utf-8")
        text = agent8.show_plan("почини сложение")
        assert "edit calc_mod.py" in text
        assert (project / "calc_mod.py").read_text(encoding="utf-8") == before

    def test_отчёт_работы_показывает_маршрут_и_план(self, project, monkeypatch, fix_plan):
        monkeypatch.setattr(pipeline, "request_model",
                            fake_model([edit_answer("calc_mod.py", 2, 2, "    return a + b")]))
        monkeypatch.setattr(pipeline, "execute", fake_run(green=True))
        monkeypatch.setattr(agent8, "run_pipeline",
                            lambda task, tests="": pipeline.run_pipeline(task, tests=tests, plan=fix_plan))
        text = agent8.work("почини сложение")
        assert "Маршрут прогона" in text
        assert "вернуть a + b" in text


# ====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ: НУЖНА ЗАПУЩЕННАЯ OLLAMA
# ====================================================================
# Запуск: python -m pytest chapter8/tests.py -m integration -v -s

BROKEN = "def add(a, b):\n    return a - b\n"
BROKEN_TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"


@pytest.fixture(scope="session")
def warm_model():
    """Прогревает модель до первого замера времени.

    Нужна не для скорости, а для честности результата. Ollama держит
    в памяти одну модель, и если перед прогоном там лежал эмбеддер,
    первый запрос платит за загрузку — на слабой машине это уезжало
    за таймаут в 120 с, планировщик отдавал запасной план, и тест падал
    по причине, к планировщику отношения не имеющей.
    """
    from chapter1.agent import preload_model

    if not preload_model():
        pytest.skip("модель не загрузилась — запущена ли Ollama?")
    return True


@pytest.fixture
def broken_project(workspace, warm_model):
    """Проект с одной сломанной функцией и тестом, который её ловит."""
    (workspace / "calc.py").write_text(BROKEN, encoding="utf-8")
    (workspace / "test_calc.py").write_text(BROKEN_TEST, encoding="utf-8")
    return workspace


@pytest.mark.integration
class TestRealPlanner:
    def test_задача_с_нуля_планируется_кодом_с_именем_от_модели(self, workspace, warm_model):
        """У модели спрашивают ОДНО имя файла, а не план целиком.

        Три живых прогона показали, что план от модели теряет требования
        задачи, а на квадратном уравнении разваливает одну маленькую
        программу на четыре файла — по файлу на функцию. Кодом строится
        всё, что выводится из задачи однозначно.
        """
        task = "напиши приложение на python, которое выводит hello world в консоль"
        plan = make_plan(task)
        print(f"\n{render_plan(plan)}")

        assert plan.source == FROM_FALLBACK, "план строит код"
        assert [s.action for s in plan.steps] == ["create", "test"]
        assert plan.steps[0].target.endswith(".py"), f"негодное имя: {plan.steps[0].target!r}"
        assert plan.steps[0].detail == task, "формулировка человека доезжает целиком"

    def test_план_на_несколько_вещей_составляет_модель(self, workspace, warm_model):
        plan = make_plan("напиши app.py, а также набросай README.md")
        print(f"\n{render_plan(plan)}")
        assert plan.source == FROM_MODEL, f"план запасной: {plan.problems}"

    def test_правка_названного_файла_идёт_без_модели(self, broken_project):
        plan = make_plan("В файле calc.py функция add вычитает вместо сложения")
        assert plan.source == FROM_FALLBACK
        assert plan.steps[0].target == "calc.py"

    def test_каждый_путь_в_плане_либо_существует_либо_назван_претензией(self, broken_project):
        """Проверяется механизм, а не удача модели.

        Насколько хорош план — вопрос замера, и на него отвечает
        TestPlannerModels числом. Тест утверждает более скромное
        и более полезное: план с неверным путём не проходит молча.
        """
        plan = make_plan("Почини функцию add в файле calc.py", use_model=True)
        problems = validate_plan(plan)
        print(f"\n{render_plan(plan, problems)}")

        for number, step in enumerate(plan.steps, start=1):
            if step.action not in ("read", "edit"):
                continue
            exists = bool(step.target) and (broken_project / step.target).is_file()
            flagged = any(f"шаг {number}:" in claim for claim in problems)
            assert exists or flagged, f"шаг {number} с путём {step.target!r} прошёл проверку молча"

    def test_описание_отлипает_от_пути(self, broken_project):
        """3B склеивает путь с объяснением в одном поле. Разбор их разделяет."""
        plan = make_plan("Почини функцию add в файле calc.py", use_model=True)
        for step in plan.steps:
            if step.action in ("read", "edit", "create"):
                assert " " not in step.target, f"в пути осталось описание: {step.target!r}"


@pytest.mark.integration
class TestRealScratch:
    """Задача с нуля — то, ради чего глава переписывалась.

    Тесты здесь проверяют МЕХАНИКУ, а не удачу модели. Довести задачу
    до зелёной проверки 3B удаётся не всегда — это число, и его даёт
    замер (TestCoderModels: 2-3 из 5). Требовать успеха в тесте значило
    бы мерить везение и падать через раз, а на исправление это не
    указывало бы ничем.

    Утверждается более скромное и более полезное: агент СДЕЛАЛ ход —
    составил план, написал файлы, проверил, — а если не вышло, честно
    сказал об этом и не оставил после себя половины работы.
    """

    def test_агент_доводит_задачу_до_конца_или_честно_откатывает(self, workspace, warm_model):
        state = run_pipeline("напиши приложение на python, которое выводит hello world в консоль")
        print(f"\nМаршрут: {state.trace()}")
        print(state.answer)

        assert state.extra["plan"]["steps"], "план пуст — планировщик не сработал"
        assert any(line.startswith("1.") for line in state.extra["log"]), "не выполнено ни одного шага"

        if state.extra.get("tests_green"):
            written = sorted(p.name for p in workspace.glob("*.py"))
            assert written, "проверка зелёная, а файлов нет"
            body = "\n".join(p.read_text(encoding="utf-8").lower() for p in workspace.glob("*.py"))
            assert "hello" in body
            assert state.answer.startswith("Готово")
        else:
            # Не вышло — значит откачено целиком. Наполовину сделанная
            # работа хуже несделанной: она выглядит как сделанная.
            assert list(workspace.glob("*.py")) == [], "провал не откатил файлы"
            assert "Готово" not in state.answer

    def test_отказ_человека_оставляет_каталог_пустым(self, workspace, warm_model):
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: False)
        state = run_pipeline("напиши приложение, которое выводит hello world")
        assert state.steps == ["plan", "confirm"]
        assert list(workspace.glob("*.py")) == []


@pytest.mark.integration
class TestRealFix:
    def test_агент_чинит_функцию_и_доводит_тесты_до_зелёных(self, broken_project):
        state = run_pipeline(
            "В файле calc.py функция add вычитает вместо сложения. Верни сложение.",
            plan=Plan("почини add", [Step("edit", "calc.py", "заменить a - b на a + b"),
                                     Step("test", "", "прогнать тесты")]),
        )
        print(f"\nМаршрут: {state.trace()}")
        print(f"Форма правки: {state.extra.get('edit_form')}, починок: {state.extra.get('attempt')}")
        print(f"Итог: {state.answer}")
        assert state.extra.get("tests_green"), state.extra.get("failure")
        assert "a + b" in (broken_project / "calc.py").read_text(encoding="utf-8")

    def test_провал_откатывает_файлы(self, broken_project):
        """Задача без решения: агент должен вернуть файл, а не оставить огрызок."""
        before = (broken_project / "calc.py").read_text(encoding="utf-8")
        (broken_project / "test_calc.py").write_text(
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 5\n", encoding="utf-8"
        )
        state = run_pipeline(
            "Сделай так, чтобы add(2, 2) возвращало 4, не меняя тест.",
            plan=Plan("почини", [Step("edit", "calc.py", "исправить сложение"), Step("test", "", "тесты")]),
        )
        if not state.extra.get("tests_green"):
            assert (broken_project / "calc.py").read_text(encoding="utf-8") == before


@pytest.mark.integration
class TestRealDeps:
    def test_недостающий_пакет_виден_агенту(self, workspace, warm_model):
        """Зависимости вычисляются из кода, а не спрашиваются у модели."""
        (workspace / "app.py").write_text("import такогопакетаточнонет\n", encoding="utf-8")
        asked = []
        guard.set_policy(mode=guard.ASK, confirm=lambda a, d: asked.append(d) or False)

        state = State(user_input="з")
        state.extra.update({"plan": Plan("з", []).to_dict(), "touched": ["app.py"], "log": []})
        pipeline.node_deps(state)

        assert state.extra["missing_packages"] == ["такогопакетаточнонет"]
        assert asked, "установка обязана спрашивать человека"
        assert "из сети" in asked[0]


# ====================================================================
# ЗАМЕРЫ: ЧИСЛА, КОТОРЫЕ ПОПАДАЮТ В ТЕКСТ ГЛАВЫ
# ====================================================================
# Запуск: python -m pytest chapter8/tests.py -m slow -v -s

# Пять задач ПРАВКИ: файл есть, ошибка в одной строке.
FIX_TASKS = [
    (
        "сложение",
        "def add(a, b):\n    return a - b\n",
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n",
        "В файле calc.py функция add вычитает вместо сложения. Верни сложение.",
    ),
    (
        "умножение",
        "def mul(a, b):\n    return a + b\n",
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n",
        "В файле calc.py функция mul складывает вместо умножения. Верни умножение.",
    ),
    (
        "сравнение",
        "def is_adult(age):\n    return age > 18\n",
        "from calc import is_adult\n\n\ndef test_adult():\n    assert is_adult(18) is True\n",
        "В файле calc.py функция is_adult должна возвращать True и для ровно 18 лет.",
    ),
    (
        "пустой список",
        "def first(items):\n    return items[0]\n",
        "from calc import first\n\n\ndef test_first():\n    assert first([]) is None\n",
        "В файле calc.py функция first падает на пустом списке. Пусть возвращает None.",
    ),
    (
        "регистр",
        "def shout(text):\n    return text\n",
        "from calc import shout\n\n\ndef test_shout():\n    assert shout('да') == 'ДА'\n",
        "В файле calc.py функция shout должна возвращать текст заглавными буквами.",
    ),
]

# Пять задач С НУЛЯ: каталог пустой, всё придумывает агент — сколько
# файлов завести, как их назвать, что в них написать и чем проверить.
# Именно этот набор отличает кодинг-агента от чинилки кода, и именно
# на нём видно, что 3B хватает не на всё.
SCRATCH_TASKS = [
    ("привет", "напиши приложение на python, которое выводит Hello, World! в консоль"),
    ("сложение", "напиши модуль с функцией add(a, b), которая складывает два числа, и тест к нему"),
    ("длина строк", "напиши функцию longest(words), возвращающую самое длинное слово списка, и тест"),
    ("чётные", "напиши функцию evens(numbers), которая оставляет только чётные числа, и тест к ней"),
    ("счётчик слов", "напиши функцию count_words(text), считающую слова в строке, и тест к ней"),
]


def make_task_project(root, source, test_source, long_file=False):
    """Кладёт одну задачу правки во временный каталог."""
    if long_file:
        source, test_source = pad(source, test_source)
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text(source, encoding="utf-8")
    (root / "test_calc.py").write_text(test_source, encoding="utf-8")
    guard.set_policy(root=root, mode=guard.AUTO, dry_run=False)
    guard.forget_changes()
    return root


def rounds(times, tasks):
    """Задачи, повторённые нужное число раз, парой «номер прогона, задача».

    Один прогон замера — ещё не замер: разброс между прогонами в этой
    главе не раз оказывался больше измеряемой разницы. Отдельная функция
    нужна, чтобы повтор не добавлял в замеры лишний уровень отступа
    и выглядел во всех одинаково.
    """
    return [(attempt, task) for attempt in range(times) for task in tasks]


def make_empty_project(root):
    """Пустой каталог под задачу с нуля."""
    root.mkdir(parents=True, exist_ok=True)
    guard.set_policy(root=root, mode=guard.AUTO, dry_run=False)
    guard.forget_changes()
    return root


def folder_name(text):
    """Имя каталога из имени модели: двоеточие на Windows — разделитель диска."""
    return text.replace(":", "-").replace(" ", "_").replace("(", "").replace(")", "")


# Посторонние функции, которыми задача обкладывается во второй половине
# замера форм. Каждая проверяется своим утверждением в тесте — и это
# главное в них: код, потерянный при перезаписи файла целиком, обязан
# ронять тесты. Замер, в котором потеря чужого кода ничего не ломает,
# показал бы все три формы одинаково хорошими, что и случилось
# в первой версии.
FILLER = [
    ("half", "def half(x):\n    return x / 2\n", "assert half(4) == 2"),
    ("twice", "def twice(x):\n    return x * 2\n", "assert twice(3) == 6"),
    ("clamp", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n", "assert clamp(9, 0, 5) == 5"),
    ("initials", "def initials(name):\n    return name[:1].upper()\n", "assert initials('анна') == 'А'"),
    ("total", "def total(items):\n    return sum(items)\n", "assert total([1, 2]) == 3"),
    ("empty", "def empty(items):\n    return not items\n", "assert empty([]) is True"),
    ("last", "def last(items):\n    return items[-1] if items else None\n", "assert last([1, 2]) == 2"),
    ("join", "def join(parts):\n    return ', '.join(parts)\n", "assert join(['a', 'b']) == 'a, b'"),
]


def pad(source, test_source):
    """Обкладывает задачу посторонним кодом — тем, который трогать нельзя.

    Половина функций ставится до задачи, половина после: если бы все
    восемь шли перед ней, номера строк у задачи были бы одинаковыми
    во всех пяти случаях, и форма `lines` мерилась бы на одном адресе.
    """
    head = "\n".join(body for _, body, _ in FILLER[:4])
    tail = "\n".join(body for _, body, _ in FILLER[4:])
    names = ", ".join(name for name, _, _ in FILLER)
    checks = "\n".join(f"    {check}" for _, _, check in FILLER)

    padded = f"{head}\n{source}\n{tail}"
    imports = test_source.split("\n")[0].rstrip()
    rest = "\n".join(test_source.split("\n")[1:])
    return padded, (
        f"{imports}\nfrom calc import {names}\n{rest}\n\ndef test_остальное_на_месте():\n{checks}\n"
    )


def ask_for_edit(task, path, forms=None):
    """Один запрос к модели за правкой в заданной форме. Возвращает отчёт."""
    text = read_lines(path, "1", "120")
    messages = [
        {"role": "system", "content": pipeline.EDIT_RULES},
        {"role": "user", "content": f"Задача: {task}\n\nФайл {path}:\n{text}\n"},
    ]
    try:
        raw = request_model(messages, response_format=edit_schema(forms))
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return f"Правка не применена. Модель: {exc}"
    return pipeline.apply_edit(data, default_path=path)


@pytest.mark.slow
class TestFormAccuracy:
    """Замер 1: какую из трёх форм правки 3B выдаёт применимой чаще.

    Форма задаётся схемой по одной, а не оставляется на выбор модели:
    иначе замер показал бы её предпочтения, а не пригодность форм.

    Мерится дважды — на файле в две строки и на нём же, обложенном
    восемью посторонними функциями. Одного короткого файла мало:
    на нём все три формы справляются, и различить их нечем. Разница
    между формами — это разница в том, что происходит с ЧУЖИМ кодом,
    а чужого кода в двухстрочном файле нет.
    """

    def test_три_формы_на_одном_наборе(self, tmp_path, warm_model):
        rounds = 2
        report = {}

        for long_file in (False, True):
            for form in EDIT_FORMS:
                applied = green = 0
                spent = 0.0
                for attempt in range(rounds):
                    for name, source, test_source, task in FIX_TASKS:
                        size = "длинный" if long_file else "короткий"
                        root = make_task_project(
                            tmp_path / f"{size}-{form}-{attempt}-{name}", source, test_source, long_file
                        )
                        started_at = time.monotonic()
                        ask_for_edit(task, "calc.py", forms=(form,))
                        spent += time.monotonic() - started_at
                        # «Применилась» считается по журналу изменений,
                        # а не по тексту ответа. Первая версия замера
                        # искала в ответе слова «не применена» и завышала
                        # результат: правка, отменённая проверкой на
                        # пропавшие определения, отвечает другими словами.
                        if not guard.change_count():
                            continue
                        applied += 1
                        if suite_passed(execute("python -m pytest -q --no-header", timeout=120)):
                            green += 1
                        guard.set_workspace(root)
                total = rounds * len(FIX_TASKS)
                report[(long_file, form)] = (applied, green, total, spent)

        print("\n\nЗАМЕР 1: три формы правки на одном наборе задач")
        print(f"Модель: {base.MODEL}, задач: {len(FIX_TASKS)}, прогонов: {rounds}")
        for long_file in (False, True):
            where = "файл из 10 функций" if long_file else "файл из одной функции"
            print(f"\n{where}:")
            print(f"{'форма':<10}{'применилась':>14}{'тесты зелёные':>16}{'секунд':>10}")
            for form in EDIT_FORMS:
                applied, green, total, spent = report[(long_file, form)]
                print(f"{form:<10}{applied:>10}/{total:<3}{green:>12}/{total:<3}{spent:>10.1f}")

        assert sum(applied for applied, _, _, _ in report.values()) > 0, \
            "ни одна форма не дала ни одной применимой правки — проверьте, запущена ли Ollama"


def bare_plan(task: str) -> Plan:  # noqa: ARG001
    """Нижняя граница замера планировщика: «правь этот файл, потом проверь».

    Это и есть «работа вообще без плана» из программы главы. Планировать
    здесь нечего: файл назван прямо, шагов ровно два, модель к составлению
    плана не привлекается.
    """
    return Plan(task, [Step("edit", "calc.py", "исправить по задаче"), Step("test", "", "прогнать тесты")])


@pytest.mark.slow
class TestPlannerModels:
    """Замер 2: план от модели против плана из кода против работы без плана.

    Обязательный замер главы. Он отвечает на вопрос, ради которого
    планировщик вынесен в отдельный модуль с отдельной моделью:
    окупается ли планирование, и окупается ли вторая модель под него.

    Мерится ТОЛЬКО на задачах правки, и это ограничение принципиальное.
    На задачах с нуля сравнивать не с чем: плана из кода там не бывает,
    придумать имена файлов эвристикой нельзя. Переносить вывод «код
    выигрывает» на работу с нуля поэтому нельзя — там у кода нет ответа,
    а не плохой ответ.
    """

    def test_план_окупается_или_нет(self, tmp_path, warm_model):
        sources = [("без плана (файл назван)", bare_plan), ("план из кода", fallback_plan)]
        for model_name in [base.MODEL, os.environ.get("AGENT_PLANNER_MODEL", "")]:
            if model_name and model_name not in [n for n, _ in sources]:
                sources.append((model_name, lambda task, m=model_name: make_plan(task, model=m)))

        report = []
        for label, build in sources:
            green = with_edit = clean = 0
            spent = 0.0
            for name, source, test_source, task in FIX_TASKS:
                make_task_project(tmp_path / f"{folder_name(label)}-{name}", source, test_source)
                started_at = time.monotonic()
                plan = build(task)
                # Две характеристики самого плана, до его исполнения.
                # Они объясняют итог: план без шага правки не может
                # довести тесты до зелёных, сколько его ни выполняй.
                if any(step.action in ("edit", "create") for step in plan.steps):
                    with_edit += 1
                if not validate_plan(plan):
                    clean += 1
                state = run_pipeline(task, plan=plan)
                spent += time.monotonic() - started_at
                if state.extra.get("tests_green"):
                    green += 1
            report.append((label, green, with_edit, clean, spent))

        print("\n\nЗАМЕР 2: планировщик на задачах ПРАВКИ")
        print(f"Задач: {len(FIX_TASKS)}, модель кода: {pipeline.coder_model()}")
        print(f"{'планировщик':<28}{'зелёные':>10}{'есть правка':>14}{'без претензий':>16}{'секунд':>10}")
        for label, green, with_edit, clean, spent in report:
            print(f"{label:<28}{green:>7}/{len(FIX_TASKS):<2}"
                  f"{with_edit:>11}/{len(FIX_TASKS):<2}{clean:>13}/{len(FIX_TASKS):<2}{spent:>10.1f}")

        assert report, "замер не собрал ни одной строки"


def prompt_tokens(text: str, model: str) -> int:
    """Во сколько токенов модель считает этот текст.

    Числа берутся у самой Ollama (`prompt_eval_count`), а не оцениваются
    формулой «символы делить на четыре»: у русского и английского
    текста разное число символов на токен, и оценка по символам как раз
    в этом месте и врёт.
    """
    response = requests.post(
        f"{base.OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            "options": {"num_predict": 1},
        },
        timeout=120,
    )
    response.raise_for_status()
    return int(response.json().get("prompt_eval_count", 0))


def code_files(plan: Plan) -> int:
    """Сколько файлов с кодом заводит план. Ровно тот дефект, что видели живьём."""
    return len({step.target for step in plan.steps
                if step.action == "create" and step.target.endswith(".py")})


@pytest.mark.slow
class TestPlannerLanguage:
    """Замер 7: на каком языке разговаривать с планировщиком.

    Гипотеза ровно такая же проверяемая, как «скелет вместо целого
    файла» из замера 6, и звучит так: модели на 3B русский промпт
    обходится дороже в токенах и хуже ложится на то, чему её учили,
    поэтому английский промпт даст более годные планы.

    Проверять её отдельным замером пришлось потому, что «дороже
    в токенах» — не то же самое, что «хуже планы». Первое видно
    из `prompt_eval_count` сразу; второе может не сдвинуться вовсе.

    Мерится на обоих наборах задач, и на каждом — своё. На правках
    важно, чтобы план не выдумал путь и содержал сам шаг правки.
    На задачах с нуля важно другое: сколько файлов план заводит.
    Именно там модель разваливала одну маленькую программу на четыре
    файла, по файлу на функцию, — и правило «одна программа — один
    файл» написано против этого. Если перевод помогает, он должен
    помочь прежде всего здесь.

    Язык ЗАДАЧИ при этом не трогается: она идёт в промпт как есть,
    словами человека. Переводить её было бы уже не заменой языка
    промпта, а вмешательством в постановку.
    """

    LANGUAGES = ("ru", "en")

    def test_русский_промпт_против_английского(self, tmp_path, warm_model):
        installed = set(base.list_installed_models())
        models = [m for m in (base.MODEL, os.environ.get("AGENT_PLANNER_MODEL", "")) if m and m in installed]
        if not models:
            pytest.skip("планировать нечем: ни одна модель-кандидат не установлена")

        report = []
        sizes = {}
        for model in models:
            for language in self.LANGUAGES:
                clean = with_work = own_path = 0
                scratch_clean = one_file = 0
                spent = 0.0

                for name, source, test_source, task in FIX_TASKS:
                    make_task_project(tmp_path / f"{folder_name(model)}-{language}-fix-{name}",
                                      source, test_source)
                    started_at = time.monotonic()
                    plan = make_plan(task, model=model, use_model=True, language=language)
                    spent += time.monotonic() - started_at
                    if not validate_plan(plan):
                        clean += 1
                    if any(step.action in ("edit", "create") for step in plan.steps):
                        with_work += 1
                    # Выдуманный путь — самая частая беда плана от модели:
                    # файл в проекте назван прямо, а в плане стоит другой.
                    targets = [step.target for step in plan.steps
                               if step.action in ("read", "edit") and step.target]
                    if targets and all(t in ("calc.py", "test_calc.py") for t in targets):
                        own_path += 1

                for name, task in SCRATCH_TASKS:
                    make_empty_project(tmp_path / f"{folder_name(model)}-{language}-new-{name}")
                    started_at = time.monotonic()
                    plan = make_plan(task, model=model, use_model=True, language=language)
                    spent += time.monotonic() - started_at
                    if not validate_plan(plan):
                        scratch_clean += 1
                    if code_files(plan) <= 2:
                        one_file += 1

                report.append((model, language, clean, with_work, own_path,
                               scratch_clean, one_file, spent))
            for language in self.LANGUAGES:
                text = build_planner_prompt("В файле calc.py функция add вычитает вместо сложения.",
                                            files=["calc.py", "test_calc.py"], language=language)
                sizes[(model, language)] = prompt_tokens(text, model)

        print("\n\nЗАМЕР 7: язык промпта планировщика")
        print(f"Задач правки: {len(FIX_TASKS)}, задач с нуля: {len(SCRATCH_TASKS)}")
        print(f"{'модель':<34}{'язык':>6}{'токенов':>9}{'правка: без претензий':>23}"
              f"{'есть шаг':>10}{'путь из проекта':>17}")
        for model, language, clean, with_work, own_path, _, _, _ in report:
            print(f"{model:<34}{language:>6}{sizes[(model, language)]:>9}"
                  f"{clean:>20}/{len(FIX_TASKS):<2}{with_work:>7}/{len(FIX_TASKS):<2}"
                  f"{own_path:>14}/{len(FIX_TASKS):<2}")
        print(f"\n{'модель':<34}{'язык':>6}{'с нуля: без претензий':>23}"
              f"{'не больше 2 файлов':>20}{'секунд':>9}")
        for model, language, _, _, _, scratch_clean, one_file, spent in report:
            print(f"{model:<34}{language:>6}{scratch_clean:>20}/{len(SCRATCH_TASKS):<2}"
                  f"{one_file:>17}/{len(SCRATCH_TASKS):<2}{spent:>9.1f}")

        assert report, "замер не собрал ни одной строки"


# Модели, которые имеет смысл сравнивать на написании кода. Список —
# кандидаты, а не требование: замер берёт из него только те, что
# действительно установлены у читателя.
# Пара «qwen2_5coder3b_q5 против qwen2.5-coder:3b» стоит здесь не для
# полноты. Это одна и та же архитектура одного размера, и различаются
# они ровно тем, на чём дообучены: первая собрана из базового
# `Qwen/Qwen2.5-Coder-3B`, вторая — Instruct-вариант из библиотеки
# Ollama. Всё остальное — токенизатор, шаблон чата, размер — совпадает,
# так что разница в числах говорит именно про дообучение на инструкции.
CODER_CANDIDATES = (
    "qwen2.5:3b",
    "qwen2_5coder3b_q5:latest",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
)


@pytest.mark.slow
class TestCoderModels:
    """Замер 3: кто лучше пишет код — обычная модель или coder-версия (8.5).

    Мерится на ОБОИХ видах задач, и это важно. Правка одной строки
    и написание модуля с тестами — разная работа, и модель, выигравшая
    в первом, не обязана выигрывать во втором.

    Прогонов ДВА, и это не перестраховка. Первая версия делала один,
    и её вывод не пережил повторения: на пяти задачах с нуля базовая
    сборка дала 3/5, а Instruct-вариант того же размера — 5/5, и разница
    выглядела решающей. Второй прогон дал 5/5 и 4/5, то есть обратный
    порядок. Замер, у которого каждый прогон свой ответ, — ещё не замер;
    пять задач для такого утверждения просто мало.

    Считается доля прогонов, доведённых до зелёной проверки, и время.
    Время здесь не второстепенно: 7B на 6 ГБ вытесняет из памяти
    всё остальное, и каждое чередование с эмбеддером стоит загрузки.
    """

    ROUNDS = 2

    def test_модели_на_обоих_видах_задач(self, tmp_path, warm_model):
        installed = set(base.list_installed_models())
        models = [m for m in CODER_CANDIDATES if m in installed]
        if not models:
            pytest.skip(f"ни одна из моделей {CODER_CANDIDATES} не установлена")

        report = []
        for model in models:
            fixed = scratch = 0
            fixed_spent = scratch_spent = 0.0
            with using_model(model):
                for attempt in range(self.ROUNDS):
                    for name, source, test_source, task in FIX_TASKS:
                        make_task_project(
                            tmp_path / f"{folder_name(model)}-fix{attempt}-{name}", source, test_source
                        )
                        started_at = time.monotonic()
                        state = run_pipeline(task, plan=bare_plan(task))
                        fixed_spent += time.monotonic() - started_at
                        fixed += bool(state.extra.get("tests_green"))

                    for name, task in SCRATCH_TASKS:
                        make_empty_project(tmp_path / f"{folder_name(model)}-new{attempt}-{name}")
                        started_at = time.monotonic()
                        state = run_pipeline(task)
                        scratch_spent += time.monotonic() - started_at
                        scratch += bool(state.extra.get("tests_green"))
            report.append((model, fixed, fixed_spent, scratch, scratch_spent))

        fix_total = len(FIX_TASKS) * self.ROUNDS
        new_total = len(SCRATCH_TASKS) * self.ROUNDS
        print("\n\nЗАМЕР 3: модели на написании кода")
        print(f"Задач правки: {len(FIX_TASKS)}, задач с нуля: {len(SCRATCH_TASKS)}, "
              f"прогонов каждой: {self.ROUNDS}")
        print(f"{'модель':<26}{'правка':>11}{'секунд':>9}{'с нуля':>11}{'секунд':>9}")
        for model, fixed, fixed_spent, scratch, scratch_spent in report:
            print(f"{model:<26}{fixed:>8}/{fix_total:<2}{fixed_spent:>9.0f}"
                  f"{scratch:>8}/{new_total:<2}{scratch_spent:>9.0f}")

        assert report, "замер не собрал ни одной строки"


PLACE_MODULE = (
    "import math\n"
    "\n\n"
    "def get_roots(a, b, c):\n"
    "    d = b*b - 4*a*c\n"
    "    if d < 0:\n"
    "        return None\n"
    "    return ((-b + math.sqrt(d))/(2*a), (-b - math.sqrt(d))/(2*a))\n"
    "\n\n"
    "def main():\n"
    "    a = float(input('a: '))\n"
    "    b = float(input('b: '))\n"
    "    c = float(input('c: '))\n"
    "    print(get_roots(a, b, c))\n"
    "    input('Enter')\n"
    "\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)

# Задачи, где нужная функция В ПРОЕКТЕ ЕСТЬ. Ни одна не называет её
# по имени: если бы называла, место нашлось бы текстовым поиском
# и карта была бы не нужна.
PLACE_TASKS = [
    ("производная", "сделай чтобы приложение выводило еще и производную", "main"),
    ("ввод", "программа не спрашивает коэффициент c", "main"),
    ("дискриминант", "неверно считается дискриминант", "get_roots"),
    ("ожидание", "окно закрывается сразу после вывода ответа", "main"),
]

# Задачи, где менять нечего: нужна новая функция или дело вообще
# не в коде. Правильный ответ — «ни одна не подходит».
NO_PLACE_TASKS = [
    ("сохранение", "добавь функцию сохранения результата в файл"),
    ("README", "исправь опечатку в README"),
]


@pytest.mark.slow
class TestPlaceChoice:
    """Замер 8: умеет ли модель выбрать МЕСТО правки по карте функций.

    Вопрос замера: заменяет ли выбор из списка текстовый поиск. Поиск
    находит место, только если человек процитировал свой код; стоит ему
    сказать по-человечески — «сделай чтобы приложение выводило ещё
    и производную», — и искать нечего. Карта даёт другой способ:
    показать список функций с назначением каждой и попросить выбрать
    одну, а `enum` не даст назвать несуществующую.

    Мерятся ДВЕ разные способности, и это главное в замере:

      * выбрать нужную функцию, когда она есть;
      * сказать «ни одна не подходит», когда её нет.

    Смешивать их в одно число нельзя. Первая — выбор из списка, вторая —
    отказ от выбора, и модели на 3B даются они совершенно по-разному.
    От второй зависит, можно ли доверить модели решение «правка или
    новый файл»; от первой — можно ли вообще заменить поиск картой.
    """

    ROUNDS = 2

    def test_выбор_места_на_моделях(self, tmp_path, warm_model, monkeypatch):
        installed = set(base.list_installed_models())
        models = [m for m in CODER_CANDIDATES if m in installed]
        if not models:
            pytest.skip(f"ни одна из моделей {CODER_CANDIDATES} не установлена")

        root = tmp_path / "проект"
        root.mkdir()
        (root / "quadratic.py").write_text(PLACE_MODULE, encoding="utf-8")
        monkeypatch.setenv("AGENT_CODEMAP_FILE", str(tmp_path / "codemap.json"))
        guard.set_policy(root=root, mode=guard.AUTO, dry_run=False)

        report = []
        for model in models:
            hit = refused = 0
            spent = 0.0
            with using_model(model):
                for _, (_, task, expected) in rounds(self.ROUNDS, PLACE_TASKS):
                    codemap.forget_cache()
                    started_at = time.monotonic()
                    place = codemap.choose(task)
                    spent += time.monotonic() - started_at
                    hit += bool(place and place.name == expected)
                for _, (_, task) in rounds(self.ROUNDS, NO_PLACE_TASKS):
                    codemap.forget_cache()
                    started_at = time.monotonic()
                    place = codemap.choose(task)
                    spent += time.monotonic() - started_at
                    refused += place is None
            report.append((model, hit, refused, spent))

        found_total = len(PLACE_TASKS) * self.ROUNDS
        none_total = len(NO_PLACE_TASKS) * self.ROUNDS
        print("\n\nЗАМЕР 8: выбор места правки по карте функций")
        print(f"Задач с местом: {len(PLACE_TASKS)}, без места: {len(NO_PLACE_TASKS)}, "
              f"прогонов каждой: {self.ROUNDS}")
        print(f"{'модель':<26}{'нашёл нужную':>15}{'сказал «ни одна»':>20}{'секунд':>10}")
        for model, hit, refused, spent in report:
            print(f"{model:<26}{hit:>12}/{found_total:<2}"
                  f"{refused:>17}/{none_total:<2}{spent:>10.1f}")

        assert report, "замер не собрал ни одной строки"


GOOD_DERIVATIVE = (
    "def get_roots(a, b, c):\n"
    "    return (a, b, c)\n"
    "\n\n"
    "def derivative(a, b):\n"
    "    return 2*a\n"
    "\n\n"
    "def main():\n"
    "    a = float(input('a: '))\n"
    "    print('Корни:', get_roots(a, 1, 1))\n"
    "    print('Производная:', derivative(a, 1))\n"
)

# Функция есть, но её никто не зовёт — самая частая беда живых прогонов.
DEAD_DERIVATIVE = GOOD_DERIVATIVE.replace(
    "    print('Производная:', derivative(a, 1))\n", "")

# Вызов не сходится с определением: TypeError при запуске. Ни одна
# механическая проверка главы этого не видит — файл разбирается,
# импортируется, имена определены.
WRONG_ARITY = GOOD_DERIVATIVE.replace(
    "print('Производная:', derivative(a, 1))",
    "print('Производная:', derivative(a, 1, 1))")

# Просят спрашивать три коэффициента, спрашивается один.
MISSING_INPUT = GOOD_DERIVATIVE

GOOD_WAIT = GOOD_DERIVATIVE + "    input('Нажмите Enter')\n"

# Пары «задача, файл» и ждём ли претензию. Файлы подобраны так, чтобы
# все механические проверки главы их пропускали: разбираются,
# импортируются, неопределённых имён нет.
REVIEW_CASES = [
    ("мёртвая функция", "выводи корни и производную", DEAD_DERIVATIVE, True),
    ("вызов не сходится", "выводи корни и производную", WRONG_ARITY, True),
    ("не спрашивает", "спрашивай у человека все три коэффициента a, b и c",
     MISSING_INPUT, True),
    ("не ждёт", "после вывода жди нажатия клавиши, чтобы окно не закрылось",
     GOOD_DERIVATIVE, True),
    ("всё на месте", "выводи корни и производную", GOOD_DERIVATIVE, False),
    ("ждёт как просили", "после вывода жди нажатия клавиши", GOOD_WAIT, False),
]


@pytest.mark.slow
class TestReviewQuality:
    """Замер 9: годится ли модель судьёй собственной работы.

    Разбор — единственная проверка конвейера, где спрашивают модель,
    и заведён он ради вопроса, на который механические проверки
    не отвечают: «то ли это, о чём просили». Файл разбирается,
    импортируется, имена определены, тесты зелёные — а производная,
    которую просили печатать, не печатается.

    Меряются ДВЕ величины, и обе нужны. «Поймал» без «промолчал»
    ничего не значит: судья, который придирается всегда, ловит все
    беды и не годится никуда — каждая его претензия отправляет готовую
    работу на лишний круг починки.

    Файлы подобраны так, чтобы ВСЕ механические проверки главы их
    пропускали. Иначе замер мерил бы не разбор, а `ruff` и `ast`.
    """

    ROUNDS = 2

    def test_разбор_на_моделях(self, tmp_path, warm_model, monkeypatch):
        installed = set(base.list_installed_models())
        models = [m for m in CODER_CANDIDATES if m in installed]
        if not models:
            pytest.skip(f"ни одна из моделей {CODER_CANDIDATES} не установлена")

        monkeypatch.setattr(review, "review", REAL_REVIEW)
        root = tmp_path / "разбор"
        root.mkdir()
        guard.set_policy(root=root, mode=guard.AUTO, dry_run=False)

        bad_total = sum(1 for *_, flag in REVIEW_CASES if flag) * self.ROUNDS
        good_total = sum(1 for *_, flag in REVIEW_CASES if not flag) * self.ROUNDS

        report = []
        for model in models:
            caught = quiet = 0
            spent = 0.0
            with using_model(model):
                for _, (name, task, source, should_flag) in rounds(self.ROUNDS, REVIEW_CASES):
                    (root / "app.py").write_text(source, encoding="utf-8")
                    started_at = time.monotonic()
                    done, problems = review.review(task, "app.py")
                    spent += time.monotonic() - started_at
                    flagged = bool(problems) and not done
                    if should_flag:
                        caught += flagged
                    else:
                        quiet += not flagged
            report.append((model, caught, quiet, spent))

        print("\n\nЗАМЕР 9: разбор написанного как судья")
        print(f"Случаев с бедой: {bad_total // self.ROUNDS}, без беды: "
              f"{good_total // self.ROUNDS}, прогонов каждого: {self.ROUNDS}")
        print(f"{'модель':<26}{'поймал беду':>14}{'промолчал зря нет':>20}{'секунд':>10}")
        for model, caught, quiet, spent in report:
            print(f"{model:<26}{caught:>11}/{bad_total:<2}{quiet:>17}/{good_total:<2}{spent:>10.1f}")

        assert report, "замер не собрал ни одной строки"


def flat_edit_schema() -> dict:
    """ПЕРВАЯ версия схемы правки: один объект, поля форм необязательны.

    Оставлена в тестах, а не в библиотеке: это то, с чем сравнивают.
    Рассуждение при её написании было такое — объявить `start`
    обязательным нельзя, иначе грамматика заставит заполнять его
    и при правке по якорю; значит, обязательны только `form` и `path`,
    а полноту проверим сами после разбора.
    """
    return {
        "type": "object",
        "properties": {
            "form": {"type": "string", "enum": list(EDIT_FORMS)},
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "content": {"type": "string"},
        },
        "required": ["form", "path"],
    }


def ask_with_schema(task, path, schema):
    """Один запрос за правкой по заданной схеме. Возвращает (форма, поля, отчёт)."""
    text = read_lines(path, "1", "120")
    messages = [
        {"role": "system", "content": pipeline.EDIT_RULES},
        {"role": "user", "content": f"Задача: {task}\n\nФайл {path}:\n{text}\n"},
    ]
    try:
        raw = request_model(messages, response_format=schema)
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return "", [f"модель: {exc}"], ""
    return str(data.get("form", "")), missing_fields(data), pipeline.apply_edit(data, default_path=path)


@pytest.mark.slow
class TestSchemaShape:
    """Замер 5: схема из вариантов против схемы с общим набором полей.

    Вопрос, который замер решает: достаточно ли `enum` в поле `form`,
    чтобы модель заполнила поля ИМЕННО ЭТОЙ формы. Проверяется по двум
    числам — сколько ответов пришло с недостающими полями и сколько
    правок легло на диск.

    Мерится по всем установленным моделям, потому что ответ у них разный:
    instruct-модель связь «форма — поля» держит из промпта, а сборка
    без такого тюнинга — нет, и для неё структура схемы решает всё.
    """

    def test_две_схемы_на_одном_наборе(self, tmp_path, warm_model):
        installed = set(base.list_installed_models())
        models = [m for m in CODER_CANDIDATES if m in installed]
        if not models:
            pytest.skip(f"ни одна из моделей {CODER_CANDIDATES} не установлена")

        shapes = (("общая (первая версия)", flat_edit_schema()), ("по вариантам, oneOf", edit_schema()))
        report = []
        for model in models:
            for label, schema in shapes:
                gaps = applied = 0
                with using_model(model):
                    for name, source, test_source, task in FIX_TASKS:
                        make_task_project(
                            tmp_path / f"{folder_name(model)}-{folder_name(label)}-{name}",
                            source, test_source,
                        )
                        _, missing, _ = ask_with_schema(task, "calc.py", schema)
                        gaps += bool(missing)
                        applied += bool(guard.change_count())
                report.append((model, label, gaps, applied))

        print("\n\nЗАМЕР 5: форма схемы ответа")
        print(f"Задач: {len(FIX_TASKS)}, форму выбирает модель")
        print(f"{'модель':<26}{'схема':<24}{'неполный ответ':>16}{'правка легла':>14}")
        for model, label, gaps, applied in report:
            print(f"{model:<26}{label:<24}{gaps:>13}/{len(FIX_TASKS):<2}{applied:>11}/{len(FIX_TASKS):<2}")

        assert report, "замер не собрал ни одной строки"


@pytest.mark.slow
class TestWriteMode:
    """Замер 6: файл целиком одним запросом против скелета и дописывания.

    Вопрос замера: помогает ли перевод задачи в форму, которой модель
    обучена. Базовые Coder-модели учили ДОПОЛНЯТЬ код, а не исполнять
    инструкции; «напиши файл по описанию» — второе, «допиши тела
    в готовый скелет» — первое.

    Мерится на задачах С НУЛЯ, потому что режим касается только их:
    правка существующего файла оба раза идёт одним и тем же путём.

    Считается доля прогонов, доведённых до зелёной проверки, и время.
    Время здесь не второстепенно: режим скелета делает ДВА запроса
    вместо одного, и если он не выигрывает в качестве, то проигрывает
    вдвойне.

    Отдельно считается, СКОЛЬКО прогонов кончились ненаписанным файлом.
    Без этой колонки замер отвечал не на свой вопрос: он складывал
    «код хуже» с «файл не дошёл до диска», а это разные ответы.
    Различив их, замер сразу показал две наши поломки — `import assert`
    в собранном скелете и разгон дописывания до предела длины, — и обе
    были починены. Вывод после починки не перевернулся, но теперь он
    про приём, а не про наши ошибки.

    Прогонов ДВА, по той же причине, что и в замере моделей: разброс
    между прогонами оказался больше измеряемой разницы. Одна и та же
    модель в соседних прогонах дала 1/5 и 0/5, её ровесница — 1/5 и 3/5;
    поодиночке эти цифры не значат ничего, и на паре из них я успел
    сделать вывод, который следующий прогон отменил.
    """

    ROUNDS = 2

    def test_два_режима_на_обеих_моделях(self, tmp_path, warm_model):
        installed = set(base.list_installed_models())
        models = [m for m in CODER_CANDIDATES if m in installed]
        if not models:
            pytest.skip(f"ни одна из моделей {CODER_CANDIDATES} не установлена")

        report = []
        for model in models:
            for mode in ("direct", "skeleton"):
                green = unwritten = 0
                spent = 0.0
                with using_model(model), pytest.MonkeyPatch.context() as patch:
                    patch.setattr(pipeline, "WRITE_MODE", mode)
                    for attempt, (name, task) in rounds(self.ROUNDS, SCRATCH_TASKS):
                        make_empty_project(tmp_path / f"{folder_name(model)}-{mode}{attempt}-{name}")
                        started_at = time.monotonic()
                        state = run_pipeline(task)
                        spent += time.monotonic() - started_at
                        green += bool(state.extra.get("tests_green"))
                        # Провал провалу рознь, и различать их обязательно.
                        # «Файл не написался» — беда механическая: ответ
                        # не разобрался, определения потерялись, вышло
                        # время. «Файл написался, проверка красная» — это
                        # уже про качество кода. Режим, проигрывающий
                        # первым, чинится; проигрывающий вторым —
                        # отвергается.
                        unwritten += bool(state.extra.get("failed_steps"))
                report.append((model, mode, green, unwritten, spent))

        print("\n\nЗАМЕР 6: как писать файл — целиком или по скелету")
        total = len(SCRATCH_TASKS) * self.ROUNDS
        print(f"Задач с нуля: {len(SCRATCH_TASKS)}, прогонов каждой: {self.ROUNDS}")
        print(f"{'модель':<26}{'режим':<12}{'зелёные':>10}{'файл не написан':>18}{'секунд':>10}")
        for model, mode, green, unwritten, spent in report:
            print(f"{model:<26}{mode:<12}{green:>7}/{total:<3}"
                  f"{unwritten:>15}/{total:<3}{spent:>10.0f}")

        assert report, "замер не собрал ни одной строки"


def dependency_closure(roots):
    """Все пакеты, которые тянут за собой названные, — по метаданным.

    Список руками здесь был ошибкой, и не потому, что в нём нашлась
    опечатка (её как раз не нашлось), а потому, что найти её было бы
    нечем: замер печатает «пакетов установлено: X из N ожидаемых»,
    и при опечатке в имени X молча уменьшается — замер продолжает
    работать и врать. Число, ради которого он написан, обязано браться
    оттуда же, откуда его берёт pip.

    Условия учитываются: `httpx2` нужен langsmith всегда, а `pytest` —
    только с extra `pytest`. Пустое окружение в `evaluate()` означает
    «нужен всегда», и необязательное отсекается само.
    """
    import importlib.metadata as md

    from packaging.requirements import Requirement

    seen: set[str] = set()
    queue = [str(name) for name in roots]
    while queue:
        name = queue.pop().lower().replace("_", "-")
        if name in seen:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen.add(name)
        for line in dist.requires or []:
            requirement = Requirement(line)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            queue.append(requirement.name)
    return seen


def disk_size(names):
    """Сколько места занимают названные пакеты, в байтах."""
    import importlib.metadata as md

    total = 0
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        for file in dist.files or []:
            try:
                total += dist.locate_file(file).stat().st_size
            except OSError:
                pass
    return total


@pytest.mark.slow
@needs_langgraph
class TestLangGraphCost:
    """Замер 4: во что обходится готовый сборщик графов.

    Здесь нет модели и нет случайности, поэтому замер быстрый — но
    метка `slow` стоит по другой причине: он лезет в метаданные
    установленных пакетов и на чужой машине даст другие числа.

    Считается не «сколько пакетов у langgraph», а сколько их СВЕРХ
    того, что курс ставит и так. Общее число было бы нечестным
    в обе стороны: pydantic и requests в нём уже есть, а вот
    langsmith со своим http-клиентом — нет.
    """

    # Корни, от которых считается «курс и так это ставит». Из
    # requirements.txt, кроме самой langgraph: она и есть предмет замера.
    COURSE_ROOTS = ("requests", "pytest", "pytest-timeout", "chromadb", "snowballstemmer")

    def test_цена_зависимости(self):
        added = dependency_closure(["langgraph"]) - dependency_closure(self.COURSE_ROOTS)
        if not added:
            pytest.skip("langgraph не установлена — считать нечего")

        own = len(Path("chapter7/src/graph.py").read_text(encoding="utf-8").splitlines())
        adapter = len(Path("chapter8/src/pipeline_lg.py").read_text(encoding="utf-8").splitlines())

        print("\n\nЗАМЕР 4: цена LangGraph")
        print(f"Пакетов сверх тех, что курс ставит и так: {len(added)}")
        print(f"   {', '.join(sorted(added))}")
        print(f"На диске: {disk_size(added) / 1024 / 1024:.1f} МБ")
        print(f"Свой граф (chapter7/src/graph.py): {own} строк, зависимостей 0")
        print(f"Переходник на LangGraph (pipeline_lg.py): {adapter} строк")

        started_at = time.monotonic()
        pipeline_lg.build_langgraph_pipeline()
        print(f"Сборка графа LangGraph: {time.monotonic() - started_at:.2f} с")

        started_at = time.monotonic()
        build_pipeline()
        print(f"Сборка своего графа: {time.monotonic() - started_at:.4f} с")

        assert "langgraph" in added, "предмет замера обязан попасть в счёт"
