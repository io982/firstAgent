"""
Обход репозитория: что берём в индекс и чем это разбирать (пункт 5.2).

Корпус Главы 4 — папка с документами, которую собрал человек. Здесь корпус
другой: рабочий репозиторий, в котором рядом с исходниками лежит всё, что
насыпали инструменты, — виртуальное окружение, кэши, собранные бандлы,
база самой Chroma. Поэтому обход начинается не с вопроса «какие расширения
берём», а с вопроса «куда вообще не заходим».

Три причины пропускать папку, и они разные:

  * **это не наш код** — node_modules, .venv: чужие исходники в индексе
    не просто занимают место, они выигрывают у своих по количеству;
  * **это производное** — __pycache__, dist, chroma_db: то, что собирается
    из кода, отвечать на вопросы о коде не должно;
  * **это не в репозитории** — всё, что перечислено в .gitignore. Индекс,
    который знает больше, чем git, показывает пользователю файлы, которых
    для проекта не существует.
"""

import os
import re
from pathlib import Path

# ====================================================================
# ЯЗЫКИ
# ====================================================================

# Расширение → язык. Язык здесь означает ровно одно: каким способом файл
# будет нарезан (см. codechunks.chunk_source). Поэтому .json и .toml —
# один «язык» config: разбираются они одинаково.
LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".rst": "markdown",
    ".txt": "text",
    ".json": "config",
    ".yml": "config",
    ".yaml": "config",
    ".toml": "config",
    ".ini": "config",
    ".cfg": "config",
}

# Файлы без расширения, которые всё равно стоит индексировать: у половины
# инфраструктуры проекта имя вместо расширения.
NAMED_FILES: dict[str, str] = {
    "Dockerfile": "text",
    "Makefile": "text",
    ".gitignore": "text",
    ".env.example": "config",
    # Лицензия — тоже файл проекта, и вопрос «какая тут лицензия» законный.
    # Без неё в индексе агент на этот вопрос отвечал «в документах этого нет»,
    # а на уточняющий — выдумывал файл, в котором лицензия якобы описана.
    "LICENSE": "text",
    "LICENSE.txt": "text",
    "COPYING": "text",
    "NOTICE": "text",
    "CHANGELOG": "text",
}

# ====================================================================
# ЧТО ПРОПУСКАЕМ
# ====================================================================

# Папки, в которые не заходим никогда, даже если git про них ничего не знает.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
    ".venv", "venv", "env", "site-packages",
    "node_modules", "bower_components",
    "dist", "build", "out", "target", "htmlcov", ".next", ".nuxt",
    ".idea", ".vscode",
    "chroma_db",
    # Настройки самого агента-помощника: скиллы, хуки, права. Это про то,
    # чем проект разрабатывают, а не про то, из чего он состоит.
    ".claude",
})

# Файлы, которые формально исходники, а по сути данные: собранные бандлы,
# карты соответствия, зафиксированные версии зависимостей. Один
# package-lock.json даёт больше «фрагментов кода», чем весь проект.
SKIP_SUFFIXES: tuple[str, ...] = (
    ".min.js", ".min.css", ".bundle.js", ".map",
    "-lock.json", ".lock",
)
SKIP_NAMES: frozenset[str] = frozenset({
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
})

# Файлы с тестами. Они — часть проекта и остаются в индексе, но на вопрос
# «где реализовано» отвечает реализация, а не тест: и в поиске, и в
# перечислениях тесты уезжают в хвост (см. demote_tests и list_symbols).
TEST_FILE = re.compile(r"(^|/)(tests?|test_[\w-]+|[\w-]+_test)\.\w+$")


def is_test_source(source: str) -> bool:
    """Похож ли путь на файл с тестами."""
    return bool(TEST_FILE.search(source or ""))


# Потолок размера файла. Больше — почти наверняка не исходник, а выгрузка:
# сгенерированный клиент API, словарь, дамп. Нарезать его можно, но в индексе
# он даст сотни одинаковых фрагментов и утопит настоящий код.
#
# 200 КБ — это примерно 5000 строк Python: файл такого размера уже редкость,
# а всё, что заметно больше, писал не человек.
MAX_FILE_BYTES = 200_000


# ====================================================================
# .gitignore
# ====================================================================

def gitignore_entries(root: Path) -> tuple[set[str], set[str]]:
    """Что из .gitignore мы умеем понять: имена папок и пути к файлам.

    Возвращает пару (папки, файлы). Разбирается НЕ весь формат .gitignore,
    и это осознанно: полная его поддержка — отдельная библиотека
    с приоритетами, отрицаниями и вложенными файлами правил.

    Две разные вещи, которые раньше были одной. Строка `drafts/` — это папка,
    в неё не заходим. Строка `chapter3/memory.json` — это ФАЙЛ внутри папки,
    и пропустить надо именно его, а не всю Главу 3.

    Второй случай сначала просто выбрасывался, и это была не мелочь: файл
    `chapter3/memory.json` с фактами о пользователе и `previous_session.json`
    с хвостом прошлого разговора попадали в индекс кода наравне с исходниками
    — уезжали в модель эмбеддингов и ложились на диск в chroma_db. Ровно то,
    что тексты глав 5 и 6 обещали не делать.
    """
    directories, files = _parse_gitignore(root)
    return directories, files


def gitignore_dirs(root: Path) -> set[str]:
    """Только имена папок. Оставлено ради совместимости с Главой 5.

    Разбирается НЕ весь формат .gitignore, и это осознанно: полная его
    поддержка — это отдельная библиотека с приоритетами, отрицаниями и
    вложенными файлами правил. Берём то, что покрывает главный случай:
    строки вида `chroma_db/`, `drafts/`, `node_modules` — то есть имя папки
    без спецсимволов.

    Зачем вообще: без этого индекс кода этого репозитория собирал бы старые
    черновики глав из `drafts/` наравне с живым кодом — и агент отвечал бы
    по коду, которого в проекте нет.
    """
    return _parse_gitignore(root)[0]


def _parse_gitignore(root: Path) -> tuple[set[str], set[str]]:
    """Разбор .gitignore на имена папок и относительные пути к файлам."""
    ignore_file = Path(root) / ".gitignore"
    if not ignore_file.exists():
        return set(), set()

    directories: set[str] = set()
    files: set[str] = set()
    try:
        lines = ignore_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return directories, files

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if any(char in line for char in "*?[]"):
            continue  # шаблон, а не имя — не наш случай
        name = line.rstrip("/").lstrip("/")
        if not name:
            continue
        if "/" in name:
            # Путь: `chapter3/memory.json`, `chapter4/index`. Папка это или
            # файл — решаем по диску, а не по написанию: строка `chapter4/index`
            # без слэша на конце всё равно папка.
            if (Path(root) / name).is_dir():
                directories.add(name)
            else:
                files.add(name)
        else:
            directories.add(name)

    return directories, files


# ====================================================================
# ОБХОД
# ====================================================================

def language_of(path: Path | str) -> str | None:
    """Язык файла или None, если такой файл в индекс не берём."""
    path = Path(path)
    if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES):
        return None
    if path.name in NAMED_FILES:
        return NAMED_FILES[path.name]
    return LANGUAGES.get(path.suffix.lower())


def iter_sources(
    root: Path | str,
    respect_gitignore: bool = True,
    max_bytes: int = MAX_FILE_BYTES,
) -> list[Path]:
    """Файлы репозитория, годные для индексации, в стабильном порядке.

    Обход идёт через os.walk, а не через rglob, ровно по одной причине:
    os.walk позволяет ОТРЕЗАТЬ папку целиком, а rglob сначала зайдёт
    в node_modules и перечислит все сорок тысяч файлов, чтобы мы их потом
    отфильтровали. На настоящем фронтенд-проекте разница между этими
    двумя способами — секунды против минут.

    Порядок сортируется по той же причине, что и в Главе 4: от него зависят
    позиции фрагментов, а значит и их id.
    """
    root = Path(root)
    if root.is_file():
        return [root] if language_of(root) else []

    skip = set(SKIP_DIRS)
    skip_files: set[str] = set()
    if respect_gitignore:
        ignored_dirs, skip_files = gitignore_entries(root)
        # Имена папок отсекаются на любом уровне вложенности, пути — целиком.
        skip |= {name for name in ignored_dirs if "/" not in name}
        nested = {name for name in ignored_dirs if "/" in name}
    else:
        nested = set()

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Список dirnames правится НА МЕСТЕ — так os.walk понимает, что
        # внутрь этих папок заходить не надо. Присваивание новому имени
        # (dirnames = [...]) не сработало бы.
        #
        # Скрытые папки целиком не отсекаются: .github — это код проекта
        # (сборка, тесты, релиз), и вопрос «где настроен CI» законный.
        dirnames[:] = sorted(d for d in dirnames if d not in skip)

        relative_dir = Path(dirpath).relative_to(root).as_posix()
        if relative_dir != "." and any(
            relative_dir == name or relative_dir.startswith(name + "/") for name in nested
        ):
            dirnames[:] = []
            continue

        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if not language_of(path):
                continue
            # Файл, названный в .gitignore поимённо: `chapter3/memory.json`.
            # Проверяется до размера и языка — там лежат личные данные,
            # и в индексе им не место ни при каких условиях.
            if path.relative_to(root).as_posix() in skip_files:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            found.append(path)

    return found


def read_source(path: Path | str) -> str | None:
    """Читает файл, если он текстовый. Иначе — None, без исключения.

    Двоичный файл с «правильным» расширением — не выдумка: .ts бывает
    видеопотоком MPEG, а .cfg — бинарным конфигом чужой программы.
    Индексация из-за такого падать не должна.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
