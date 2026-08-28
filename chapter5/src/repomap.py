"""
Карта проекта: не всё есть поиск (пункт 5.5).

Часть вопросов о коде вообще не нуждается в эмбеддингах, и это стоит
сказать прямо в главе про поиск по коду.

«Где определён `estimate_tokens`» — вопрос с ровно одним правильным
ответом. Векторная близость на нём не помогает, а мешает: она вернёт
пять фрагментов, где это имя встречается, и лучший из них не обязательно
тот, где стоит `def`. Точный ответ даёт таблица символов — обычный
словарь «имя → место определения».

«Кто импортирует Главу 3» — вопрос про граф. Ребро либо есть, либо нет,
приблизить его нельзя, и никакой реранкер тут не нужен.

Всё это собирается одним проходом по файлам, без единого запроса к модели:
для Python — тем же `ast`, что и нарезка, для остальных языков — теми же
чанками, у которых уже есть имена (см. codechunks).
"""

import ast
import difflib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .codechunks import chunk_code, first_doc_line, signature_of
from .languages import is_test_source, iter_sources, language_of, read_source

# Корень проекта — папка на уровень выше пакета главы. Меняется через
# переменную окружения, если агента натравливают на чужой репозиторий:
#   PowerShell:   $env:AGENT_CODE_ROOT = "C:\\projects\\other"
#   Linux/macOS:  export AGENT_CODE_ROOT=~/projects/other
DEFAULT_ROOT = Path(os.environ.get("AGENT_CODE_ROOT", str(Path(__file__).parent.parent.parent)))

# Сколько строк отдаём в обзоре проекта. Обзор едет в контекст модели,
# а там место общее с найденным кодом и разговором.
OVERVIEW_PACKAGES = 12
OVERVIEW_EXTERNAL = 12
OVERVIEW_ENTRYPOINTS = 8

# Что считать внешней зависимостью. `os` и `json` формально импортируются
# так же, как `requests`, но зависимостью проекта не являются: они приезжают
# вместе с Python. Список стандартных модулей интерпретатор знает сам —
# выдумывать его не нужно.
STDLIB = set(sys.stdlib_module_names)

# Импорт в JavaScript и TypeScript: разбора у нас нет, но `from "..."`
# и `require("...")` узнаются надёжно — это не тот случай, где регулярное
# выражение притворяется парсером.
JS_IMPORT = re.compile(r"""(?:from\s+|require\(\s*)["']([^"']+)["']""")


# Как человек называет язык. Пишут и «typescript», и «ts», и по-русски —
# а промахиваются мимо любого из вариантов регулярно: живой прогон начался
# с вопроса про «typeScrypt». Поэтому имя языка ещё и подбирается по
# похожести (см. resolve_language).
LANGUAGE_ALIASES = {
    "python": "python", "питон": "python", "py": "python", "пайтон": "python",
    "javascript": "javascript", "js": "javascript", "джаваскрипт": "javascript",
    "typescript": "typescript", "ts": "typescript", "тайпскрипт": "typescript",
    "markdown": "markdown", "md": "markdown", "документация": "markdown",
    "config": "config", "конфиг": "config", "конфиги": "config",
}


def resolve_language(name: str) -> str:
    """Приводит название языка к тому, каким его знает индексатор."""
    key = name.strip().lower().lstrip(".")
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    close = difflib.get_close_matches(key, list(LANGUAGE_ALIASES), n=1, cutoff=0.8)
    return LANGUAGE_ALIASES[close[0]] if close else ""


@dataclass(frozen=True)
class Symbol:
    """Определение: что, где и с какой сигнатурой."""

    name: str          # квалифицированное: KnowledgeBase.search
    kind: str          # function | class | method | type | constant
    source: str        # путь относительно корня, через /
    line: int
    signature: str = ""
    docstring: str = ""

    @property
    def short_name(self) -> str:
        """Имя без класса — то, которое обычно называет человек."""
        return self.name.rsplit(".", 1)[-1]

    def label(self) -> str:
        return f"{self.source}:{self.line}"

    def render(self) -> str:
        head = f"{self.kind} {self.name} — {self.label()}"
        parts = [head]
        if self.signature:
            parts.append(f"    {self.signature}")
        if self.docstring:
            parts.append(f"    {self.docstring}")
        return "\n".join(parts)


@dataclass
class ProjectMap:
    """Символы, импорты и структура проекта — всё, что известно без модели."""

    root: Path
    files: list[str] = field(default_factory=list)
    definitions: list[Symbol] = field(default_factory=list)        # все, по одному разу
    symbols: dict[str, list[Symbol]] = field(default_factory=dict)  # указатель: имя → где
    modules: dict[str, str] = field(default_factory=dict)          # модуль → файл
    imports: dict[str, set[str]] = field(default_factory=dict)     # модуль → внутренние
    external: dict[str, set[str]] = field(default_factory=dict)    # чужой пакет → кто импортирует
    stdlib: dict[str, set[str]] = field(default_factory=dict)      # модуль stdlib → кто импортирует
    entrypoints: list[str] = field(default_factory=list)
    seconds: float = 0.0

    # ------------------------------------------------------------ поиск

    def find(self, name: str) -> list[Symbol]:
        """Все определения с таким именем.

        Ищется и по короткому имени (`search`), и по квалифицированному
        (`KnowledgeBase.search`), и без учёта регистра — человек называет
        символ так, как помнит, а не так, как он записан.
        """
        name = name.strip().strip("()")
        if not name:
            return []

        found = list(self.symbols.get(name, []))
        if found:
            return found

        lowered = name.lower()
        for key, symbols in self.symbols.items():
            if key.lower() == lowered:
                found.extend(symbols)
        if found:
            return found

        # Последняя попытка: человек назвал метод вместе с классом, а мы
        # храним его и под коротким именем тоже — или наоборот.
        for symbols in self.symbols.values():
            for symbol in symbols:
                if symbol.name.lower().endswith("." + lowered):
                    found.append(symbol)
        return found

    def imported_by(self, module: str) -> set[str]:
        """Кто импортирует этот модуль. Обратные рёбра графа импортов."""
        return {source for source, targets in self.imports.items() if module in targets}

    # ------------------------------------------------------------ перечисление

    def in_file(self, path: str) -> list[Symbol]:
        """Все определения одного файла, в порядке следования в нём.

        Вопрос «что есть в этом файле» — перечисление, а не поиск. Живой
        прогон показал, чем он оборачивается без такого метода: на вопрос
        «какие инструменты есть в chapter4/src/tools.py» векторный поиск
        честно вернул ШАПКУ этого файла первым же местом (близость 0.886),
        а в шапке лежат импорты и константы. Агент ответил «инструментов
        в файле нет», хотя их там два.
        """
        wanted = path.strip().replace("\\", "/").lstrip("./").lower()
        if not wanted:
            return []
        found = [
            symbol for symbol in self.definitions
            if symbol.source.lower() == wanted or symbol.source.lower().endswith("/" + wanted)
        ]
        return sorted(found, key=lambda symbol: symbol.line)

    def in_language(self, language: str) -> list[Symbol]:
        """Все определения на одном языке — по расширениям файлов."""
        wanted = resolve_language(language)
        if not wanted:
            return []
        found = [
            symbol for symbol in self.definitions
            if language_of(Path(symbol.source)) == wanted
        ]
        return sorted(found, key=lambda symbol: (symbol.source, symbol.line))

    def list_symbols(self, where: str, limit: int = 40, kind: str = "") -> str:
        """Перечисляет определения файла, языка или модуля — что назвали.

        Ответ группируется по файлам: список из сорока имён подряд читается
        хуже, чем те же сорок имён под тремя заголовками.

        `kind` отбирает вид определения. Без него вопрос «какие классы
        реализованы в chapter5» отвечался так: в справку попадали первые
        сорок определений пакета — все из `agent.py`, где классов нет, —
        и агент честно отвечал «классов нет». Спросили про классы — значит
        и в справке должны быть классы, а не всё подряд.
        """
        where = where.strip()
        if not where:
            return "Не понял, что перечислить: назовите файл, язык или модуль."

        # Сначала выясняем, о каких ФАЙЛАХ речь, и только потом смотрим,
        # что в них лежит. Порядок важен: файл без единого определения —
        # это не «не нашёл», а «нашёл, и там пусто». Путать эти два ответа
        # нельзя: на «не нашёл» модель охотно придумывает содержимое,
        # что и произошло в живом прогоне с `./__init__.py`.
        files = self.files_matching(where)
        # Подобрали похожий путь вместо названного — об этом надо сказать
        # прямо. Живой прогон: спросили про `./test.py`, которого в проекте
        # нет, получили содержимое `chapter2/tests.py` — и агент выдал его
        # за содержимое `./test.py`.
        substituted = bool(files) and not any(
            source.lower() == where.lower().replace("\\", "/").lstrip("./")
            or source.lower().endswith("/" + where.lower().replace("\\", "/").lstrip("./"))
            for source in files
        )
        found = [symbol for symbol in self.definitions if symbol.source in files]

        if not files:
            found = self.in_language(where)
        if not found and not files:
            for module in self.resolve_module(where):
                found.extend(self.in_file(self.modules[module]))

        if kind:
            narrowed = [symbol for symbol in found if symbol.kind == kind]
            if not narrowed and found:
                return (
                    f"Определений вида «{kind}» в «{where}» нет "
                    f"(всего определений: {len(found)}). Так и скажи пользователю."
                )
            found = narrowed

        if files and not found:
            listed = ", ".join(sorted(files)[:5])
            return (
                f"Файл найден, но определений в нём нет: {listed}. "
                f"Так и скажи пользователю — не придумывай содержимое."
            )

        if not found:
            return f"«{where}» не похоже ни на файл проекта, ни на язык, ни на модуль."

        # Тесты — в хвост, по той же причине, что и в поиске: на вопрос
        # «какие классы в chapter5» тридцать шесть тестовых классов
        # вытесняли из справки четыре настоящих.
        found = sorted(found, key=lambda symbol: is_test_source(symbol.source))

        by_file: dict[str, list[Symbol]] = {}
        for symbol in found[:limit]:
            by_file.setdefault(symbol.source, []).append(symbol)

        blocks = []
        if substituted:
            blocks.append(
                f"Файла «{where}» в проекте нет. Ближайший по имени — "
                f"{', '.join(sorted(files)[:3])}. Скажи об этом пользователю."
            )
        # Имя вроде `__init__.py` есть в каждом пакете. Молча слить их
        # в один список — значит показать модели «содержимое файла»,
        # собранное из шести разных файлов; она так и ответит.
        if len(by_file) > 1:
            blocks.append(
                f"Под «{where}» подходит файлов: {len(by_file)}. "
                f"Ниже определения каждого по отдельности."
            )

        for source, symbols in by_file.items():
            lines = [f"{source} ({len(symbols)} определений):"]
            lines += [
                f"  {symbol.kind} {symbol.name} — строка {symbol.line}"
                for symbol in symbols
            ]
            blocks.append("\n".join(lines))

        if len(found) > limit:
            blocks.append(f"…и ещё {len(found) - limit} определений.")
        return "\n\n".join(blocks)

    def files_matching(self, path: str) -> list[str]:
        """Файлы проекта, подходящие под то, что назвал человек.

        Совпадение по полному пути, по имени файла или по имени без
        расширения, а если ничего не сошлось — по похожести: опечатки
        в путях неизбежны, и «gitignore» вместо «.gitignore» человек
        пишет чаще, чем правильный вариант.
        """
        wanted = path.strip().replace("\\", "/").lstrip("./").lower()
        # Два символа — это не имя файла, а предлог; такое сравнивать
        # по похожести бессмысленно.
        if len(wanted) < 3:
            return []

        matches = [
            source for source in self.files
            if source.lower() == wanted or source.lower().endswith("/" + wanted)
        ]
        if matches:
            return matches

        # Имя без пути и без расширения: «cards», «LICENSE», «gitignore».
        matches = [
            source for source in self.files
            if source.split("/")[-1].lower() in (wanted, f".{wanted}")
            or source.split("/")[-1].rsplit(".", 1)[0].lower() == wanted
        ]
        if matches:
            return matches

        closest = self.closest_file(wanted)
        if not closest:
            return []
        closest = closest.lower()
        return [
            source for source in self.files
            if source.lower() == closest or source.lower().endswith("/" + closest)
        ]

    def closest_file(self, path: str) -> str:
        """Ближайший существующий путь к тому, что назвал человек.

        Сравниваются и полные пути, и одни имена файлов: «./__init__.pyм»
        ближе всего к `__init__.py`, а «chapter4/tools.py» — к
        `chapter4/src/tools.py`.
        """
        wanted = path.strip().replace("\\", "/").lstrip("./").lower()
        if not wanted:
            return ""

        names = {source.split("/")[-1].lower(): source for source in self.files}
        paths = {source.lower(): source for source in self.files}

        for shelf in (paths, names):
            close = difflib.get_close_matches(wanted, list(shelf), n=1, cutoff=0.75)
            if close:
                return shelf[close[0]].split("/")[-1] if shelf is names else shelf[close[0]]
        return ""

    def resolve_module(self, query: str) -> list[str]:
        """Превращает то, что назвал человек, в имена модулей.

        Принимаются и `chapter4.src.knowledge`, и `chapter4/src/knowledge.py`,
        и просто `knowledge`: спрашивают обычно последним способом.
        """
        query = query.strip().replace("\\", "/").removesuffix(".py")
        if not query:
            return []
        dotted = query.replace("/", ".")

        # Внутренности пакета — это тоже он: «что импортирует chapter4»
        # означает весь пакет, а не только его __init__.py, который в этом
        # курсе как раз пустой.
        inside = sorted(module for module in self.modules if module.startswith(dotted + "."))

        if dotted in self.modules:
            return [dotted] + inside

        matches = [
            module for module in self.modules
            if module.endswith("." + dotted) or module.split(".")[-1] == dotted
        ]
        return sorted(matches) if matches else inside

    # ------------------------------------------------------------ отчёты

    def dependencies(self, query: str, direction: str = "both") -> str:
        """Импорты модуля в обе стороны, человекочитаемо.

        `direction` управляет не содержанием, а ПОРЯДКОМ: справка одна и та
        же, но первой строкой идёт ответ на заданный вопрос.

        Это не косметика. Живой прогон на qwen2.5:3b: на вопрос «кто
        импортирует chapter3.src.context» модель получила справку, где
        первой строкой стояло «импортирует из проекта», и пересказала
        именно её — то есть ответила про обратное направление. Порядок
        строк в справке для 3B значит больше, чем их содержание.

        Значения: "in" — кто импортирует модуль, "out" — что импортирует он,
        "both" — как есть, без ведущей строки.
        """
        modules = self.resolve_module(query)
        if not modules:
            return f"Модуль «{query}» в проекте не найден."

        blocks: list[str] = []
        if direction in ("in", "out"):
            module = modules[0]
            if direction == "in":
                who = sorted(self.imported_by(module))
                blocks.append(
                    f"Прямой ответ: модуль {module} импортируют — {', '.join(who) or 'никто'}."
                )
            else:
                what = sorted(self.imports.get(module, set()))
                blocks.append(
                    f"Прямой ответ: модуль {module} импортирует — {', '.join(what) or 'ничего'}."
                )

        for module in modules[:5]:
            uses = sorted(self.imports.get(module, set()))
            used_by = sorted(self.imported_by(module))
            outside = sorted(
                package for package, users in self.external.items() if module in users
            )
            # Стандартная библиотека печатается отдельной строкой: `os` и
            # `requests` импортируются одинаково, но зависимость проекта —
            # только второй, и путать их в ответе про зависимости нельзя.
            standard = sorted(
                package for package, users in self.stdlib.items() if module in users
            )
            blocks.append(
                f"{module} ({self.modules[module]})\n"
                f"  импортирует из проекта: {', '.join(uses) or '—'}\n"
                f"  внешние пакеты: {', '.join(outside) or '—'}\n"
                f"  стандартная библиотека: {', '.join(standard) or '—'}\n"
                f"  импортируется модулями: {', '.join(used_by) or '—'}"
            )
        if len(modules) > 5:
            blocks.append(f"…и ещё {len(modules) - 5} модулей с таким именем.")
        return "\n\n".join(blocks)

    def overview(self) -> str:
        """Структура проекта: пакеты, точки входа, внешние зависимости.

        Это ответ на вопрос «из чего вообще состоит проект», который
        поиском по фрагментам не отвечается в принципе: ни в одном
        отдельно взятом файле такого текста нет.
        """
        packages: dict[str, int] = {}
        for path in self.files:
            top = path.split("/")[0] if "/" in path else "."
            packages[top] = packages.get(top, 0) + 1

        lines = [
            f"Проект: {self.root.name} ({len(self.files)} файлов, "
            f"{len(self.definitions)} определений)",
            "",
            "Папки верхнего уровня:",
        ]
        for name, count in sorted(packages.items(), key=lambda item: -item[1])[:OVERVIEW_PACKAGES]:
            lines.append(f"  {name}: {count} файлов")

        if self.entrypoints:
            lines += ["", "Точки входа (if __name__ == '__main__'):"]
            lines += [f"  {path}" for path in self.entrypoints[:OVERVIEW_ENTRYPOINTS]]

        if self.external:
            lines += ["", "Внешние зависимости (сколько модулей импортируют):"]
            ranked = sorted(self.external.items(), key=lambda item: (-len(item[1]), item[0]))
            for package, users in ranked[:OVERVIEW_EXTERNAL]:
                lines.append(f"  {package}: {len(users)}")

        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            "symbols": len(self.definitions),
            "modules": len(self.modules),
            "edges": sum(len(targets) for targets in self.imports.values()),
        }


# ====================================================================
# СБОРКА КАРТЫ
# ====================================================================

def module_name(source: str) -> str:
    """Путь файла → имя модуля: chapter4/src/knowledge.py → chapter4.src.knowledge."""
    stem = source.removesuffix(".py")
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _is_main_guard(node: ast.stmt) -> bool:
    """Тот самый `if __name__ == "__main__":` — признак запускаемого файла."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _python_symbols(tree: ast.Module, source: str) -> list[Symbol]:
    """Определения верхнего уровня плюс методы классов плюс КОНСТАНТЫ.

    Константы попадают в таблицу не для красоты: половина решений этого
    курса живёт именно в них — `CHUNK_SIZE`, `TOP_K`, `NUM_CTX`. Вопрос
    «где задан размер чанка» — это вопрос про константу, а не про функцию.
    """
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol(
                name=node.name, kind="function", source=source, line=node.lineno,
                signature=signature_of(node), docstring=first_doc_line(ast.get_docstring(node)),
            ))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                name=node.name, kind="class", source=source, line=node.lineno,
                signature=signature_of(node), docstring=first_doc_line(ast.get_docstring(node)),
            ))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(Symbol(
                        name=f"{node.name}.{child.name}", kind="method", source=source,
                        line=child.lineno, signature=signature_of(child),
                        docstring=first_doc_line(ast.get_docstring(child)),
                    ))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(Symbol(
                        name=target.id, kind="constant", source=source, line=target.lineno,
                    ))
    return symbols


def _python_imports(tree: ast.Module, module: str) -> list[str]:
    """Имена модулей, которые импортирует файл, включая относительные.

    Относительный импорт (`from .chunking import chunk_text`) разворачивается
    в полное имя: без этого половина рёбер графа теряется, а именно
    относительными импортами и связаны модули внутри одного пакета.
    """
    package = module.rsplit(".", 1)[0] if "." in module else ""
    names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = ".".join(parts[: len(parts) - node.level + 1])
                if node.module:
                    names.append(f"{base}.{node.module}")
                else:
                    # `from . import module` — имя модуля лежит не в node.module,
                    # а в списке импортируемых имён. Без этой ветки ребро ведёт
                    # в пакет вместо конкретного модуля, и половина связей
                    # внутри пакета теряется.
                    names.extend(f"{base}.{alias.name}" for alias in node.names)
            elif node.module:
                names.append(node.module)

    return [name for name in names if name]


def scan(root: Path | str | None = None) -> ProjectMap:
    """Собирает карту проекта одним проходом по файлам.

    Стоит это ровно чтения файлов и разбора Python: ни одного запроса
    к модели эмбеддингов, ни одного к LLM. Поэтому карту можно пересобирать
    при каждом запуске, а индекс — сверять (см. codebase).
    """
    started = time.time()
    root = Path(root) if root else DEFAULT_ROOT
    project = ProjectMap(root=root)

    for path in iter_sources(root):
        try:
            source = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            source = path.name
        project.files.append(source)

        language = language_of(path)
        text = read_source(path)
        if text is None:
            continue

        if language == "python":
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                continue  # сломанный файл не должен ломать карту

            module = module_name(source)
            project.modules[module] = source
            project.imports.setdefault(module, set())

            for symbol in _python_symbols(tree, source):
                project.definitions.append(symbol)
                project.symbols.setdefault(symbol.name, []).append(symbol)
                if "." in symbol.name:
                    # Метод кладётся и под коротким именем: спрашивают чаще
                    # «где search», чем «где KnowledgeBase.search».
                    project.symbols.setdefault(symbol.short_name, []).append(symbol)

            for target in _python_imports(tree, module):
                project.imports[module].add(target)

            if any(_is_main_guard(node) for node in tree.body):
                project.entrypoints.append(source)

        elif language in ("javascript", "typescript"):
            for chunk in chunk_code(text, source, language):
                if chunk.name and chunk.kind != "block":
                    symbol = Symbol(
                        name=chunk.name, kind=chunk.kind, source=source,
                        line=chunk.start_line, signature=chunk.signature,
                        docstring=chunk.docstring,
                    )
                    project.definitions.append(symbol)
                    project.symbols.setdefault(symbol.name, []).append(symbol)
            for match in JS_IMPORT.finditer(text):
                target = match.group(1)
                if not target.startswith("."):
                    project.external.setdefault(target.split("/")[0], set()).add(source)

    # Импорты разъезжаются на внутренние и внешние только сейчас, когда
    # известен весь список модулей: `chapter3.src.context` — свой, `requests`
    # — чужой, и понять это заранее, по одному файлу, нельзя.
    for module, targets in project.imports.items():
        internal = {target for target in targets if _internal(target, project.modules)}
        for target in targets - internal:
            package = target.split(".")[0]
            shelf = project.stdlib if package in STDLIB else project.external
            shelf.setdefault(package, set()).add(module)
        project.imports[module] = {_resolve(target, project.modules) for target in internal}

    project.seconds = time.time() - started
    return project


def _internal(target: str, modules: dict[str, str]) -> bool:
    """Импорт ведёт внутрь проекта, если такой модуль или пакет у нас есть."""
    return bool(_resolve(target, modules))


def _resolve(target: str, modules: dict[str, str]) -> str:
    """Приводит цель импорта к имени модуля проекта, если это он.

    `from chapter4.src.chunking import chunk_text` даёт цель-модуль,
    а `import chapter4.src` — цель-пакет: у пакета есть свой `__init__.py`,
    и в таблице модулей он записан просто как `chapter4.src`.
    """
    if target in modules:
        return target
    head = target
    while "." in head:
        head = head.rsplit(".", 1)[0]
        if head in modules:
            return head
    return ""


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_project_map: ProjectMap | None = None


def get_project_map() -> ProjectMap:
    """Общая карта проекта (singleton, как база знаний в Главе 4)."""
    global _project_map
    if _project_map is None:
        _project_map = scan()
    return _project_map


def set_project_map(project: ProjectMap | None) -> None:
    """Подменяет общую карту. Нужно тестам и команде пересборки в REPL."""
    global _project_map
    _project_map = project
