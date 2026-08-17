"""Глава 4. Система плагинов: реестр инструментов агента.

Каждый инструмент — это обычная функция с декоратором @tool.
Чтобы добавить новый инструмент, достаточно 5 строк:

    @tool("weather", "Возвращает погоду для города.")
    def weather(city: str) -> str:
        return f"В городе {city} солнечно, +25°C"

Ядро агента (chapter4/agent.py) само подхватит новый плагин.
"""

import ast
import inspect
import operator
import os
import subprocess
from datetime import datetime

import requests

# ====================================================================
# РЕЕСТР ИНСТРУМЕНТОВ
# ====================================================================

# name -> {"func": callable, "description": str, "parameters": dict}
TOOL_REGISTRY = {}


def _extract_parameters(func) -> dict:
    """Извлекает параметры функции через inspect.signature."""
    sig = inspect.signature(func)
    params = {}
    for param_name, param in sig.parameters.items():
        param_info = {"type": "any"}

        if param.annotation != inspect.Parameter.empty:
            if param.annotation is str:
                param_info["type"] = "string"
            elif param.annotation is int:
                param_info["type"] = "integer"
            elif param.annotation is float:
                param_info["type"] = "number"
            elif param.annotation is bool:
                param_info["type"] = "boolean"

        if param.default != inspect.Parameter.empty:
            param_info["required"] = False
            param_info["default"] = param.default
        else:
            param_info["required"] = True

        params[param_name] = param_info
    return params


def tool(name: str, description: str):
    """Декоратор: регистрирует функцию как инструмент агента."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "func": func,
            "description": description,
            "parameters": _extract_parameters(func)
        }
        return func
    return decorator


def known_tools() -> set:
    """Множество имён зарегистрированных инструментов."""
    return set(TOOL_REGISTRY)


def execute_tool(name: str, args: dict) -> str:
    """Вызывает инструмент по имени, фильтруя лишние аргументы."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Неизвестный инструмент: {name}"
    try:
        func = entry["func"]
        sig = inspect.signature(func)
        # Функция с **kwargs принимает любые аргументы — отбрасывать нечего.
        # Это позволяет инструменту самому разобрать то, что выдумала модель.
        accepts_any = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_any:
            valid_args = args
        else:
            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        return str(func(**valid_args))
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


def render_tools_for_prompt() -> str:
    """Собирает список инструментов С ПАРАМЕТРАМИ для системного промпта."""
    lines = []
    for number, (name, entry) in enumerate(TOOL_REGISTRY.items(), 1):
        line = f"{number}. {name} — {entry['description']}"

        params = entry.get("parameters", {})
        if params:
            param_parts = []
            for pname, pinfo in params.items():
                if pinfo.get("required", True):
                    param_parts.append(f'{pname} ({pinfo["type"]}, обязательно)')
                else:
                    default = pinfo.get("default")
                    param_parts.append(f'{pname} ({pinfo["type"]}, по умолчанию: {default!r})')
            line += f"\n   Параметры: {', '.join(param_parts)}"

        lines.append(line)
    return "\n".join(lines)

# ====================================================================
# ВСТРОЕННЫЕ ПЛАГИНЫ (те же инструменты, что в Главе 1)
# ====================================================================

# --- Безопасный калькулятор (AST-парсер вместо eval) ---

ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Разрешены только числа")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise ValueError("Недопустимый оператор")
        return ALLOWED_BIN_OPS[op_type](
            _safe_eval_node(node.left),
            _safe_eval_node(node.right),
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("Недопустимый унарный оператор")
        return ALLOWED_UNARY_OPS[op_type](_safe_eval_node(node.operand))

    raise ValueError("Недопустимое выражение")

@tool("calculator", "безопасно считает арифметические выражения")
def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval_node(tree))

# --- Файловые инструменты ---

@tool("list_directory", "показывает файлы и папки в директории")
def list_directory(path: str = ".") -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return f"Путь не найден или не директория: {path}"
    entries = []
    for name in sorted(os.listdir(path))[:200]:
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            entries.append(f"[DIR]  {name}/")
        else:
            entries.append(f"[FILE] {name}")
    return "\n".join(entries) if entries else "Директория пуста"

@tool("read_file", "читает текстовый файл")
def read_file(path: str, max_chars: int = 8000) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return f"Файл не найден: {path}"
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read(max_chars + 1)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n... [файл обрезан] ..."
    return content

@tool("search_in_file", "ищет текст в файле")
def search_in_file(path: str, query: str, max_results: int = 20) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return f"Файл не найден: {path}"
    if not query:
        return "Не указан текст для поиска"
    results = []
    query_lower = query.lower()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if query_lower in line.lower():
                results.append(f"{line_number}: {line.rstrip()}")
                if len(results) >= max_results:
                    results.append(f"... [показаны первые {max_results}] ...")
                    break
    return "\n".join(results) if results else f"Ничего не найдено: {query}"

# ====================================================================
# НОВЫЕ ИНСТРУМЕНТЫ ГЛАВЫ 4
# ====================================================================

# --- Запись в файл ---

@tool("write_file", "записывает текст в файл (создаёт или перезаписывает)")
def write_file(path: str, content: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        return f"Директория не найдена: {directory}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Записано {len(content)} символов в {path}"

# --- Поиск по всей директории (рекурсивно) ---

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}

@tool("search_in_directory", "ищет текст во всех файлах директории (рекурсивно)")
def search_in_directory(path: str = ".", query: str = "", max_results: int = 30) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return f"Директория не найдена: {path}"
    if not query:
        return "Не указан текст для поиска"

    results = []
    query_lower = query.lower()

    for root, dirs, files in os.walk(path):
        # Пропускаем служебные директории
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in sorted(files):
            if len(results) >= max_results:
                break
            full_path = os.path.join(root, name)
            try:
                matches_in_file = 0
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    for line_number, line in enumerate(f, start=1):
                        if query_lower in line.lower():
                            relative = os.path.relpath(full_path, path)
                            results.append(f"{relative}:{line_number}: {line.rstrip()}")
                            matches_in_file += 1
                            # Не больше 3 совпадений на файл
                            if matches_in_file >= 3:
                                break
            except OSError:
                continue

        if len(results) >= max_results:
            break

    if not results:
        return f"Ничего не найдено: {query}"
    if len(results) >= max_results:
        results.append(f"... [показаны первые {max_results}] ...")
    return "\n".join(results)

# --- Выполнение команд с подтверждением пользователя ---

COMMAND_TIMEOUT = 30  # секунд

@tool("run_command", "выполняет команду в терминале (спрашивает подтверждение пользователя)")
def run_command(command: str) -> str:
    if not command:
        return "Не указана команда"

    # Обязательное подтверждение перед выполнением
    confirm = input(
        f"\n⚠️  Агент хочет выполнить команду:\n   {command}\n"
        f"Разрешить? [y/n]: "
    ).strip().lower()

    if confirm != "y":
        return "Команда отменена пользователем"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.strip()[:2000])
        if result.stderr:
            parts.append(f"[stderr] {result.stderr.strip()[:500]}")
        parts.append(f"[код возврата: {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Команда выполнялась дольше {COMMAND_TIMEOUT} сек и была остановлена"
    except Exception as e:
        return f"Ошибка выполнения команды: {e}"

# --- HTTP-запросы: агент ходит в интернет ---

@tool("http_get", "скачивает страницу по URL и возвращает её текст")
def http_get(url: str, max_chars: int = 4000) -> str:
    if not url.startswith(("http://", "https://")):
        return "URL должен начинаться с http:// или https://"
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "firstAgent/1.0 (tutorial agent)"}
        )
        if response.status_code != 200:
            return f"HTTP {response.status_code} для {url}"
        text = response.text[:max_chars]
        if len(response.text) > max_chars:
            text += "\n... [обрезано] ..."
        return text
    except requests.exceptions.Timeout:
        return f"Превышено время ожидания ответа от {url}"
    except Exception as e:
        return f"Ошибка HTTP-запроса: {e}"

# ====================================================================
# ПРИМЕР НОВОГО ПЛАГИНА: 5 строк — и инструмент доступен агенту
# ====================================================================

@tool("current_time", "возвращает текущие дату и время")
def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    print("Зарегистрированные инструменты:")
    for name, entry in TOOL_REGISTRY.items():
        print(f"  - {name}: {entry['description']}")
    print()
    print("Проверка calculator:", execute_tool("calculator", {"expression": "(15 * 7) + 3"}))
    print("Проверка current_time:", execute_tool("current_time", {}))
