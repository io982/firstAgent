"""Глава 4. Система плагинов: реестр инструментов агента.

Каждый инструмент — это обычная функция с декоратором @tool.
Чтобы добавить новый инструмент, достаточно 5 строк:

    @tool("weather", "Возвращает погоду для города.")
    def weather(city: str) -> str:
        return f"В городе {city} солнечно, +25°C"

Ядро агента (chapter4/agent.py) само подхватит новый плагин.
"""

import ast
import operator
import os
from datetime import datetime


# ====================================================================
# РЕЕСТР ИНСТРУМЕНТОВ
# ====================================================================

# name -> {"func": callable, "description": str}
TOOL_REGISTRY = {}


def tool(name: str, description: str):
    """Декоратор: регистрирует функцию как инструмент агента."""
    def decorator(func):
        TOOL_REGISTRY[name] = {"func": func, "description": description}
        return func
    return decorator


def known_tools() -> set:
    """Множество имён зарегистрированных инструментов."""
    return set(TOOL_REGISTRY)


def execute_tool(name: str, args: dict) -> str:
    """Вызывает инструмент по имени, передавая аргументы в функцию."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Неизвестный инструмент: {name}"
    try:
        return str(entry["func"](**args))
    except TypeError as e:
        return f"Неверные аргументы для инструмента {name}: {e}"
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


def render_tools_for_prompt() -> str:
    """Собирает нумерованный список инструментов для системного промпта."""
    lines = []
    for number, (name, entry) in enumerate(TOOL_REGISTRY.items(), 1):
        lines.append(f"{number}. {name} — {entry['description']}")
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
    with open(path, "r", encoding="utf-8", errors="replace") as f:
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
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if query_lower in line.lower():
                results.append(f"{line_number}: {line.rstrip()}")
                if len(results) >= max_results:
                    results.append(f"... [показаны первые {max_results}] ...")
                    break
    return "\n".join(results) if results else f"Ничего не найдено: {query}"


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
