import inspect
import json
import os
import re
from collections.abc import Callable
from typing import Any

# Глобальный реестр инструментов
TOOL_REGISTRY: dict[str, dict[str, Any]] = {}

def tool(func: Callable) -> Callable:
    """Декоратор для регистрации функции как инструмента агента."""
    description = func.__doc__.strip().split('\n')[0] if func.__doc__ else "Нет описания"

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        schema["function"]["parameters"]["properties"][name] = {
            "type": "string",
            "description": f"Параметр '{name}'"
        }
        if param.default == inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(name)

    TOOL_REGISTRY[func.__name__] = {
        "schema": schema,
        "function": func
    }
    return func

# === РЕАЛЬНЫЕ ИНСТРУМЕНТЫ ===

@tool
def calculator(expression: str) -> str:
    """Безопасно вычисляет результат простого математического выражения."""
    try:
        safe_dict = {"__builtins__": {}}
        return str(eval(expression, safe_dict, {}))
    except Exception as e:
        return f"Ошибка вычисления: {e}"

@tool
def read_file(path: str) -> str:
    """Читает содержимое текстового файла по указанному пути."""
    try:
        safe_path = os.path.normpath(path)
        if not os.path.exists(safe_path):
            return f"Файл не найден: {safe_path}"
        with open(safe_path, encoding='utf-8') as f:
            return f.read()[:2000]
    except Exception as e:
        return f"Ошибка чтения: {e}"

@tool
def get_weather(city: str) -> str:
    """Возвращает текущую погоду в указанном городе (имитация API)."""
    return f"Погода в {city}: +20°C, ясно. (Демо-данные)"

# === API ДЛЯ АГЕНТА ===

def get_all_tools_schemas() -> list[dict[str, Any]]:
    """Возвращает список JSON Schema всех зарегистрированных инструментов."""
    return [info["schema"] for info in TOOL_REGISTRY.values()]

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Единый диспетчер вызовов."""
    if tool_name not in TOOL_REGISTRY:
        return f"Ошибка: инструмент '{tool_name}' не найден. Доступные: {list(TOOL_REGISTRY.keys())}"

    func = TOOL_REGISTRY[tool_name]["function"]
    try:
        result = func(**arguments)
        return str(result)
    except TypeError as e:
        return f"Ошибка валидации аргументов: {e}. Проверь имена и количество параметров."
    except Exception as e:
        return f"Ошибка выполнения: {e}"

# === ПАРСИНГ ОТВЕТА МОДЕЛИ ===

def extract_tool_calls(text: str) -> list:
    """
    Извлекает вызовы инструментов из текста ответа модели.
    Ищет JSON-объекты вида {"name": "...", "arguments": {...}}.
    """
    calls = []

    # 1. Пытаемся найти JSON внутри markdown-блоков
    code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in code_blocks:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # 2. Если блоков нет, ищем любые JSON-объекты в тексте
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
            pos = start + end
        except json.JSONDecodeError:
            pos = start + 1

    return calls
