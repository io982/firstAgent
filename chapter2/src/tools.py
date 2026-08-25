import ast
import inspect
import json
import os
import re
import sys
from collections.abc import Callable
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Переиспользуем безопасный AST-калькулятор из Главы 1 вместо eval:
# eval с пустым __builtins__ обходится через (1).__class__.__base__.__subclasses__().
from chapter1.agent import _safe_eval_node

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
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval_node(tree))
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
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

def describe_tools() -> str:
    """Собирает описание всех инструментов для системного промпта.

    Читает реестр В МОМЕНТ ВЫЗОВА, а не при импорте модуля. Это важно:
    Глава 3 регистрирует свои инструменты позже, и если бы описание
    считалось один раз на уровне модуля, они бы в промпт не попали.

    В сигнатуре печатаются имена параметров — иначе модель узнаёт их
    только из few-shot примеров и начинает выдумывать свои.
    """
    lines = []
    for info in get_all_tools_schemas():
        fn = info["function"]
        params = ", ".join(fn["parameters"]["properties"].keys())
        lines.append(f"- {fn['name']}({params}): {fn['description']}")
    return "\n".join(lines)

def get_tool_parameters(tool_name: str) -> list[str]:
    """Возвращает имена параметров инструмента в порядке объявления."""
    if tool_name not in TOOL_REGISTRY:
        return []
    schema = TOOL_REGISTRY[tool_name]["schema"]["function"]["parameters"]
    return list(schema["properties"].keys())

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Единый диспетчер вызовов."""
    if tool_name not in TOOL_REGISTRY:
        return f"Ошибка: инструмент '{tool_name}' не найден. Доступные: {list(TOOL_REGISTRY.keys())}"

    func = TOOL_REGISTRY[tool_name]["function"]
    try:
        result = func(**arguments)
        return str(result)
    except TypeError as e:
        # Схема уже знает правильные имена — подсказываем их модели, чтобы
        # на следующей итерации она исправила вызов сама.
        expected = list(get_tool_parameters(tool_name))
        return (
            f"Ошибка валидации аргументов для '{tool_name}': {e}. "
            f"Ты передал: {list(arguments.keys())}. Ожидались строго: {expected}."
        )
    except Exception as e:
        return f"Ошибка выполнения: {e}"

# === CONSTRAINED DECODING: СХЕМА ОТВЕТА АГЕНТА ===

def build_response_schema() -> dict[str, Any]:
    """Собирает JSON Schema ответа агента по текущему содержимому реестра.

    Схема уходит в Ollama параметром `format`. Дальше модель генерирует не
    «текст, похожий на JSON», а токены, разрешённые грамматикой: невалидный
    JSON становится физически невозможен, а поле `name` ограничено enum'ом
    реально зарегистрированных инструментов — выдумать несуществующий
    инструмент модель тоже не может.

    Функция, а не константа — по той же причине, что и build_system_prompt():
    Глава 3 регистрирует свои инструменты позже, и снимок, снятый при импорте,
    их бы не увидел.

    Ответ всегда один объект — потому что шагов у агента ровно два:

        action = "tool_call"     -> заполнены name и arguments
        action = "final_answer"  -> заполнен answer
    """
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tool_call", "final_answer"],
            },
            "name": {
                "type": "string",
                "enum": list(TOOL_REGISTRY.keys()),
            },
            "arguments": {"type": "object"},
            "answer": {"type": "string"},
        },
        "required": ["action"],
    }

# === ПАРСИНГ ОТВЕТА МОДЕЛИ ===

def parse_agent_response(text: str) -> tuple[list, str | None]:
    """Разбирает ответ модели в пару (вызовы инструментов, финальный ответ).

    Порядок попыток важен:

    1. Ответ по схеме build_response_schema() — основной путь, когда
       constrained decoding включён.
    2. Свободный текст — фоллбэк через extract_tool_calls() для случаев,
       когда сервер не поддерживает `format` или он выключен вручную.

    Ровно один из элементов пары непустой.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None

    if isinstance(obj, dict) and "action" in obj:
        if obj["action"] == "tool_call" and obj.get("name"):
            return [{"name": obj["name"], "arguments": obj.get("arguments", {})}], None
        return [], (obj.get("answer") or "").strip()

    # Фоллбэк: сервер без constrained decoding или отключённая схема
    calls = extract_tool_calls(text)
    if calls:
        return calls, None
    return [], text.strip()

def extract_tool_calls(text: str) -> list:
    """
    Извлекает вызовы инструментов из свободного текста ответа модели.
    Ищет JSON-объекты вида {"name": "...", "arguments": {...}}.

    ⚠️ Путь для совместимости, а не основной. Пока схема не передаётся
    в `format`, разбор ответа остаётся угадыванием: модель может обернуть
    JSON в markdown, добавить пояснение до и после, сломать кавычки.
    Основной путь — parse_agent_response() поверх build_response_schema().
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
            pos = end
        except json.JSONDecodeError:
            pos = start + 1

    return calls
