"""Ядро с поддержкой двух форматов вызова инструментов.

Отличие от Главы 1: запрос уходит с полем `tools`, а ответ разбирается
двумя способами — сначала ищем `message.tool_calls`, при пустом поле
откатываемся на текстовый парсер Главы 1.

Ловушка, ради которой глава и написана: у модели с нативным форматом
при заполненном `tool_calls` поле `content` ПУСТОЕ. Ядро Главы 1 в этом
случае не найдёт вызова, решит, что перед ним финальный ответ, и вернёт
пользователю пустую строку. Проверка на пустой content обязательна.

СТАТУС: каркас Главы 8. Реализации нет.
"""

from chapter1 import agent as base


def request_model_with_tools(messages: list, schemas: list) -> dict:
    """Запрос к модели с описанием инструментов в поле `tools`."""
    raise NotImplementedError("Глава 8: запрос с tools")


def extract_native_calls(message: dict) -> list:
    """Достаёт вызовы из `message.tool_calls` и приводит их к виду Главы 1.

    Формат Ollama:
        {"function": {"name": ..., "arguments": {...}}}
    Формат ядра курса:
        {"name": ..., "arguments": {...}}
    """
    raise NotImplementedError("Глава 8: разбор native tool_calls")


def extract_calls_any_format(message: dict) -> tuple:
    """Возвращает (вызовы, использованный_формат).

    Порядок: сначала нативный, затем текстовый парсер Главы 1
    (`base.extract_tool_calls`).
    """
    raise NotImplementedError("Глава 8: разбор в любом из двух форматов")


def ask_agent_dual(user_task: str, tool_format: str) -> str:
    """ReAct-цикл Главы 1, умеющий оба формата.

    Дополнительно к циклу Главы 1:
    - пустой ответ без вызова инструмента считается сбоем, а не ответом;
    - результат инструмента при нативном формате возвращается сообщением
      с ролью "tool", а не "user".
    """
    raise NotImplementedError("Глава 8: цикл с поддержкой двух форматов")


__all__ = ["base", "request_model_with_tools", "extract_native_calls",
           "extract_calls_any_format", "ask_agent_dual"]
