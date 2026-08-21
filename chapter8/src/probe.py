"""Определение формата вызова инструментов пробным запросом.

Замер 20 августа 2026 показал, почему нельзя доверять флагу `tools`
из `ollama show`:

    qwen2_5coder3b_q5   tools заявлен  →  tool_calls ПУСТО, вызов в content
    qwen2.5-coder:7b    tools заявлен  →  tool_calls ПУСТО, вызов в content
    llama3.1:8b         tools заявлен  →  tool_calls ЗАПОЛНЕН, content пуст

Поэтому формат определяется не документацией, а одним запросом при старте.

СТАТУС: каркас Главы 8. Реализации нет.
"""

# Форматы, которые агент должен уметь различать
FORMAT_NATIVE = "native"    # модель возвращает message.tool_calls
FORMAT_TEXT = "text"        # модель кладёт JSON-вызов в message.content
FORMAT_UNKNOWN = "unknown"  # ни то, ни другое — вызов инструментов недоступен


def probe_tool_format(model: str, timeout: int = 60) -> str:
    """Выясняет, каким форматом модель отвечает на запрос с `tools`.

    Отправляет один короткий запрос с заведомо подходящим инструментом
    и смотрит, куда попал вызов. Возвращает одну из констант FORMAT_*.
    """
    raise NotImplementedError("Глава 8: пробный запрос и разбор ответа")


def describe_format(model: str, detected: str) -> str:
    """Человекочитаемая строка для вывода при старте агента."""
    raise NotImplementedError("Глава 8: сообщение о выбранном формате")
