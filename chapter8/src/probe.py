"""Определение формата вызова инструментов пробным запросом.

Замер 20 августа 2026 показал, почему нельзя доверять флагу `tools`
из `ollama show`:

    qwen2_5coder3b_q5   tools заявлен  ->  tool_calls ПУСТО, вызов в content
    qwen2.5-coder:7b    tools заявлен  ->  tool_calls ПУСТО, вызов в content
    llama3.1:8b         tools заявлен  ->  tool_calls ЗАПОЛНЕН, content пуст

Поэтому формат определяется не документацией, а одним запросом при старте.
Стоит это пары секунд и снимает целый класс загадочных поломок.
"""

import json
import re

import requests

from chapter1 import agent as base

# Форматы, которые агент должен уметь различать
FORMAT_NATIVE = "native"    # модель возвращает message.tool_calls
FORMAT_TEXT = "text"        # модель кладёт JSON-вызов в message.content
FORMAT_UNKNOWN = "unknown"  # ни то, ни другое — вызвать инструмент не удалось

# Инструмент, существующий только ради пробы. В реестр он не попадает:
# проба должна работать до того, как агент к чему-либо подключился.
PROBE_TOOL_NAME = "probe_add"
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": PROBE_TOOL_NAME,
        "description": "складывает два целых числа",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
PROBE_QUESTION = "Сложи 2 и 3, обязательно вызвав инструмент probe_add."


def _mentions_probe_tool(content: str) -> bool:
    """Ищет в тексте JSON-вызов пробного инструмента.

    Намеренно не используем парсер Главы 1: он сверяется с KNOWN_TOOLS,
    а проба должна работать независимо от того, что сейчас в реестре.
    """
    if not content or PROBE_TOOL_NAME not in content:
        return False

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            obj, _ = decoder.raw_decode(content, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        candidates = [obj, obj.get("function")]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("name") == PROBE_TOOL_NAME:
                return True
    return False


def probe_tool_format(model: str, timeout: int = 120) -> str:
    """Выясняет, каким форматом модель отвечает на запрос с `tools`.

    Возвращает одну из констант FORMAT_*. Сетевые ошибки не глушим:
    если Ollama недоступна, агенту всё равно нечего делать.
    """
    response = requests.post(
        base.OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROBE_QUESTION}],
            "tools": [PROBE_TOOL],
            "stream": False,
            "keep_alive": base.KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_ctx": 2048},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    message = response.json().get("message", {})

    if message.get("tool_calls"):
        return FORMAT_NATIVE
    if _mentions_probe_tool(message.get("content", "")):
        return FORMAT_TEXT
    return FORMAT_UNKNOWN


def describe_format(model: str, detected: str) -> str:
    """Человекочитаемая строка для вывода при старте агента."""
    if detected == FORMAT_NATIVE:
        return f"🔌 {model}: нативные tool_calls — вызовы приходят структурой"
    if detected == FORMAT_TEXT:
        return f"📝 {model}: текстовый формат — вызовы придётся вылавливать из ответа"
    return (f"⚠️ {model}: инструмент не вызван ни одним из способов. "
            "Агент будет работать, но инструменты, скорее всего, не заработают")
