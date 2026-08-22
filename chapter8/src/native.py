"""Ядро с поддержкой двух форматов вызова инструментов.

Отличие от Главы 1 в трёх местах:

1. Запрос уходит с полем `tools` — описанием инструментов в JSON Schema.
2. Ответ разбирается двумя способами: сначала `message.tool_calls`,
   при пустом поле — текстовым парсером Главы 1.
3. Результат инструмента возвращается сообщением с ролью `tool`, если
   формат нативный, и как раньше — с ролью `user` и префиксом
   "Observation from", если текстовый.

Ловушка, ради которой глава и написана: у модели с нативным форматом при
заполненном `tool_calls` поле `content` ПУСТОЕ. Ядро Главы 1 в этом случае
не найдёт вызова, решит, что перед ним финальный ответ, и вернёт
пользователю пустую строку — молча и именно на той модели, которая
формат поддерживает.
"""

import requests

from chapter1 import agent as base

from .probe import FORMAT_NATIVE, FORMAT_TEXT
from .schema import registry_to_schemas

# Системный промпт Главы 4 объясняет модели, как выводить вызов инструмента
# текстом: "ответь ТОЛЬКО JSON-вызовом". При нативном формате это второй
# протокол поверх первого — и модель начинает делать обе вещи сразу: вызывает
# инструмент структурой, а следом повторяет вызов, потому что промпт просил
# JSON. Замер: с промптом Главы 4 llama3.1 вызывала calculator дважды подряд
# с одинаковыми аргументами, с промптом ниже — один раз.
#
# Описания инструментов здесь тоже не нужны: они уходят отдельно, в поле tools.
NATIVE_SYSTEM_PROMPT = """
Ты — автономный агент-ассистент для разработчика.

Инструменты доступны тебе напрямую: система вызовет их сама, когда ты
об этом попросишь. Не описывай вызов текстом и не выводи JSON.

Правила:
1. Нужен инструмент — вызови его.
2. Получив результат, используй его в ответе пользователю.
3. Не вызывай один и тот же инструмент дважды с теми же аргументами.
4. Когда готов дать финальный ответ — отвечай обычным текстом.
5. Никогда не выдумывай результаты инструментов.
""".strip()


def system_prompt_for(tool_format: str) -> str:
    """Системный промпт под выбранный протокол.

    Дополнительные правила глав (память, работа с проектом) дописываются
    к промпту после его установки, поэтому при нативном формате берём
    всё, что идёт после базовой части.
    """
    if tool_format != FORMAT_NATIVE:
        return base.SYSTEM_PROMPT
    return NATIVE_SYSTEM_PROMPT


def request_model_with_tools(messages: list, schemas: list | None = None) -> dict:
    """Запрос к модели. Если переданы схемы — уходит поле `tools`."""
    payload = {
        "model": base.MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": base.KEEP_ALIVE,
        "options": {"temperature": 0.1, "num_ctx": base.NUM_CTX},
    }
    if schemas:
        payload["tools"] = schemas

    response = requests.post(base.OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    message = response.json().get("message", {})
    message.setdefault("content", "")
    return message


def extract_native_calls(message: dict) -> list:
    """Достаёт вызовы из `message.tool_calls` и приводит к виду Главы 1.

    Формат Ollama:  {"function": {"name": ..., "arguments": {...}}}
    Формат курса:   {"name": ..., "arguments": {...}}
    """
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        calls.append({"name": name, "arguments": function.get("arguments", {})})
    return calls


def extract_calls_any_format(message: dict) -> tuple:
    """Возвращает (вызовы, каким_форматом_они_пришли).

    Нативный формат проверяется первым: если модель его использует,
    в `content` вызова не будет вовсе.
    """
    native = extract_native_calls(message)
    if native:
        return native, FORMAT_NATIVE
    return base.extract_tool_calls(message.get("content", "")), FORMAT_TEXT


def _observation_message(tool_name: str, result: str, tool_format: str) -> dict:
    """Сообщение с результатом инструмента — в формате, который ждёт модель."""
    if tool_format == FORMAT_NATIVE:
        return {"role": "tool", "tool_name": tool_name, "content": result}
    return {"role": "user", "content": f"Observation from {tool_name}: {result}"}


def ask_agent_dual(user_task: str, tool_format: str) -> str:
    """ReAct-цикл Главы 1, умеющий оба формата вызова инструментов."""
    schemas = registry_to_schemas() if tool_format == FORMAT_NATIVE else None

    messages = [
        {"role": "system", "content": system_prompt_for(tool_format)},
        {"role": "user", "content": user_task},
    ]

    for iteration in range(1, base.MAX_ITERATIONS + 1):
        if base.VERBOSE:
            print(f"\n--- Итерация {iteration} ---")

        message = request_model_with_tools(messages, schemas)
        content = message.get("content", "")
        calls, used_format = extract_calls_any_format(message)

        if base.VERBOSE:
            print(f"[Model] формат ответа: {used_format if calls else 'без вызовов'}")
            print(content[:500] if content else "[content пуст]")

        if not calls:
            # Пустой ответ без вызова — это сбой, а не ответ пользователю.
            # Именно так выглядит нативный формат, разобранный старым ядром.
            if not content.strip():
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": ("Ответ пришёл пустым. Вызови подходящий инструмент "
                                "или ответь текстом."),
                })
                continue

            unknown = base.extract_unknown_tool_names(content)
            if unknown:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Инструмента {unknown[0]} не существует. "
                        f"Доступные инструменты: {', '.join(sorted(base.KNOWN_TOOLS))}. "
                        "Вызови подходящий из списка или ответь пользователю обычным текстом."
                    ),
                })
                continue

            return content.strip()

        # Ответ ассистента возвращаем в историю в том же виде, в каком получили:
        # при нативном формате там пустой content и заполненный tool_calls.
        messages.append({
            "role": "assistant",
            "content": content,
            **({"tool_calls": message["tool_calls"]} if used_format == FORMAT_NATIVE else {}),
        })

        for call in calls:
            tool_name = call.get("name", "")
            tool_args = base.normalize_arguments(call.get("arguments", {}))

            if base.VERBOSE:
                print(f"[Agent] Вызываю инструмент: {tool_name}")
                print(f"[Agent] Аргументы: {tool_args}")

            result = base.execute_tool(tool_name, tool_args)

            if base.VERBOSE:
                print(f"[Tool] Результат: {result[:300]}".replace("\n", " "))

            messages.append(_observation_message(tool_name, result, used_format))

    return "Агент достиг лимита итераций и не смог завершить задачу."


__all__ = ["base", "FORMAT_NATIVE", "FORMAT_TEXT", "NATIVE_SYSTEM_PROMPT",
           "system_prompt_for", "request_model_with_tools",
           "extract_native_calls", "extract_calls_any_format", "ask_agent_dual"]
