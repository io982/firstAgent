import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем ядро из Главы 1 (только request_model)
# Импортируем утилиты запуска из Главы 1
from chapter1.agent import ensure_ollama_running, preload_model, request_model

# Импортируем новые возможности из Главы 2
from chapter2.src.tools import execute_tool, extract_tool_calls, get_all_tools_schemas

# 1. Динамическая генерация описания инструментов из кода!
# ИСПРАВЛЕНО: get_all_tools_schemas() возвращает сами схемы, поэтому обращаемся напрямую к 'function'
TOOLS_DESCRIPTION = "\n".join(
    f"- {info['function']['name']}: {info['function']['description']}"
    for info in get_all_tools_schemas()
)

# 2. Системный промпт теперь всегда синхронизирован с реальным кодом
SYSTEM_PROMPT = f"""Ты — автономный AI-ассистент. У тебя есть доступ к следующим инструментам:

{TOOLS_DESCRIPTION}

Правила вызова:
1. Если для ответа нужен инструмент, верни ТОЛЬКО валидный JSON в формате:
{{"name": "имя_инструмента", "arguments": {{"параметр": "значение"}}}}
2. Строго следуй именам параметров, указанным в описании инструментов выше.
3. После получения результата (Observation) проанализируй его и либо вызови следующий инструмент, либо дай финальный ответ пользователю обычным текстом.
""".strip()

def ask_agent(user_input: str, max_iterations: int = 5) -> str:
    """Цикл ReAct, использующий новый Tool API."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    for i in range(max_iterations):
        print(f"\n--- Итерация {i+1} ---")

        response = request_model(messages)
        # ИСПРАВЛЕНО: request_model может возвращать строку напрямую или словарь
        if isinstance(response, str):
            content = response
        else:
            content = response.get("content", "")

        print(f"🤖 Модель:\n{content}")

        # Используем наш парсер из tools.py
        tool_calls = extract_tool_calls(content)

        if tool_calls:
            messages.append({"role": "assistant", "content": content})

            # Обрабатываем все вызовы, которые модель вернула в этом шаге
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})

                # Если модель вернула строку вместо словаря, пытаемся её распарсить
                if isinstance(arguments, str):
                    try:
                        import json
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"expression": arguments}  # Fallback для calculator

                print(f"🛠️ Вызов: {tool_name} | Аргументы: {arguments}")

                # Единый диспетчер сам разберется с валидацией и выполнением
                observation = execute_tool(tool_name, arguments)
                print(f"👁️ Результат: {observation}")

                messages.append({"role": "user", "content": f"Observation from {tool_name}: {observation}"})
        else:
            # Нет вызова инструмента = финальный ответ
            return content.strip()

    return "⚠️ Превышен лимит итераций."

if __name__ == "__main__":

    # Гарантируем, что Ollama запущена
    if not ensure_ollama_running():
        print("\n❌ Не удалось запустить Ollama. Завершение работы.")
        sys.exit(1)

    # Прогреваем модель для мгновенного первого ответа
    preload_model()

    print("🤖 Агент с Tool API готов. Введите 'выход' для завершения.")
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            break
        if not user_input.strip():
            continue

        answer = ask_agent(user_input)
        print(f"\n✅ Ответ: {answer}")
