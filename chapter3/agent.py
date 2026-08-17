"""Глава 3. Агент с памятью между запросами.

В Главе 1 каждый запрос начинался с чистого листа: список messages
создавался заново и уничтожался после ответа. Здесь история диалога
сохраняется в conversation_history и передаётся модели при каждом запросе.

Ключевые моменты:
- conversation_history хранит пары user/assistant между запросами;
- история обрезается до MAX_HISTORY сообщений, чтобы не раздувать контекст;
- num_ctx увеличен до 8192, чтобы в окно помещалась история диалога;
- команды /reset, /clear, «очистить» сбрасывают память.
"""

import requests

from chapter1 import agent


# Сколько сообщений хранить в памяти (сообщения user + assistant)
MAX_HISTORY = 10

# Размер контекста: больше, чем в Главе 1, чтобы влезала история диалога
NUM_CTX = 8192

# История диалога между запросами
conversation_history = []


def request_model(messages: list) -> dict:
    """Запрос к модели с увеличенным контекстом под историю диалога."""
    payload = {
        "model": agent.MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": agent.KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    }
    response = requests.post(agent.OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    message = response.json().get("message", {})
    if "content" not in message:
        message["content"] = ""
    return message


def ask_agent_with_memory(user_task: str) -> str:
    """ReAct-цикл как в Главе 1, но с историей диалога."""
    conversation_history.append({"role": "user", "content": user_task})

    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    messages.extend(conversation_history)

    final_answer = "Агент достиг лимита итераций и не смог завершить задачу."

    for iteration in range(1, agent.MAX_ITERATIONS + 1):
        if agent.VERBOSE:
            print(f"\n--- Итерация {iteration} ---")

        assistant_message = request_model(messages)
        content = assistant_message.get("content", "")

        if agent.VERBOSE:
            print("[Model raw answer]")
            print(content[:500])

        tool_calls = agent.extract_tool_calls(content)

        if not tool_calls:
            final_answer = content.strip()
            break

        messages.append({"role": "assistant", "content": content})

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = agent.normalize_arguments(tool_call.get("arguments", {}))

            if agent.VERBOSE:
                print(f"[Agent] Вызываю инструмент: {tool_name}")
                print(f"[Agent] Аргументы: {tool_args}")

            tool_result = agent.execute_tool(tool_name, tool_args)

            if agent.VERBOSE:
                preview = tool_result[:300].replace("\n", " ")
                print(f"[Tool] Результат: {preview}")

            messages.append({
                "role": "user",
                "content": f"Observation from {tool_name}: {tool_result}",
            })

    # Сохраняем финальный ответ в историю
    conversation_history.append({"role": "assistant", "content": final_answer})

    # Обрезаем историю, чтобы не раздувать контекст
    if len(conversation_history) > MAX_HISTORY:
        conversation_history[:] = conversation_history[-MAX_HISTORY:]

    return final_answer


def main():
    print("=" * 60)
    print("🧠 Агент с памятью (Глава 3)")
    print("=" * 60)

    if not agent.ensure_ollama_running():
        print("\n❌ Не удалось запустить Ollama. Завершение работы.")
        return

    print(f"\n🔍 Проверяю модель '{agent.MODEL}'...")
    if not agent.model_exists(agent.MODEL):
        print(agent.model_not_found_message(agent.MODEL))
        return
    print("✅ Модель готова.")

    agent.preload_model(agent.MODEL)

    print(f"\n🤖 Агент готов. Модель: {agent.MODEL}")
    print("🧹 /reset, /clear или «очистить» — очистить память.")
    print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Пока!")
                break

            if user_input.lower() in {"/reset", "/clear", "очистить"}:
                conversation_history.clear()
                print("🧹 История очищена.\n")
                continue

            if not agent.VERBOSE:
                print("⏳ Думаю...")

            answer = ask_agent_with_memory(user_input)

            print("\n🤖 Агент >")
            print(answer)
            print(f"   [в памяти: {len(conversation_history)} сообщений]\n")

        except KeyboardInterrupt:
            print("\n⚠️ Остановлено.")
            break

        except requests.exceptions.ConnectionError:
            print("\n⚠️ Связь потеряна. Восстанавливаю...")
            if agent.ensure_ollama_running():
                print("✅ Восстановлено. Повтори запрос.")
            else:
                print("❌ Не удалось восстановить.")
                break

        except Exception as e:
            print(f"\nОшибка: {e}")


if __name__ == "__main__":
    main()
