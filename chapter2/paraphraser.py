import requests

from chapter1 import agent  # импортируем нашего первого агента

# Режим вывода: True = видно все шаги, False = только результат
VERBOSE = False

# Передаём режим первому агенту
agent.VERBOSE = VERBOSE


PARAPHRASE_PROMPT = """
Ты — перефразировщик запросов для ИИ-агента.

Твоя задача — перефразировать запрос пользователя, сделав его максимально чётким, конкретным и пригодным для автоматического выполнения.

Правила:
1. Сохраняй исходный смысл.
2. НИКОГДА не меняй числа, арифметические выражения, пути к файлам, имена файлов и другие конкретные данные.
3. Убирай слова-паразиты и лишние вводные конструкции.
4. Если запрос размытый, добавь конкретику, но не выдумывай данные, которых нет.
5. Если запрос уже чёткий — верни его без изменений.
6. Отвечай ТОЛЬКО перефразированным текстом.
7. Без кавычек, без префиксов вроде "Перефразированный запрос:", без пояснений.
""".strip()


def paraphrase(text: str) -> str:
    """Перефразирует запрос пользователя через LLM."""
    payload = {
        "model": agent.MODEL,
        "messages": [
            {"role": "system", "content": PARAPHRASE_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 2048
        }
    }

    response = requests.post(
        agent.OLLAMA_URL,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    result = response.json().get("message", {}).get("content", "").strip()

    # Чистка на случай, если модель всё же добавила лишнее
    result = result.strip('"').strip("'").strip()

    lower = result.lower()
    for prefix in ["перефразированный запрос:", "перефразировано:", "ответ:"]:
        if lower.startswith(prefix):
            result = result[len(prefix):].strip()

    return result


def main():
    print("=" * 60)
    print("🔁 Мульти-агентная система")
    print("   Перефразировщик → ReAct-агент")
    print("=" * 60)

    if not agent.ensure_ollama_running():
        print("❌ Не удалось запустить Ollama.")
        return

    print(f"\n🔍 Проверяю модель '{agent.MODEL}'...")
    if not agent.model_exists(agent.MODEL):
        print(agent.model_not_found_message(agent.MODEL))
        return

    agent.preload_model(agent.MODEL)

    print(f"\n🤖 Система готова. Модель: {agent.MODEL}")
    print("🛑 q / exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "выход", "q"}:
                print("👋 Пока!")
                break

            # Шаг 1: Перефразирование
            if VERBOSE:
                print(f"\n[Paraphraser] Исходный запрос: {user_input}")

            paraphrased = paraphrase(user_input)

            if VERBOSE:
                print(f"[Paraphraser] Перефразировано: {paraphrased}")
            else:
                print(f"🔄 Перефразировано: {paraphrased}")

            # Шаг 2: Передача первому агенту
            if not VERBOSE:
                print("⏳ Агент выполняет...")

            answer = agent.ask_agent(paraphrased)

            print("\n🤖 Агент >")
            print(answer)
            print()

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
