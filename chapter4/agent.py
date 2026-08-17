"""Глава 4. Агент с системой плагинов.

Ядро агента взято из Главы 1 без переписывания: мы лишь подменяем
список инструментов, системный промпт и диспетчер инструментов на
версии из реестра плагинов (chapter4/src/tools.py).

Чтобы добавить агенту новый инструмент, откройте chapter4/src/tools.py
и добавьте функцию с декоратором @tool — 5 строк кода.
"""

import requests

from chapter1 import agent as base
from chapter4.src import tools

SYSTEM_PROMPT_TEMPLATE = """
Ты — автономный агент-ассистент для разработчика.

У тебя есть инструменты:
{tools}

Формат вызова инструмента:
{{"name": "имя_инструмента", "arguments": {{...}}}}

Пример:

User: Посчитай 2+2
Assistant: {{"name": "calculator", "arguments": {{"expression": "2+2"}}}}
User: Observation from calculator: 4
Assistant: 2 + 2 = 4

Правила:
1. Если нужен инструмент, ответь ТОЛЬКО JSON-вызовом или несколькими JSON-вызовами.
2. Не добавляй пояснения до или после JSON при вызове инструмента.
3. После получения Observation проанализируй результат.
4. Если нужно вызвать ещё инструмент — снова верни JSON.
5. Когда готов дать финальный ответ пользователю, ответь обычным текстом без JSON.
6. Никогда не выдумывай результаты инструментов.
""".strip()


def install_plugins():
    """Подключает реестр плагинов к ядру агента из Главы 1."""
    base.KNOWN_TOOLS = tools.known_tools()
    base.SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
        tools=tools.render_tools_for_prompt()
    )
    base.execute_tool = tools.execute_tool


def main():
    print("=" * 60)
    print("🔌 Агент с системой плагинов (Глава 4)")
    print("=" * 60)

    install_plugins()

    print(f"🔌 Загружено плагинов: {len(tools.TOOL_REGISTRY)}")
    for name, entry in tools.TOOL_REGISTRY.items():
        print(f"   - {name}: {entry['description']}")

    if not base.ensure_ollama_running():
        print("\n❌ Не удалось запустить Ollama. Завершение работы.")
        return

    print(f"\n🔍 Проверяю модель '{base.MODEL}'...")
    if not base.model_exists(base.MODEL):
        print(base.model_not_found_message(base.MODEL))
        return
    print("✅ Модель готова.")

    base.preload_model(base.MODEL)

    print(f"\n🤖 Агент готов. Модель: {base.MODEL}")
    print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Пока!")
                break

            if not base.VERBOSE:
                print("⏳ Думаю...")

            answer = base.ask_agent(user_input)

            print("\n🤖 Агент >")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\n⚠️ Остановлено.")
            break

        except requests.exceptions.ConnectionError:
            print("\n⚠️ Связь потеряна. Восстанавливаю...")
            if base.ensure_ollama_running():
                print("✅ Восстановлено. Повтори запрос.")
            else:
                print("❌ Не удалось восстановить.")
                break

        except Exception as e:
            print(f"\nОшибка: {e}")


# ============================================================
# ПРОБРОС БАЗОВЫХ ФУНКЦИЙ ИЗ ГЛАВЫ 1
# ============================================================
# Чтобы следующие главы вызывали одну точку входа, а не выясняли,
# что взять из Главы 1, а что из Главы 4.
#
# Пробрасываем только функции. Конфигурацию (MODEL, VERBOSE, NUM_CTX)
# главы читают напрямую из chapter1: скопированное сюда значение было бы
# снимком на момент импорта и разошлось бы с оригиналом при любой правке.

ensure_ollama_running = base.ensure_ollama_running
model_exists = base.model_exists
model_not_found_message = base.model_not_found_message
preload_model = base.preload_model
ask_agent = base.ask_agent


if __name__ == "__main__":
    main()
