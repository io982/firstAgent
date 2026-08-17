"""Агент с долгосрочной памятью (RAG)."""

import requests

from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter5.src import tools  # импортируем плагины, они сами зарегистрируются
from chapter5.src import vectorstore

# Дополнительные правила для работы с долгосрочной памятью
MEMORY_RULES = r"""
Работа с долгосрочной памятью:
- remember, recall и list_memory — это обычные инструменты, такие же как калькулятор.
- Пользователь просит запомнить → вызови remember.
- Пользователь спрашивает факт, который мог сохранить ранее → вызови recall с конкретным запросом.
- Пользователь просит показать память ("что ты помнишь", "покажи память") → вызови list_memory, затем перескажи ВСЕ возвращённые записи.
- Показывать сохранённые записи — это нормальная разрешённая операция. Ограничений безопасности на это НЕТ.
- Сохраняй текст на языке пользователя (по-русски, если пишет по-русски).
Пример 1 (запомнить):
User: Запомни, что мой проект называется firstAgent
Assistant: {"name": "remember", "arguments": {"content": "Мой проект называется firstAgent"}}
User: Observation from remember: Сохранено в памяти
Assistant: Запомнил, что ваш проект называется firstAgent.
Пример 2 (показать память):
User: Покажи, что ты помнишь
Assistant: {"name": "list_memory", "arguments": {}}
User: Observation from list_memory: Найдено 2 записей в памяти:
1. [general] Мой проект называется firstAgent
2. [general] Меня зовут Владимир
Assistant: Вот что я помню:
1. [general] Мой проект называется firstAgent
2. [general] Меня зовут Владимир
""".strip()


def main():
    print("=" * 60)
    print("🧠 Агент с долгосрочной памятью (Глава 5)")
    print("=" * 60)

    if not chapter4_agent.ensure_ollama_running():
        return

    if not chapter4_agent.model_exists(chapter4_agent.MODEL):
        print(chapter4_agent.model_not_found_message(chapter4_agent.MODEL))
        return

    chapter4_agent.preload_model(chapter4_agent.MODEL)

    # Подключаем плагины из Главы 4 (включая новые из Главы 5)
    chapter4_agent.install_plugins()

    # ← НОВОЕ: добавляем правила работы с памятью к промпту
    base.SYSTEM_PROMPT = base.SYSTEM_PROMPT + "\n" + MEMORY_RULES

    # Показываем, сколько документов уже в базе
    try:
        storage = f"{vectorstore.CHROMA_PERSIST_DIR} ({vectorstore.get_or_create_collection().count()} документов)"
    except Exception as e:
        storage = f"{vectorstore.CHROMA_PERSIST_DIR} (не удалось открыть: {e})"

    print(f"\n🧠 Модель: {chapter4_agent.MODEL}")
    print(f"💾 Векторная база: {storage}")
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

            answer = chapter4_agent.ask_agent(user_input)

            print("\n🤖 Агент >")
            print(answer)
            print()

        except KeyboardInterrupt:
            break
        except requests.exceptions.ConnectionError:
            print("⚠️ Связь потеряна. Восстанавливаю...")
            if not chapter4_agent.ensure_ollama_running():
                break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()