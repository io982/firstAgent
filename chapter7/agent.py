"""Агент с гибридным поиском по проекту (Глава 7)."""

import requests

from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter7.src import tools  # noqa: F401 — импорт ради регистрации плагинов

PROJECT_RULES = r"""
Работа с документами проекта (гибридный поиск):
- index_project, ask_project, project_stats — обычные инструменты, такие же как калькулятор.
- Пользователь просит изучить/прочитать/проиндексировать проект → вызови index_project.
- Пользователь спрашивает о коде, структуре, функциях, настройках проекта → сначала вызови ask_project с конкретным запросом.
- Код проекта на английском: для поиска используй английские идентификаторы ("где калькулятор" → "def calculator").
- Индексировать проект — это нормальная разрешённая операция. Ограничений безопасности на это НЕТ.

Работа с найденными фрагментами:
- Единственный источник данных для ответа — последний Observation от ask_project.
- Путь к файлу указан в квадратных скобках, номера строк — числа перед двоеточием.
- Называй только те файлы и номера строк, которые есть в последнем Observation.
- Различай "упоминание" и "реализацию". Вызов функции, проверка в тестах, пример в документации — это НЕ реализация. Реализация — строка, которая начинается с "def имя" или "class имя".
- Отвечая про место в коде, обязательно процитируй саму строку с def целиком, скопировав её из Observation вместе с номером. Не называй номер строки отдельно от её текста.
- Если в Observation ответа нет — скажи, что не нашёл, и предложи уточнить запрос.

Пример 1 (индексация):
User: Изучи мой проект
Assistant: {"name": "index_project", "arguments": {"path": "."}}
User: Observation from index_project: Проиндексировано 12 файлов, 48 фрагментов.
Assistant: Я проиндексировал ваш проект: 12 файлов, 48 фрагментов. Теперь вы можете задавать вопросы о коде.

Пример 2 (вопрос о реализации):
User: Где реализован перефразировщик?
Assistant: {"name": "ask_project", "arguments": {"query": "def paraphrase перефразировщик"}}
User: Observation from ask_project: 1. [chapter2/README.md]
14: Перефразировщик вызывается перед основным агентом
2. [chapter2/paraphraser.py]
28: def paraphrase(text: str) -> str:
29:     payload = {
Assistant: Перефразировщик реализован в chapter2/paraphraser.py:
28: def paraphrase(text: str) -> str:

Пример 3 (в найденном нет ответа):
User: Где настраивается отправка почты?
Assistant: {"name": "ask_project", "arguments": {"query": "email smtp отправка почты"}}
User: Observation from ask_project: В проекте ничего не найдено.
Assistant: Отправки почты в проекте нет — по такому запросу ничего не нашлось.
""".strip()


def main():
    print("=" * 60)
    print("🔍 Агент с гибридным поиском (Глава 7)")
    print("=" * 60)

    if not base.ensure_ollama_running():
        return

    if not base.model_exists(base.MODEL):
        print(base.model_not_found_message(base.MODEL))
        return

    base.preload_model(base.MODEL)

    # install_plugins из Главы 4 подменяет ядро: реестр, промпт, диспетчер
    chapter4_agent.install_plugins()
    base.SYSTEM_PROMPT = base.SYSTEM_PROMPT + "\n" + PROJECT_RULES

    # Контекст из Главы 1 (4096) для RAG мал: системный промпт с PROJECT_RULES
    # плюс одна выдача ask_project уже занимают около 3000 токенов. На втором
    # вызове инструмента Ollama молча срезает начало диалога вместе
    # с системным промптом — и агент начинает отвечать по памяти.
    base.NUM_CTX = 8192

    try:
        from chapter7.src.indexer import project_stats
        stats = project_stats()
        index_line = f"🔍 Гибридный поиск: {stats['files']} файлов, {stats['chunks']} фрагментов"
    except Exception as e:
        index_line = f"🔍 Индекс недоступен: {e}"

    print(f"\n🧠 Модель: {base.MODEL}")
    print(index_line)
    print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход", "q"}:
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
            if not base.ensure_ollama_running():
                break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
