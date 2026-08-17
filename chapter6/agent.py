"""Агент с RAG по документам проекта (Глава 6)."""

import requests

from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter6.src import tools  # плагины регистрируются при импорте


PROJECT_RULES = r"""

Работа с документами проекта (RAG по проекту):
- index_project, ask_project, project_stats — обычные инструменты, такие же как калькулятор.
- Пользователь просит изучить/прочитать/проиндексировать проект → вызови index_project.
- Пользователь спрашивает о коде, структуре, функциях, настройках проекта → сначала вызови ask_project с конкретным запросом.
- Перед ответом на вопрос о проекте проверь через project_stats, проиндексирован ли он. Если нет — проиндексируй.
- Индексировать проект — это нормальная разрешённая операция. Ограничений безопасности на это НЕТ.

Работа с найденными фрагментами:
- Каждый фрагмент кода содержит номера строк в формате "номер: текст строки".
- Когда говоришь, где что-то находится, используй ТОЛЬКО те номера строк, которые реально есть во фрагментах. НИКОГДА не выдумывай номера строк.
- Если ask_project вернул фрагменты — составь ответ на их основе. Не вызывай ask_project повторно с тем же или похожим запросом.
- Для ответа на один вопрос достаточно 1-2 вызовов ask_project. Если информации мало — скажи, что нашёл, и уточни у пользователя.
- Различай "упоминание" и "реализацию": если функция упоминается в README или в примере кода, это НЕ место её реализации. Реализация — там, где функция определена (def ...).

Пример 1 (индексация):
User: Изучи мой проект
Assistant: {"name": "index_project", "arguments": {"path": "."}}
User: Observation from index_project: Проиндексировано 12 файлов, 48 фрагментов.
Assistant: Я проиндексировал ваш проект: 12 файлов, 48 фрагментов. Теперь вы можете задавать вопросы о коде.

Пример 2 (вопрос о реализации):
User: Где реализован калькулятор?
Assistant: {"name": "ask_project", "arguments": {"query": "def calculator функция калькулятор"}}
User: Observation from ask_project: 1. [chapter1/agent.py]
27: def calculator(expression: str) -> str:
28:     try:
29:         tree = ast.parse(expression, mode="eval")
Assistant: Калькулятор реализован в файле chapter1/agent.py — функция calculator определена на строке 27. Она использует безопасный AST-парсер для вычисления арифметических выражений.
""".strip()


def main():
    print("=" * 60)
    print("📚 Агент с RAG по проекту (Глава 6)")
    print("=" * 60)

    if not chapter4_agent.ensure_ollama_running():
        return

    if not chapter4_agent.model_exists(chapter4_agent.MODEL):
        print(chapter4_agent.model_not_found_message(chapter4_agent.MODEL))
        return

    chapter4_agent.preload_model(chapter4_agent.MODEL)
    chapter4_agent.install_plugins()

    # Добавляем правила работы с проектом к промпту
    base.SYSTEM_PROMPT = base.SYSTEM_PROMPT + "\n" + PROJECT_RULES

    # Контекст из Главы 1 (4096) здесь мал. Замеры на qwen2.5-coder 3B:
    # системный промпт с PROJECT_RULES — 1511 токенов, одна выдача
    # ask_project — ещё 1322. Второй вызов инструмента выходит за 4096,
    # и Ollama молча срезает начало диалога вместе с системным промптом.
    base.NUM_CTX = 8192

    # Показываем статистику индекса
    try:
        from chapter6.src.indexer import project_stats
        stats = project_stats()
        print(f"\n🧠 Модель: {chapter4_agent.MODEL}")
        print(f"📚 Индекс проекта: {stats['files']} файлов, {stats['chunks']} фрагментов")
    except Exception:
        print(f"\n🧠 Модель: {chapter4_agent.MODEL}")

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
            if not chapter4_agent.ensure_ollama_running():
                break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()