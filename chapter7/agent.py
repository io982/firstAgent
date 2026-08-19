"""Агент с гибридным поиском по проекту (Глава 7)."""
import requests
from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter7.src import tools  # плагины регистрируются при импорте

base.VERBOSE = True

PROJECT_RULES = r"""
Работа с документами проекта (гибридный поиск):
- index_project, ask_project, project_stats — обычные инструменты.
- Пользователь просит изучить/проиндексировать проект → вызови index_project.
- Пользователь спрашивает о коде, структуре, функциях → сначала вызови ask_project.
- Код проекта на английском: для поиска используй английские идентификаторы
  ("где калькулятор" → ищи "def calculator").

ГЛАВНОЕ ПРАВИЛО — источник данных:
- Единственный источник правды — это Observation, который вернул ask_project.
- Путь к файлу в Observation указан в квадратных скобках: [путь/к/файлу].
- Номера строк — это числа перед двоеточием, например "165: def calculator".
- В ответе указывай ТОЛЬКО тот файл и те номера строк, которые есть в последнем Observation.
- ЗАПРЕЩЕНО называть файлы или строки, которых нет в последнем Observation.
- ЗАПРЕЩЕНО отвечать по памяти или по примерам.
- Если Observation не содержит ответа — честно скажи, что не нашёл.
""".strip()


def main():
    print("=" * 60)
    print("🔍 Агент с гибридным поиском (Глава 7)")
    print("=" * 60)
    # ← Читаем конфигурацию напрямую из chapter1 (base), а не через chapter4
    if not base.ensure_ollama_running():
        return
    if not base.model_exists(base.MODEL):
        print(f"❌ Модель '{base.MODEL}' не найдена.")
        return
    base.preload_model(base.MODEL)
    # install_plugins из chapter4 всё равно нужен — он подменяет ядро
    chapter4_agent.install_plugins()
    base.SYSTEM_PROMPT = base.SYSTEM_PROMPT + "\n" + PROJECT_RULES
    try:
        from chapter7.src.indexer import project_stats
        stats = project_stats()
        print(f"\n🧠 Модель: {base.MODEL}")
        print(f"🔍 Гибридный поиск: {stats['files']} файлов, {stats['chunks']} фрагментов")
    except Exception:
        print(f"\n🧠 Модель: {base.MODEL}")
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
            # ask_agent вызываем через chapter4_agent — там ядро уже подменено
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
