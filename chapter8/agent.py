"""Агент с нативными tool_calls и откатом на текстовый формат (Глава 8).

Отличие от предыдущих глав: при старте агент выясняет пробным запросом,
каким форматом отвечает выбранная модель, и дальше работает подходящим
способом. Флагу `tools` из `ollama show` доверять нельзя — его объявляют
и те модели, которые нативный формат не используют.

Запуск:
    python -m chapter8.agent

    $env:AGENT_MODEL = "llama3.1:8b"        # нативный формат
    $env:AGENT_MODEL = "qwen2.5-coder:7b"   # текстовый формат
"""

import requests

from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter8.src.native import ask_agent_dual
from chapter8.src.probe import FORMAT_UNKNOWN, describe_format, probe_tool_format


def main():
    print("=" * 60)
    print("🔌 Агент с нативными tool_calls (Глава 8)")
    print("=" * 60)

    if not base.ensure_ollama_running():
        return

    if not base.model_exists(base.MODEL):
        print(base.model_not_found_message(base.MODEL))
        return

    base.preload_model(base.MODEL)
    chapter4_agent.install_plugins()

    print("\n🔍 Определяю формат вызова инструментов...")
    try:
        tool_format = probe_tool_format(base.MODEL)
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Проба не удалась ({e}). Работаем текстовым форматом.")
        tool_format = "text"

    print(describe_format(base.MODEL, tool_format))
    if tool_format == FORMAT_UNKNOWN:
        print("   Попробуйте модель из раздела «Рекомендуемые модели» README.")

    print(f"\n🧠 Модель: {base.MODEL}")
    print(f"🔌 Инструментов в реестре: {len(base.KNOWN_TOOLS)}")
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

            answer = ask_agent_dual(user_input, tool_format)

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
