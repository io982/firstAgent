"""Сравнение моделей: кто отдаёт нативные tool_calls, а кто нет.

Аналог test_hybrid.py из Главы 7 — файл, который превращает утверждение
главы в проверяемый факт.

Запуск:
    python -m chapter8.test_formats
"""

import requests

from chapter1 import agent as base
from chapter8.src.probe import FORMAT_NATIVE, FORMAT_TEXT, probe_tool_format

# Модели, на которых снимался замер 20 августа 2026, и ожидаемый результат.
# Расхождение будет видно сразу, если Ollama или модель изменят поведение.
KNOWN_MODELS = [
    ("qwen2_5coder3b_q5:latest", FORMAT_TEXT),
    ("qwen2.5-coder:7b", FORMAT_TEXT),
    ("llama3.1:8b", FORMAT_NATIVE),
]


def main():
    installed = set(base.list_installed_models())
    if not installed:
        print("Ollama не отвечает или моделей нет.")
        return

    print("=" * 62)
    print(f"{'модель':<28} {'ожидалось':<10} {'получено':<10} итог")
    print("=" * 62)

    checked = 0
    mismatches = 0
    for model, expected in KNOWN_MODELS:
        if model not in installed:
            print(f"{model:<28} {expected:<10} {'—':<10} не установлена")
            continue

        try:
            detected = probe_tool_format(model)
        except requests.exceptions.RequestException as e:
            print(f"{model:<28} {expected:<10} {'—':<10} ошибка: {e}")
            continue

        checked += 1
        if detected == expected:
            print(f"{model:<28} {expected:<10} {detected:<10} совпало")
        else:
            mismatches += 1
            print(f"{model:<28} {expected:<10} {detected:<10} РАСХОЖДЕНИЕ")

    print("=" * 62)
    print(f"Проверено моделей: {checked}, расхождений: {mismatches}")
    if checked:
        print("\nВывод главы: флаг tools в `ollama show` объявляют все три,")
        print("а нативный формат из них использует не каждая.")


if __name__ == "__main__":
    main()
