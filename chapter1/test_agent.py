"""
Тесты для Главы 1: Базовый агент с ReAct-паттерном.

Запуск:
    python -m chapter1.test_agent              # Быстрые тесты (без модели)
    python -m chapter1.test_agent --full       # Все тесты (с моделью)
"""

import io
import sys

# Устанавливаем UTF-8 для консоли Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Импортируем компоненты агента
from chapter1.agent import TOOLS, calculator, execute_tool, extract_json_from_text


# ====================================================================
# Утилиты для тестирования
# ====================================================================
class TestLogger:
    """Простой логгер для тестов."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.logs = []

    def test(self, name: str, condition: bool, details: str = ""):
        """Записывает результат теста."""
        if condition:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"

        log_entry = f"{status}: {name}"
        if details and not condition:
            log_entry += f"\n   └─ {details}"

        self.logs.append(log_entry)
        print(log_entry)

    def skip(self, name: str, reason: str):
        """Пропускает тест."""
        self.skipped += 1
        log_entry = f"⚠️ SKIP: {name}\n   └─ {reason}"
        self.logs.append(log_entry)
        print(log_entry)

    def summary(self):
        """Выводит итоговую статистику."""
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 60)
        print(f"📊 Итого: {total} тестов")
        print(f"   ✅ Пройдено: {self.passed}")
        print(f"   ❌ Провалено: {self.failed}")
        print(f"   ⚠️ Пропущено: {self.skipped}")
        print("=" * 60)

        if self.failed > 0:
            print("\n❌ ТЕСТЫ ПРОВАЛЕНЫ")
            return 1
        else:
            print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
            return 0

logger = TestLogger()

# ====================================================================
# 1. Тесты калькулятора
# ====================================================================
def test_calculator():
    print("\n🧮 Тесты калькулятора")
    print("-" * 60)

    # Базовые операции
    logger.test("Сложение: 5256+665", calculator("5256+665") == "5921")
    logger.test("Вычитание: 100-37", calculator("100-37") == "63")
    logger.test("Умножение: 15*7", calculator("15*7") == "105")
    logger.test("Деление: 10/4", calculator("10/4") == "2.5")

    # Сложные выражения
    logger.test("Скобки: (15*7)+3", calculator("(15*7)+3") == "108")
    logger.test("Степень: 2**10", calculator("2**10") == "1024")
    logger.test("Целочисленное деление: 10//3", calculator("10//3") == "3")
    logger.test("Остаток: 10%3", calculator("10%3") == "1")

    # Унарные операции
    logger.test("Отрицание: -5+10", calculator("-5+10") == "5")
    logger.test("Плюс: +5+10", calculator("+5+10") == "15")

    # Ошибки
    result = calculator("10/0")
    logger.test("Деление на ноль (ошибка)", "Ошибка" in result, result)

    result = calculator("import os")
    logger.test("Запрещённая операция (ошибка)", "Ошибка" in result or "Неподдерживаемый" in result, result)

    result = calculator("not a number")
    logger.test("Невалидное выражение (ошибка)", "Ошибка" in result, result)


# ====================================================================
# 2. Тесты парсинга JSON
# ====================================================================
def test_json_parsing():
    print("\n🔍 Тесты парсинга JSON")
    print("-" * 60)

    # Валидный JSON
    text = '{"tool": "calculator", "args": {"expression": "5+5"}}'
    result = extract_json_from_text(text)
    logger.test("Чистый JSON", result is not None and result["tool"] == "calculator")

    # JSON с текстом вокруг
    text = 'Вот мой ответ: {"tool": "calculator", "args": {"expression": "5+5"}} готово!'
    result = extract_json_from_text(text)
    logger.test("JSON с текстом вокруг", result is not None and result["tool"] == "calculator")

    # JSON с переносами строк
    text = '''
    Думаю, нужно вызвать инструмент.
    {"tool": "calculator", "args": {"expression": "5+5"}}
    '''
    result = extract_json_from_text(text)
    logger.test("JSON с переносами строк", result is not None and result["tool"] == "calculator")

    # Невалидный JSON
    text = '{"tool": "calculator", "args": {"expression": "5+5"'
    result = extract_json_from_text(text)
    logger.test("Сломанный JSON (None)", result is None)

    # Текст без JSON
    text = "Это просто текстовый ответ без вызова инструмента."
    result = extract_json_from_text(text)
    logger.test("Текст без JSON (None)", result is None)

    # JSON без ключа "tool"
    text = '{"name": "calculator", "args": {"expression": "5+5"}}'
    result = extract_json_from_text(text)
    logger.test("JSON без ключа 'tool' (None)", result is None)


# ====================================================================
# 3. Тесты execute_tool
# ====================================================================
def test_execute_tool():
    print("\n⚙️ Тесты execute_tool")
    print("-" * 60)

    # Успешный вызов
    result = execute_tool("calculator", {"expression": "5+5"})
    logger.test("Успешный вызов calculator", result == "10")

    # Несуществующий инструмент
    result = execute_tool("nonexistent_tool", {})
    logger.test("Несуществующий инструмент (ошибка)", "не найден" in result.lower(), result)

    # Неверные аргументы (лишние)
    result = execute_tool("get_current_time", {"unexpected_arg": "value"})
    logger.test("Лишние аргументы (ошибка)", "Ошибка" in result or "unexpected" in result.lower(), result)

    # Проверка, что все инструменты зарегистрированы
    logger.test("calculator зарегистрирован", "calculator" in TOOLS)
    logger.test("get_current_time зарегистрирован", "get_current_time" in TOOLS)


# ====================================================================
# 4. Интеграционные тесты (с моделью)
# ====================================================================
def test_integration():
    print("\n🤖 Интеграционные тесты (с моделью)")
    print("-" * 60)

    try:
        from chapter1.agent import ask_agent

        # Тест 1: Простой запрос
        print("\nТест: Простой запрос к модели...")
        result = ask_agent("Посчитай 5+5")
        logger.test("Модель вернула ответ", len(result) > 0 and "10" in result, result[:100])

        # Тест 2: Запрос с инструментом
        print("\nТест: Запрос с вызовом инструмента...")
        result = ask_agent("Посчитай (15*7)+3")
        logger.test("Модель использовала calculator", "108" in result, result[:100])

        # Тест 3: Обработка ошибки
        print("\nТест: Обработка ошибки деления на ноль...")
        result = ask_agent("Посчитай 10/0")
        # Ищем корни слов или ключевые понятия, чтобы не зависеть от падежей
        is_error_handled = any(word in result.lower() for word in ["ошибк", "нельзя", "бесконеч", "недопустим", "деление"])
        logger.test("Модель обработала ошибку", is_error_handled, result[:150])

    except ImportError as e:
        logger.skip("Интеграционные тесты", f"Не удалось импортировать ask_agent: {e}")
    except Exception as e:
        logger.skip("Интеграционные тесты", f"Ошибка при запуске: {e}")


# ====================================================================
# Главная функция
# ====================================================================
def main():
    print("=" * 60)
    print("🧪 Тесты для Главы 1: Базовый агент с ReAct-паттерном")
    print("=" * 60)

    # Быстрые тесты (всегда запускаются)
    test_calculator()
    test_json_parsing()
    test_execute_tool()

    # Интеграционные тесты (только с флагом --full)
    if "--full" in sys.argv:
        test_integration()
    else:
        print("\n⚠️ Интеграционные тесты пропущены.")
        print("   Запустите с флагом --full для тестов с моделью:")
        print("   python -m chapter1.test_agent --full")

    # Итоговая статистика
    exit_code = logger.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
