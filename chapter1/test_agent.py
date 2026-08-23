"""
Тесты для Главы 1: Базовый агент с ReAct-паттерном.
Используется профессиональный фреймворк pytest.

Запуск:
    python -m pytest chapter1/test_agent.py -v                 # Быстрые тесты (без модели)
    python -m pytest chapter1/test_agent.py -v -m integration  # Интеграционные тесты (с моделью)
"""
import pytest

from chapter1.agent import TOOLS, calculator, execute_tool, extract_json_from_text


# ====================================================================
# 1. Тесты калькулятора (Параметризованные)
# ====================================================================
@pytest.mark.parametrize("expression, expected", [
    ("5256+665", "5921"),
    ("100-37", "63"),
    ("15*7", "105"),
    ("10/4", "2.5"),
    ("(15*7)+3", "108"),
    ("2**10", "1024"),
    ("10//3", "3"),
    ("10%3", "1"),
    ("-5+10", "5"),
    ("+5+10", "15"),
])
def test_calculator_basic(expression, expected):
    """Проверяет базовые арифметические операции."""
    assert calculator(expression) == expected

def test_calculator_errors():
    """Проверяет безопасную обработку ошибок в калькуляторе."""
    assert "Ошибка" in calculator("10/0")
    assert "Ошибка" in calculator("import os")
    assert "Ошибка" in calculator("not a number")
    assert "Ошибка" in calculator("5 & 3")  # Запрещенный оператор

# ====================================================================
# 2. Тесты парсинга JSON
# ====================================================================
def test_extract_json_valid():
    """Чистый валидный JSON."""
    result = extract_json_from_text('{"tool": "calculator", "args": {"expression": "5+5"}}')
    assert result is not None and result["tool"] == "calculator"

def test_extract_json_with_text():
    """JSON с поясняющим текстом вокруг (частая ситуация с LLM)."""
    text = 'Вот мой ответ: {"tool": "calculator", "args": {"expression": "5+5"}} готово!'
    result = extract_json_from_text(text)
    assert result is not None and result["tool"] == "calculator"

def test_extract_json_invalid():
    """Сломанный JSON или текст без JSON должен возвращать None."""
    assert extract_json_from_text('{"tool": "calculator"') is None
    assert extract_json_from_text("Просто текстовый ответ без вызова инструмента.") is None
    assert extract_json_from_text('{"name": "calculator", "args": {}}') is None  # Нет ключа 'tool'

# ====================================================================
# 3. Тесты execute_tool (Диспетчер инструментов)
# ====================================================================
def test_execute_tool_success():
    assert execute_tool("calculator", {"expression": "5+5"}) == "10"

def test_execute_tool_errors():
    # Несуществующий инструмент
    result = execute_tool("nonexistent_tool", {})
    assert "не найден" in result.lower()

    # Неверные аргументы (галлюцинация модели)
    result = execute_tool("get_current_time", {"unexpected_arg": "value"})
    assert "Ошибка" in result or "unexpected" in result.lower()

def test_tools_registered():
    """Проверка, что инструменты корректно зарегистрированы в словаре."""
    assert "calculator" in TOOLS
    assert "get_current_time" in TOOLS

# ====================================================================
# 4. Интеграционные тесты (Требуют работающей Ollama)
# ====================================================================
@pytest.mark.integration
def test_agent_integration():
    """Тесты, проверяющие реальный цикл ReAct с моделью."""
    from chapter1.agent import ask_agent

    # Тест 1: Простой запрос
    result = ask_agent("Посчитай 5+5")
    assert "10" in result

    # Тест 2: Обработка ошибки деления на ноль
    result = ask_agent("Посчитай 10/0")
    is_error_handled = any(word in result.lower() for word in ["ошибк", "нельзя", "бесконеч", "недопустим", "деление"])
    assert is_error_handled, f"Модель не обработала ошибку. Ответ: {result[:150]}"
