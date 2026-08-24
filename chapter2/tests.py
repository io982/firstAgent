import os
import sys

import pytest

# Добавляем корень проекта в путь для корректных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter2.src.tools import (
    TOOL_REGISTRY,
    calculator,
    execute_tool,
    get_all_tools_schemas,
    read_file,
)


# ====================================================================
# 1. Тесты регистрации инструментов и генерации JSON Schema
# ====================================================================
def test_tool_registry_population():
    """Проверяет, что декоратор @tool автоматически регистрирует функции."""
    assert "calculator" in TOOL_REGISTRY
    assert "read_file" in TOOL_REGISTRY
    assert "get_weather" in TOOL_REGISTRY

def test_schema_generation():
    """Проверяет корректность автоматической генерации JSON Schema из сигнатуры."""
    schema = TOOL_REGISTRY["get_weather"]["schema"]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    # ИСПРАВЛЕНО: ищем подстроку в нижнем регистре для надежности
    assert "погоду" in schema["function"]["description"].lower()
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "city" in params["properties"]
    assert "city" in params["required"]  # Параметр без default значения должен быть required

def test_get_all_tools_schemas():
    """Проверяет, что функция возвращает корректный список всех схем для промпта."""
    schemas = get_all_tools_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 3  # calculator, read_file, get_weather
    assert all(s["type"] == "function" for s in schemas)

# ====================================================================
# 2. Тесты диспетчера инструментов (execute_tool)
# ====================================================================
def test_execute_tool_success():
    """Проверяет успешное выполнение зарегистрированного инструмента через диспетчер."""
    result = execute_tool("calculator", {"expression": "10 * 5"})
    assert result == "50"

    result = execute_tool("get_weather", {"city": "Москва"})
    assert "Москва" in result and "+20°C" in result

def test_execute_tool_unknown():
    """Проверяет обработку вызова несуществующего инструмента (защита от галлюцинаций)."""
    result = execute_tool("hack_the_planet", {"target": "all"})
    assert "не найден" in result.lower()
    assert "calculator" in result.lower()  # Должен подсказать доступные инструменты

def test_execute_tool_validation_error():
    """Проверяет валидацию аргументов: модель передала неверные имена параметров."""
    # Передаем неправильное имя аргумента (вместо expression -> math_expression)
    result = execute_tool("calculator", {"math_expression": "2+2"})
    assert "Ошибка" in result or "unexpected keyword" in result.lower() or "TypeError" in result

    # Передаем недостающий обязательный аргумент
    result = execute_tool("get_weather", {})
    assert "Ошибка" in result or "missing" in result.lower()

# ====================================================================
# 3. Тесты безопасности инструментов
# ====================================================================
def test_calculator_security():
    """Проверяет, что калькулятор защищен от выполнения произвольного Python-кода."""
    assert "Ошибка" in calculator("import os")
    assert "Ошибка" in calculator("__import__('os').system('ls')")
    # ИСПРАВЛЕНО: 5 & 3 — это безопасная математическая (побитовая) операция, она вернет 1.
    # Вместо этого проверим, что модель не может вызвать опасные встроенные функции (builtins)
    assert "Ошибка" in calculator("open('secret.txt')")
    assert "Ошибка" in calculator("exec('print(1)')")
    assert "Ошибка" in calculator("eval('1+1')")

def test_read_file_safety():
    """Проверяет базовую безопасность чтения файлов (несуществующие пути, лимит размера)."""
    # 1. Несуществующий файл
    result = read_file("/nonexistent/path/file.txt")
    assert "не найден" in result.lower()

    # 2. Ограничение размера вывода (защита от переполнения контекста)
    test_file = "test_large_file.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("A" * 3000)

    result = read_file(test_file)
    assert len(result) <= 2000  # Проверка лимита в 2000 символов из tools.py

    os.remove(test_file)

# ====================================================================
# 4. Интеграционные тесты (Требуют работающей Ollama)
# ====================================================================
@pytest.mark.integration
def test_agent_tool_integration():
    """Интеграционный тест: проверяет, что агент корректно использует новый Tool API в реальном цикле ReAct с живой моделью Ollama."""
    import requests

    from chapter2.agent import ask_agent

    # 0. Предварительная проверка: запущена ли Ollama?
    # Чтобы тест не падал с непонятной ошибкой сети, а вежливо пропускался
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip("Ollama не запущена (localhost:11434 недоступен). Пропускаем интеграционный тест.")

    # Тест 1: Агент должен использовать инструмент calculator и получить верный результат
    result = ask_agent("Сколько будет 125 * 4?")
    assert "500" in result, f"Агент не смог посчитать или не использовал инструмент. Ответ: {result[:200]}"

    # Тест 2: Агент должен корректно обработать ошибку инструмента (деление на ноль)
    # и не зациклиться, а сообщить пользователю о проблеме
    result = ask_agent("Посчитай 10 / 0")
    is_error_handled = any(word in result.lower() for word in ["ошибк", "нельзя", "бесконеч", "недопустим", "деление"])
    assert is_error_handled, f"Модель не обработала ошибку деления на ноль. Ответ: {result[:200]}"

    # Тест 3: Агент должен использовать read_file и найти информацию в нем
    test_file = "test_agent_secret.txt"
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Секретное слово для интеграционного теста: OLLAMA_IS_GREAT")
        result = ask_agent(f"Прочитай файл {test_file} и скажи, какое там секретное слово.")
        assert "OLLAMA_IS_GREAT" in result, f"Агент не смог прочитать файл или не нашел слово. Ответ: {result[:200]}"
    finally:
        # Гарантированная очистка тестового файла даже при падении теста
        if os.path.exists(test_file):
            os.remove(test_file)
