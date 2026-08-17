"""Проверки, которые работают без Ollama и без векторной базы.

Запуск:
    python ci_smoke.py

Это не полноценные тесты — настоящий набор на pytest запланирован
отдельной главой (см. ROADMAP.md). Здесь проверяется детерминированная
часть кода: та, где нет ни модели, ни сети, ни случайности.
"""

import sys

failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


print("Импорт всех глав")
from chapter1 import agent as base  # noqa: E402
from chapter2 import paraphraser  # noqa: E402, F401
from chapter3 import agent as ch3  # noqa: E402, F401
from chapter4 import agent as ch4  # noqa: E402
from chapter4.src import tools as registry  # noqa: E402
from chapter5.src import tools as memory_tools  # noqa: E402, F401
from chapter6.src import indexer  # noqa: E402
from chapter6.src import tools as project_tools  # noqa: E402, F401

check("все главы импортируются", True)

print("\nРеестр плагинов")
ch4.install_plugins()
names = registry.known_tools()
check("инструменты зарегистрированы", len(names) >= 9, f"найдено {len(names)}")
check("calculator в реестре", "calculator" in names)
check("плагины Главы 5 подхвачены", "remember" in names)
check("плагины Главы 6 подхвачены", "ask_project" in names)
check("промпт содержит параметры", "Параметры:" in registry.render_tools_for_prompt())
check("ядро видит реестр", base.KNOWN_TOOLS == names)

print("\nКалькулятор")
check("арифметика", registry.execute_tool("calculator", {"expression": "(15 * 7) + 3"}) == "108")
check("деление на ноль не роняет", "Ошибка" in registry.execute_tool("calculator", {"expression": "1/0"}))
check(
    "импорт запрещён",
    "Ошибка" in registry.execute_tool("calculator", {"expression": "__import__('os').getcwd()"}),
)

print("\nФильтрация аргументов")
check(
    "лишний аргумент отбрасывается",
    registry.execute_tool("calculator", {"expression": "2+2", "reason": "любопытство"}) == "4",
)


@registry.tool("_ci_echo", "служебный инструмент для проверки **kwargs")
def _ci_echo(text: str = "", **kwargs) -> str:
    return f"{text}|{sorted(kwargs)}"


check(
    "функция с **kwargs получает всё",
    registry.execute_tool("_ci_echo", {"text": "a", "b": 1, "c": 2}) == "a|['b', 'c']",
)
del registry.TOOL_REGISTRY["_ci_echo"]

print("\nПарсер вызовов инструментов")
check(
    "простой JSON",
    base.extract_tool_calls('{"name": "calculator", "arguments": {"expression": "2+2"}}')
    == [{"name": "calculator", "arguments": {"expression": "2+2"}}],
)
check(
    "JSON в блоке кода",
    len(base.extract_tool_calls('```json\n{"name": "calculator", "arguments": {}}\n```')) == 1,
)
check("обычный текст не вызов", base.extract_tool_calls("Ответ: 4") == [])
check(
    "несуществующий инструмент опознан",
    base.extract_unknown_tool_names('{"name": "send_email", "arguments": {}}') == ["send_email"],
)
check(
    "существующий не считается неизвестным",
    base.extract_unknown_tool_names('{"name": "calculator", "arguments": {}}') == [],
)
check(
    "словарь в тексте не путается с вызовом",
    base.extract_unknown_tool_names('пример: {"name": "Иван"}') == [],
)

print("\nПоиск модели")
_real = base.list_installed_models
base.list_installed_models = lambda: ["qwen2.5-coder:3b", "nomic-embed-text:latest"]
check("точное совпадение", base.model_exists("qwen2.5-coder:3b"))
check("имя без тега", base.model_exists("nomic-embed-text"))
check("чужой тег не подходит", not base.model_exists("qwen2.5-coder:7b"))
check("отсутствующая модель", not base.model_exists("llama3"))
base.list_installed_models = _real

print("\nЧанкинг (Глава 6)")
text = "\n".join(f"строка {i}" for i in range(1, 201))
chunks = indexer.chunk_text_with_lines(text, chunk_size=200, overlap_lines=3)
check("текст разбит на несколько чанков", len(chunks) > 1, f"получено {len(chunks)}")
check("чанк — пара (чистый, с номерами)", all(len(c) == 2 for c in chunks))
first_clean, first_numbered = chunks[0]
check("нумерация начинается с первой строки", first_numbered.startswith("1: строка 1"))
check("в чистом тексте нет номеров", not first_clean.startswith("1: "))

second_numbered = chunks[1][1]
first_line_no = int(second_numbered.split(":")[0])
overlap_start = int(first_numbered.split("\n")[-3].split(":")[0])
check("следующий чанк начинается с перекрытия", first_line_no == overlap_start,
      f"{first_line_no} против {overlap_start}")

for clean, numbered in chunks:
    numbers = [int(line.split(":")[0]) for line in numbered.split("\n")]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        check("номера строк идут подряд", False, f"разрыв в чанке от {numbers[0]}")
        break
else:
    check("номера строк идут подряд", True)

check("пустой текст не даёт чанков с содержимым", indexer.chunk_text_with_lines("") == [("", "1: ")])

print()
if failures:
    print(f"❌ Провалено проверок: {len(failures)}")
    for name in failures:
        print(f"   - {name}")
    sys.exit(1)

print("✅ Все проверки прошли")
