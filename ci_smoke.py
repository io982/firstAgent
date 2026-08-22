"""Проверки, которые работают без Ollama и без векторной базы.

Запуск:
    python ci_smoke.py

Это не полноценные тесты — настоящий набор на pytest запланирован
отдельной главой (см. ROADMAP.md). Здесь проверяется детерминированная
часть кода: та, где нет ни модели, ни сети, ни случайности.
"""

import os
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
from chapter7.src import bm25 as hybrid_bm25  # noqa: E402
from chapter7.src import indexer as hybrid_indexer  # noqa: E402
from chapter7.src import tools as hybrid_tools  # noqa: E402
from chapter8.src import native as dual  # noqa: E402
from chapter8.src import probe as fmt_probe  # noqa: E402
from chapter8.src import schema as tool_schema  # noqa: E402

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

print("\nГлава 7: слияние двух поисков (RRF)")
ranks = hybrid_indexer._rrf_contribution(["первый", "второй", "третий"])
check("вклад убывает с местом", ranks["первый"] > ranks["второй"] > ranks["третий"])
check(
    "формула 1/(k + место)",
    abs(ranks["первый"] - 1.0 / (hybrid_indexer.RRF_K + 1)) < 1e-9,
)

# Ради этого свойства RRF и выбран: документ, найденный обоими способами,
# обгоняет документ, который лишь чуть выше в одном списке.
vector_list = ["A", "B"]
bm25_list = ["B", "C"]
merged = hybrid_indexer._rrf_contribution(vector_list)
for doc_id, value in hybrid_indexer._rrf_contribution(bm25_list).items():
    merged[doc_id] = merged.get(doc_id, 0.0) + value
check("найденное обоими способами выигрывает", merged["B"] > merged["A"] > merged["C"])

print("\nГлава 7: BM25")
docs = [
    {"id": "impl", "text": "def calculator(expression): return ast.parse(expression)", "metadata": {}},
    {"id": "mention", "text": "в этом разделе calculator только упоминается", "metadata": {}},
    {"id": "other", "text": "совершенно посторонний текст про погоду", "metadata": {}},
]
index = hybrid_bm25.SimpleBM25(docs)
found = index.search("def calculator")
check("реализация выше упоминания", found and found[0]["id"] == "impl")
check("нерелевантное не попадает в выдачу", all(r["id"] != "other" for r in found))
check("токены посчитаны один раз при построении", len(index.doc_tokens) == len(docs))

print("\nГлава 7: бюджет выдачи")
long_text = "\n".join(f"{i}: строка кода номер {i}" for i in range(1, 200))
trimmed = hybrid_tools._trim_to_whole_lines(long_text, 300)
check("обрезано до лимита", len(trimmed) < 400, f"длина {len(trimmed)}")
check("метка обрезки на месте", trimmed.endswith("... [фрагмент обрезан] ..."))
check(
    "каждая строка сохранила номер",
    all(part.split(":")[0].isdigit() for part in trimmed.split("\n")[:-1]),
)
check("короткий текст не трогаем", hybrid_tools._trim_to_whole_lines("1: коротко", 300) == "1: коротко")
check("бюджет фрагмента строже общего", hybrid_tools.MAX_FRAGMENT_CHARS < hybrid_tools.MAX_TOTAL_CHARS)

print("\nГлава 7: индекс отделён от Главы 6")
check(
    "коллекции разные",
    hybrid_indexer.PROJECT_COLLECTION != "project_files",
    hybrid_indexer.PROJECT_COLLECTION,
)
check("путь к базе абсолютный", os.path.isabs(hybrid_indexer.CHROMA_PERSIST_DIR))

print("\nГлава 8: инструменты в JSON Schema")
schemas = tool_schema.registry_to_schemas()
by_name = {item["function"]["name"]: item["function"] for item in schemas}
check("схема собрана для каждого инструмента", len(schemas) == len(registry.TOOL_REGISTRY))
check("форма записи как ждёт Ollama", all(item["type"] == "function" for item in schemas))

calc = by_name["calculator"]["parameters"]
check("обязательный параметр помечен", calc["required"] == ["expression"])
check("тип параметра переведён", calc["properties"]["expression"]["type"] == "string")

reader = by_name["read_file"]["parameters"]
check("параметр со значением по умолчанию не обязателен", "max_chars" not in reader["required"])
check("значение по умолчанию подсказано модели",
      "8000" in reader["properties"]["max_chars"].get("description", ""))

# Реестр Главы 4 перечисляет в параметрах и **kwargs. В промпте это безвредно,
# а в схеме превратилось бы в обязательное поле "kwargs".
remember_schema = by_name["remember"]["parameters"]
check("**kwargs не попал в схему", "kwargs" not in remember_schema["properties"])

print("\nГлава 8: разбор двух форматов")
native_message = {
    "content": "",
    "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}],
}
check(
    "нативный вызов приведён к виду Главы 1",
    dual.extract_native_calls(native_message)
    == [{"name": "calculator", "arguments": {"expression": "2+2"}}],
)
check("нативный формат распознан", dual.extract_calls_any_format(native_message)[1] == dual.FORMAT_NATIVE)

text_message = {"content": '{"name": "calculator", "arguments": {"expression": "2+2"}}'}
calls, used = dual.extract_calls_any_format(text_message)
check("текстовый формат распознан", used == dual.FORMAT_TEXT and len(calls) == 1)
check("мусор в tool_calls не роняет разбор",
      dual.extract_native_calls({"tool_calls": [None, {}, {"function": {}}]}) == [])

print("\nГлава 8: промпт под протокол")
native_prompt = dual.system_prompt_for(dual.FORMAT_NATIVE)
text_prompt = dual.system_prompt_for(dual.FORMAT_TEXT)
check("для форматов промпты разные", native_prompt != text_prompt)
check(
    "нативный промпт не учит текстовому протоколу",
    '{"name"' not in native_prompt and "Observation" not in native_prompt,
)
check("нативный промпт не перечисляет инструменты сам", "calculator" not in native_prompt)
check("текстовый промпт остался прежним", text_prompt == base.SYSTEM_PROMPT)

print("\nГлава 8: определение формата")
check("форматы различимы",
      len({fmt_probe.FORMAT_NATIVE, fmt_probe.FORMAT_TEXT, fmt_probe.FORMAT_UNKNOWN}) == 3)
check("вызов пробного инструмента опознан",
      fmt_probe._mentions_probe_tool('{"name": "probe_add", "arguments": {"a": 2, "b": 3}}'))
check("упоминание имени вызовом не считается",
      not fmt_probe._mentions_probe_tool("инструмент probe_add складывает числа"))
check("пустой ответ вызовом не считается", not fmt_probe._mentions_probe_tool(""))

print()
if failures:
    print(f"❌ Провалено проверок: {len(failures)}")
    for name in failures:
        print(f"   - {name}")
    sys.exit(1)

print("✅ Все проверки прошли")
