"""Быстрая проверка: код импортируется, тесты проходят, ссылки в тексте живые.
Запуск: python ci_smoke.py
"""
import re
import subprocess
import sys
from pathlib import Path

# Консоль Windows по умолчанию живёт в cp1251 и не умеет печатать эмодзи.
# Без этой строки скрипт падает с UnicodeEncodeError ещё до первого теста.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ====================================================================
# ССЫЛКИ ИЗ ТЕКСТА ГЛАВ НА КОД
# ====================================================================
# Главы ссылаются на код с номерами строк: [`trim_by_tokens`](src/context.py#L87).
# Стоит переставить пару функций — и ссылка молча начинает вести не туда.
# Читатель это увидит, автор — нет, поэтому проверяем машиной.

# Документы, которые проверяем. Черновики ненаписанных глав сюда не входят:
# там ссылки указывают на код, которого ещё нет.
DOCS = ["README.md", "ROADMAP.md"] + [f"chapter{n}/README.md" for n in range(5)]

MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
# Цель ссылки должна выглядеть путём. Иначе `TOOLS[name](**args)` в тексте
# читается как ссылка на файл `**args`.
PATH_LIKE = re.compile(r"^[\w./-]+(#L?[\w-]+)?$")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def strip_code(text: str) -> str:
    """Убирает блоки кода перед поиском ссылок.

    Вставки в одинарных кавычках при этом трогать нельзя: в них живут
    подписи ссылок вида [`trim_by_tokens`](src/context.py#L87), а без подписи
    самая полезная проверка — что строка ещё та самая — работать не будет.
    """
    return FENCED_CODE.sub("", text)


def check_links(doc: Path) -> list[str]:
    """Возвращает список проблем со ссылками в одном документе.

    Проверяем три вещи, от простой к строгой:
      1. файл, на который ссылаемся, существует;
      2. строка с номером из якоря #L123 в нём есть;
      3. если подпись ссылки — имя функции, оно встречается в этой строке.

    Третья проверка и ловит сдвиг номеров: файл на месте, строка на месте,
    но там уже другой код.
    """
    problems = []
    text = strip_code(doc.read_text(encoding="utf-8"))

    for label, target in MARKDOWN_LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not PATH_LIKE.match(target):
            continue  # это не путь, а кусок кода в тексте

        path, _, anchor = target.partition("#")
        full = (doc.parent / path).resolve()

        if not full.exists():
            problems.append(f"{target} — файла нет")
            continue

        if not (anchor.startswith("L") and anchor[1:].isdigit()):
            continue  # якорь на заголовок, номер строки не проверяем

        number = int(anchor[1:])
        lines = full.read_text(encoding="utf-8").splitlines()
        if number > len(lines):
            problems.append(f"{target} — в файле только {len(lines)} строк")
            continue

        name = label.strip("`")
        if IDENTIFIER.match(name) and name not in lines[number - 1]:
            problems.append(
                f"{target} — в строке {number} нет '{name}': {lines[number - 1].strip()[:50]}"
            )

    return problems


print("🔗 Проверяю ссылки из текста глав на код...")
link_problems = []
for name in DOCS:
    doc = Path(name)
    if not doc.exists():
        continue
    problems = check_links(doc)
    if problems:
        link_problems.append(name)
        print(f"❌ {name}")
        for problem in problems:
            print(f"   {problem}")
    else:
        print(f"✅ {name}")

print()
print("🔍 Запускаю быстрые тесты для всех глав...")
print("=" * 60)

# Список глав с тестами. Добавляйте сюда главы по мере написания.
chapters = [
    "chapter1/test_agent.py",
    "chapter2/tests.py",
    "chapter3/tests.py",
    "chapter4/tests.py",
]

failed = []
for test_file in chapters:
    print(f"\n📦 Тестирую {test_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        failed.append(test_file)
        print(f"❌ {test_file} провалился")
        print(result.stdout)
    else:
        print(f"✅ {test_file} прошёл")

print("\n" + "=" * 60)
if failed or link_problems:
    if failed:
        print(f"❌ Провалено тестов: {len(failed)}")
        for f in failed:
            print(f"   - {f}")
    if link_problems:
        print(f"❌ Битые ссылки в документах: {len(link_problems)}")
        for f in link_problems:
            print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ Все тесты прошли, ссылки на месте")
