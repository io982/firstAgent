# 🤖 Туториал: Создаём локального ИИ-агента с Ollama

Полный курс по созданию агентов с нуля до production-ready системы.

## 📚 Содержание

| Глава | Тема | Код |
| --- | --- | --- |
| [Глава 1](#глава-1-создание-простого-агента) | Базовый агент с ReAct-паттерном | [chapter1/agent.py](chapter1/agent.py) |
| [Глава 2](#глава-2-добавление-распознавателя) | Мульти-агентная система | [chapter2/paraphraser.py](chapter2/paraphraser.py) |
| [Глава 3](#глава-3-контекст-память-и-производительность) | Контекст, память и производительность | [chapter3/agent.py](chapter3/agent.py) |
| [Глава 4](#глава-4-система-плагинов-добавляем-новые-инструменты) | Система плагинов: расширяем инструменты | [chapter4/src/tools.py](chapter4/src/tools.py) |
| [Глава 5](#глава-5-долгосрочная-память-и-rag) | Долгосрочная память и RAG | [chapter5/agent.py](chapter5/agent.py) |

## 🎯 Что ты научишься делать

- ✅ Создавать агентов с ReAct-паттерном
- ✅ Работать с инструментами (tools/function calling)
- ✅ Строить мульти-агентные системы
- ✅ Оптимизировать память и производительность
- ✅ Создавать расширяемую систему плагинов
- ✅ Добавлять новые инструменты за 5 строк кода
- ✅ Работать с векторной памятью и RAG

## 🛠️ Требования

- Python 3.10+
- [Ollama](https://ollama.com)
- Модель: `ollama pull qwen2.5-coder:3b`

## 🚀 Быстрый старт

```bash
# Клонируй репозиторий
git clone https://github.com/io982/firstAgent.git
cd firstAgent

# Установи зависимости
pip install requests

# Запусти Главу 1 (из корня проекта)
python -m chapter1.agent

# Запусти Главу 2 (мульти-агент)
python -m chapter2.paraphraser

# Запусти Главу 3
python -m chapter3.agent

# Запусти Главу 4 (плагины)
python -m chapter4.agent

# Запусти Главу 5 (долгосрочная память)
pip install chromadb && ollama pull nomic-embed-text
python -m chapter5.agent
```

---

## 🤖 Туториал: Создаём локального ИИ-агента с Ollama

> **Цель туториала:** создать работающего локального агента с инструментами,
> добавить к нему мульти-агентную архитектуру и разобраться в тонкостях
> контекста, памяти и производительности.

> **Требования:**
> - Установленная [Ollama](https://ollama.com)
> - Python 3.10+
> - Любая модель Qwen2.5 Coder или аналогичная с поддержкой tools

---

## 📑 Содержание

- [Глава 1. Создание простого агента](#глава-1-создание-простого-агента)
- [Глава 2. Добавление распознавателя (перефразировщика)](#глава-2-добавление-распознавателя)
- [Глава 3. Контекст, память и производительность](#глава-3-контекст-память-и-производительность)
- [Глава 4. Система плагинов: добавляем новые инструменты](#глава-4-система-плагинов-добавляем-новые-инструменты)
- [Глава 5. Долгосрочная память и RAG](#глава-5-долгосрочная-память-и-rag)
- [Приложение. Шпаргалки и частые ошибки](#приложение)

---

# Глава 1. Создание простого агента

## 1.1 Что такое агент и чем он отличается от чат-бота

**Чат-бот** — это модель, которая получает текст и возвращает текст.
Она не может взаимодействовать с внешним миром.

**Агент** — это модель + инструменты + цикл принятия решений.

```text
Чат-бот:    Вопрос → Модель → Ответ

Агент:      Вопрос → Модель → "Мне нужен инструмент"
                              → Вызов инструмента
                              → Результат инструмента
                              → Модель → "Нужен ещё инструмент"
                              → ...
                              → Финальный ответ
```

Ключевое отличие: агент **сам решает**, когда и какой инструмент вызвать.

## 1.2 Проверка установки Ollama

Откройте терминал и выполните:

```bash
ollama list
```

Вы должны увидеть список установленных моделей:

```text
NAME                        ID              SIZE      MODIFIED
qwen2_5coder3b_q5:latest    f2972a356413    2.4 GB    6 weeks ago
```

Проверьте, что модель поддерживает инструменты:

```bash
ollama show qwen2_5coder3b_q5:latest
```

В выводе должна быть строка:

```text
Capabilities
    tools
    completion
```

> ⚠️ **Важно:** наличие `tools` в Capabilities не гарантирует, что модель
> будет использовать нативный формат tool_calls. Некоторые кастомные модели
> возвращают вызовы инструментов в текстовом формате. Об этом — в разделе 1.5.

## 1.3 Первый запрос к Ollama

Создайте файл `test.py`:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2_5coder3b_q5:latest",
        "messages": [
            {"role": "user", "content": "Напиши функцию факториала на Python"}
        ],
        "stream": False
    },
    timeout=120
)

print(response.json()["message"]["content"])
```

Запуск:

```bash
python test.py
```

Если получили ответ — Ollama работает, можно строить агента.

## 1.4 Архитектура ReAct-агента

ReAct (Reasoning + Acting) — паттерн, в котором модель чередует рассуждения
и действия:

```text
Thought: Мне нужно посчитать выражение
Action: calculator
Action Input: {"expression": "(15 * 7) + 3"}
Observation: 108
Thought: Теперь я знаю ответ
Final Answer: (15 * 7) + 3 = 108
```

В нашей реализации мы упрощаем формат до JSON:

```json
{"name": "calculator", "arguments": {"expression": "(15 * 7) + 3"}}
```

## 1.5 Проблема: нативные tools vs текстовый парсинг

Ollama поддерживает два формата работы с инструментами:

### Нативный формат (tool_calls)

Вы отправляете `"tools": [...]` в запросе, а модель возвращает
структурированное поле `tool_calls`:

```python
# Запрос
payload = {
    "model": MODEL,
    "messages": messages,
    "tools": TOOLS  # ← описание инструментов
}

# Ответ
message = response["message"]
if message.get("tool_calls"):
    for call in message["tool_calls"]:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
```

### Текстовый формат (JSON в содержимом)

Модель возвращает вызов инструмента как обычный текст:

```text
{"name": "calculator", "arguments": {"expression": "(15 * 7) + 3"}}
```

В этом случае поле `tool_calls` пустое, и нужно парсить текст.

> 🔍 **Как понять, какой формат использует ваша модель?**
> Сделайте тестовый запрос с tools и посмотрите на ответ. Если `tool_calls`
> пустое, но в `content` есть JSON с инструментами — модель использует
> текстовый формат.

**В этом туториале мы используем текстовый формат**, так как он работает
с любыми моделями, включая кастомные.

## 1.6 Полный код агента

Создайте файл `agent.py`:

```python
import json
import os
import ast
import operator
import re
import subprocess
import sys
import time
import requests


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen2_5coder3b_q5:latest"

MAX_ITERATIONS = 8

# Сколько держать модель в памяти после последнего запроса
KEEP_ALIVE = "5m"

# Режим отладки: True = видно все шаги, False = только результат
VERBOSE = False

# Список известных инструментов
KNOWN_TOOLS = {
    "calculator",
    "list_directory",
    "read_file",
    "search_in_file"
}

# Системный промпт для агента
SYSTEM_PROMPT = """
Ты — автономный агент-ассистент для разработчика.

У тебя есть инструменты:
1. calculator — безопасно считает арифметические выражения.
2. list_directory — показывает файлы и папки в директории.
3. read_file — читает текстовый файл.
4. search_in_file — ищет текст в файле.

Формат вызова инструмента:
{"name": "имя_инструмента", "arguments": {...}}

Примеры:

User: Посчитай 2+2
Assistant: {"name": "calculator", "arguments": {"expression": "2+2"}}
User: Observation from calculator: 4
Assistant: 2 + 2 = 4

User: Покажи файлы в текущей папке
Assistant: {"name": "list_directory", "arguments": {"path": "."}}
User: Observation from list_directory: [FILE] agent.py
Assistant: В текущей папке есть файл agent.py

Правила:
1. Если нужен инструмент, ответь ТОЛЬКО JSON-вызовом.
2. Не добавляй пояснения до или после JSON при вызове инструмента.
3. После получения Observation проанализируй результат.
4. Если нужно вызвать ещё инструмент — снова верни JSON.
5. Когда готов дать финальный ответ, ответь обычным текстом без JSON.
6. Никогда не выдумывай результаты инструментов.
""".strip()


# ============================================================
# АВТОЗАПУСК OLLAMA
# ============================================================

def is_ollama_running() -> bool:
    """Проверяет, отвечает ли сервер Ollama."""
    try:
        response = requests.get(OLLAMA_BASE, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server():
    """Запускает ollama serve как фоновый процесс."""
    print("🚀 Запускаю сервер Ollama...")

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    try:
        process = subprocess.Popen(["ollama", "serve"], **kwargs)
        print(f"✅ Сервер Ollama запущен (PID: {process.pid})")
        return True
    except FileNotFoundError:
        print("❌ Команда 'ollama' не найдена.")
        return False
    except Exception as e:
        print(f"❌ Не удалось запустить Ollama: {e}")
        return False


def wait_for_ollama(timeout: int = 30) -> bool:
    """Ждёт, пока сервер Ollama начнёт отвечать."""
    print(f"⏳ Ожидание запуска сервера (до {timeout} сек)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_ollama_running():
            print("✅ Сервер Ollama готов!")
            return True
        time.sleep(1)

    print("❌ Сервер не запустился за отведённое время.")
    return False


def ensure_ollama_running() -> bool:
    """Гарантирует, что сервер Ollama запущен."""
    if is_ollama_running():
        print("✅ Сервер Ollama уже работает.")
        return True

    if not start_ollama_server():
        return False

    return wait_for_ollama()


def model_exists(model_name: str) -> bool:
    """Проверяет, установлена ли модель."""
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if response.status_code != 200:
            return False

        models = response.json().get("models", [])
        for model in models:
            name = model.get("name", "")
            if name == model_name or name.startswith(f"{model_name.split(':')[0]}:"):
                return True
        return False
    except Exception:
        return False


def preload_model(model_name: str):
    """Предзагрузка модели в память."""
    print(f"📦 Предзагрузка модели '{model_name}'...")
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": "",
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1}
            },
            timeout=120
        )
        if response.status_code == 200:
            print(f"✅ Модель готова!")
    except requests.exceptions.Timeout:
        print("⚠️ Предзагрузка заняла много времени.")
    except Exception as e:
        print(f"⚠️ Ошибка предзагрузки: {e}")


# ============================================================
# ИНСТРУМЕНТЫ АГЕНТА
# ============================================================

# --- Безопасный калькулятор ---

ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Разрешены только числа")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise ValueError("Недопустимый оператор")
        return ALLOWED_BIN_OPS[op_type](
            _safe_eval_node(node.left),
            _safe_eval_node(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("Недопустимый унарный оператор")
        return ALLOWED_UNARY_OPS[op_type](_safe_eval_node(node.operand))

    raise ValueError("Недопустимое выражение")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval_node(tree))
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as e:
        return f"Ошибка калькулятора: {e}"


# --- Файловые инструменты ---

def list_directory(path: str = ".") -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return f"Путь не найден или не директория: {path}"
    try:
        entries = []
        for name in sorted(os.listdir(path))[:200]:
            full = os.path.join(path, name)
            prefix = "[DIR] " if os.path.isdir(full) else "[FILE]"
            entries.append(f"{prefix} {name}")
        return "\n".join(entries) if entries else "Директория пуста"
    except Exception as e:
        return f"Ошибка: {e}"


def read_file(path: str, max_chars: int = 8000) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return f"Файл не найден: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [обрезано] ..."
        return content
    except Exception as e:
        return f"Ошибка: {e}"


def search_in_file(path: str, query: str, max_results: int = 20) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return f"Файл не найден: {path}"
    if not query:
        return "Не указан текст для поиска"
    try:
        results = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if query.lower() in line.lower():
                    results.append(f"{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        break
        return "\n".join(results) if results else f"Не найдено: {query}"
    except Exception as e:
        return f"Ошибка: {e}"


# --- Реестр инструментов ---

def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "calculator":
            return calculator(args.get("expression", ""))
        if name == "list_directory":
            return list_directory(args.get("path", "."))
        if name == "read_file":
            return read_file(args.get("path", ""))
        if name == "search_in_file":
            return search_in_file(args.get("path", ""), args.get("query", ""))
        return f"Неизвестный инструмент: {name}"
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


# ============================================================
# ПАРСЕР ТЕКСТОВЫХ TOOL CALLS
# ============================================================

def normalize_arguments(arguments):
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"expression": arguments, "path": arguments, "query": arguments}
    return {}


def normalize_potential_tool(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if name in KNOWN_TOOLS:
        return {"name": name, "arguments": obj.get("arguments", {})}
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name") in KNOWN_TOOLS:
        return {"name": fn["name"], "arguments": fn.get("arguments", {})}
    return None


def extract_tool_calls(text: str):
    calls = []

    # Поиск в markdown code blocks
    for block in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
        try:
            obj = json.loads(block)
            tc = normalize_potential_tool(obj)
            if tc:
                calls.append(tc)
        except Exception:
            pass

    if calls:
        return calls

    # Поиск JSON в тексте
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            tc = normalize_potential_tool(obj)
            if tc:
                calls.append(tc)
            pos = end
        except json.JSONDecodeError:
            pos = start + 1

    return calls


# ============================================================
# ЯДРО АГЕНТА
# ============================================================

def request_model(messages: list) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    message = response.json().get("message", {})
    if "content" not in message:
        message["content"] = ""
    return message


def ask_agent(user_task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        if VERBOSE:
            print(f"\n--- Итерация {iteration} ---")

        assistant_message = request_model(messages)
        content = assistant_message.get("content", "")

        if VERBOSE:
            print(f"[Model] {content[:500]}")

        tool_calls = extract_tool_calls(content)

        if not tool_calls:
            return content.strip()

        messages.append({"role": "assistant", "content": content})

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = normalize_arguments(tool_call.get("arguments", {}))

            if VERBOSE:
                print(f"[Agent] {tool_name}({tool_args})")

            tool_result = execute_tool(tool_name, tool_args)

            if VERBOSE:
                print(f"[Tool] {tool_result[:200]}")

            messages.append({
                "role": "user",
                "content": f"Observation from {tool_name}: {tool_result}"
            })

    return "Агент достиг лимита итераций."


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    print("=" * 60)
    print("🤖 Локальный ReAct-агент Ollama")
    print("=" * 60)

    if not ensure_ollama_running():
        return

    if not model_exists(MODEL):
        print(f"❌ Модель '{MODEL}' не найдена.")
        return

    preload_model(MODEL)

    print(f"\n🧠 Модель: {MODEL}")
    print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Пока!")
                break

            if not VERBOSE:
                print("⏳ Думаю...")

            answer = ask_agent(user_input)

            print("\n🤖 Агент >")
            print(answer)
            print()

        except KeyboardInterrupt:
            break
        except requests.exceptions.ConnectionError:
            print("⚠️ Связь потеряна. Восстанавливаю...")
            if not ensure_ollama_running():
                break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
```

## 1.7 Запуск и тестирование

```bash
python agent.py
```

Примеры запросов:

```text
Вы > Посчитай (15 * 7) + 3
🤖 Агент > (15 * 7) + 3 = 108

Вы > Покажи файлы в текущей папке
🤖 Агент > [FILE] agent.py

Вы > Найди в agent.py слово calculator
🤖 Агент > Найдено 5 совпадений...
```

## 1.8 Безопасность инструментов

> 🔒 **Важное правило:** никогда не давайте агенту неконтролируемый доступ
> к выполнению команд (`shell=True`, `eval()`, `exec()`).

Наш калькулятор использует **безопасный парсер AST**, а не `eval()`:

```python
# ❌ Опасно
result = eval(user_expression)

# ✅ Безопасно
tree = ast.parse(expression, mode="eval")
result = _safe_eval_node(tree)  # разрешены только числа и базовые операторы
```

Если хотите добавить инструмент выполнения команд — делайте его
с подтверждением пользователя:

```python
def run_command(command: str) -> str:
    confirm = input(f"Выполнить '{command}'? [y/n]: ").lower()
    if confirm != "y":
        return "Команда отменена"
    # ... выполнение
```

---

# Глава 2. Добавление распознавателя

## 2.1 Зачем нужен перефразировщик

Пользователи часто пишут размыто:

```text
"посчитай это"
"что в папке"
"почитай файл"
```

Агенту нужны чёткие инструкции. Перефразировщик решает эту проблему:

```text
Пользователь: "посчитай сколько будет 156 плюс произведение 1233 минус 12 на 15"
Перефразировщик: "Посчитай арифметическое выражение: 156 + (1233 - 12) * 15"
Агент: выполняет и возвращает 18471
```

## 2.2 Архитектура мульти-агентной системы

```text
Пользователь
    ↓
[Агент-перефразировщик]  ← делает запрос чётким
    ↓
[ReAct-агент с инструментами]  ← выполняет задачу
    ↓
Ответ пользователю
```

Это классическая схема **Planner → Executor**.

## 2.3 Код перефразировщика

Создайте файл `paraphraser.py`:

```python
import requests
import agent  # импортируем первого агента


VERBOSE = False
agent.VERBOSE = VERBOSE  # передаём режим первому агенту


PARAPHRASE_PROMPT = """
Ты — перефразировщик запросов для ИИ-агента.

Твоя задача — перефразировать запрос пользователя, сделав его максимально
чётким, конкретным и пригодным для автоматического выполнения.

Правила:
1. Сохраняй исходный смысл.
2. НИКОГДА не меняй числа, выражения, пути к файлам и другие конкретные данные.
3. Убирай слова-паразиты и лишние вводные конструкции.
4. Если запрос размытый, добавь конкретику, но не выдумывай данные.
5. Если запрос уже чёткий — верни его без изменений.
6. Отвечай ТОЛЬКО перефразированным текстом.
7. Без кавычек, без префиксов, без пояснений.
""".strip()


def paraphrase(text: str) -> str:
    payload = {
        "model": agent.MODEL,
        "messages": [
            {"role": "system", "content": PARAPHRASE_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 2048
        }
    }

    response = requests.post(agent.OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json().get("message", {}).get("content", "").strip()

    # Очистка от возможных артефактов
    result = result.strip('"').strip("'").strip()

    for prefix in ["перефразированный запрос:", "перефразировано:", "ответ:"]:
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip()

    return result


def main():
    print("=" * 60)
    print("🔁 Мульти-агентная система")
    print("   Перефразировщик → ReAct-агент")
    print("=" * 60)

    if not agent.ensure_ollama_running():
        return

    if not agent.model_exists(agent.MODEL):
        print(f"❌ Модель '{agent.MODEL}' не найдена.")
        return

    agent.preload_model(agent.MODEL)

    print(f"\n🤖 Система готова. Модель: {agent.MODEL}")
    print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Пока!")
                break

            # Шаг 1: Перефразирование
            paraphrased = paraphrase(user_input)
            print(f"🔄 Перефразировано: {paraphrased}")

            # Шаг 2: Выполнение агентом
            print("⏳ Агент выполняет...")
            answer = agent.ask_agent(paraphrased)

            print("\n🤖 Агент >")
            print(answer)
            print()

        except KeyboardInterrupt:
            break
        except requests.exceptions.ConnectionError:
            print("⚠️ Связь потеряна.")
            if not agent.ensure_ollama_running():
                break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
```

## 2.4 Последовательная vs параллельная работа

Сейчас агенты работают **последовательно**:

```text
[Перефразировщик] → ждём → [Агент] → ждём → Ответ
```

Это правильно, потому что агент зависит от результата перефразировщика.

### Когда параллельность имеет смысл

| Сценарий | Решение |
|---|---|
| Несколько независимых агентов | `concurrent.futures` |
| Фоновая подготовка контекста | Запуск в отдельном потоке |
| Конвейер (pipeline) | Следующий запрос перефразируется, пока агент отвечает на текущий |

### Пример параллельного выполнения

```python
import concurrent.futures

def parallel_agents(task):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(agent1.run, task)
        future2 = executor.submit(agent2.run, task)

        result1 = future1.result()
        result2 = future2.result()

    return merge(result1, result2)
```

> ⚠️ При параллельных запросах к одной модели Ollama ставит их в очередь
> (по умолчанию `OLLAMA_NUM_PARALLEL=1`). Для реальной параллельности
> нужно поднять этот параметр (см. Главу 3).

## 2.5 Добавление агента-критика (опционально)

Третий агент может проверять ответ исполнителя:

```python
CRITIC_PROMPT = """
Ты — критик. Проверь ответ агента на корректность.
Если ответ верный — верни "OK".
Если есть ошибка — верни описание проблемы.
"""

def critic(question: str, answer: str) -> str:
    payload = {
        "model": agent.MODEL,
        "messages": [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет: {answer}"}
        ],
        "stream": False
    }
    response = requests.post(agent.OLLAMA_URL, json=payload, timeout=120)
    return response.json()["message"]["content"]
```

---

# Глава 3. Контекст, память и производительность

## 3.1 Что такое контекст (num_ctx)

`num_ctx` — это размер "окна" модели, в которое помещается весь текст:

```text
[Системный промпт] + [История сообщений] + [Текущий запрос] + [Ответ модели]
```

Измеряется в **токенах** (примерно 1 токен = 0.75 слова в английском,
или 1-2 символа в русском).

### Что происходит с памятью

Ollama резервирует память под **KV-cache** для всего контекстного окна,
даже если фактически используется малая часть:

```text
num_ctx = 65536 → Ollama резервирует память под 65536 токенов
Реально используется 2000 токенов → 63536 токенов памяти простаивает
```

### Формула потребления памяти

```text
Общая память = Веса модели + (KV-cache × num_ctx × batch_size)
```

Где `batch_size` — количество параллельных запросов.

## 3.2 Помнит ли агент предыдущие запросы?

### Короткий ответ: НЕТ

В текущей реализации каждый запрос — **новая сессия**:

```python
def ask_agent(user_task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task}  # ← каждый раз с нуля
    ]
```

После ответа переменная `messages` уничтожается.

### Что агент помнит, а что нет

| Помнит | Не помнит |
|---|---|
| Итерации внутри одной задачи | Предыдущие запросы пользователя |
| Результаты вызовов инструментов | Контекст прошлой беседы |
| Цепочку действий для текущей цели | Что вы обсуждали 5 минут назад |

## 3.3 Нужен ли большой контекст?

### Анализ потребления

Посмотрим, сколько токенов реально использует наш агент:

| Компонент | Токены |
|---|---|
| Системный промпт | ~300 |
| Запрос пользователя | ~50–200 |
| 2-3 итерации с tools | ~500–2000 |
| **Итого** | **~1000–2500** |

При `num_ctx = 65536` используется **3-5%** окна. Остальное — впустую.

### Рекомендации

| Задача | Рекомендуемый num_ctx |
|---|---|
| Агент с инструментами (без памяти) | 4096 |
| Перефразировщик | 2048 |
| Короткая история диалога (5-10 сообщений) | 8192 |
| Длинная история + большие файлы | 16384–32768 |
| Анализ очень больших документов | 65536+ |

## 3.4 Как добавить память между запросами

Если нужен диалог с памятью:

```python
conversation_history = []

def ask_agent_with_memory(user_task: str) -> str:
    global conversation_history

    # Добавляем запрос
    conversation_history.append({"role": "user", "content": user_task})

    # Формируем сообщения: system + история
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)

    # ... (рабочий цикл с tools) ...

    final_answer = "..."  # результат работы агента

    # Сохраняем ответ в историю
    conversation_history.append({"role": "assistant", "content": final_answer})

    # Обрезаем историю, чтобы не раздувать контекст
    MAX_HISTORY = 10
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return final_answer
```

### Команда очистки памяти

```python
if user_input.lower() in {"/reset", "/clear", "очистить"}:
    conversation_history.clear()
    print("🧹 История очищена.")
    continue
```

## 3.5 Загрузка модели при параллельных запросах

### Параметры Ollama

| Параметр | По умолчанию | Что делает |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 1 | Сколько запросов обрабатывать одновременно |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | Сколько разных моделей держать в памяти |
| `keep_alive` | 5m | Сколько модель живёт в памяти после запроса |

### Что происходит при параллельных запросах

**По умолчанию (`OLLAMA_NUM_PARALLEL=1`):**

```text
Запрос 1 → [Выполняется]
Запрос 2 → [Ждёт в очереди]
Запрос 3 → [Ждёт в очереди]
```

Память: одна модель, один KV-cache.

**С параллельностью (`OLLAMA_NUM_PARALLEL=3`):**

```text
Запрос 1 → [Выполняется]
Запрос 2 → [Выполняется]  ← параллельно
Запрос 3 → [Выполняется]  ← параллельно
```

Память: одна модель, но **3 KV-cache**. Потребление растёт.

### Как включить параллельность

Windows (PowerShell):

```powershell
$env:OLLAMA_NUM_PARALLEL = "2"
$env:OLLAMA_MAX_LOADED_MODELS = "2"
ollama serve
```

Linux/macOS:

```bash
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
ollama serve
```

### Мониторинг

```bash
ollama ps
```

Вывод:

```text
NAME                        SIZE      PROCESSOR    UNTIL
qwen2_5coder3b_q5:latest    2.4 GB    100% GPU     4 minutes from now
```

## 3.6 keep_alive: управление временем жизни модели

| Значение | Поведение |
|---|---|
| `"0"` | Выгружать сразу после запроса |
| `"5m"` | Держать 5 минут (по умолчанию) |
| `"1h"` | Держать 1 час |
| `"-1"` | Никогда не выгружать (до перезапуска Ollama) |

Для агента, который используется часто:

```python
KEEP_ALIVE = "-1"  # модель всегда в памяти
```

Для редкого использования:

```python
KEEP_ALIVE = "1m"  # выгружать через минуту простоя
```

## 3.7 Оптимизация: чек-лист

- [ ] Уменьшить `num_ctx` до реально необходимого
- [ ] Установить `keep_alive` в зависимости от частоты использования
- [ ] Использовать `VERBOSE = False` в продакшене
- [ ] Не загружать несколько моделей одновременно без необходимости
- [ ] Мониторить память через `ollama ps`
- [ ] Для параллельных запросов поднять `OLLAMA_NUM_PARALLEL` и уменьшить `num_ctx`

---

# Глава 4. Система плагинов: добавляем новые инструменты

## 4.1 Проблема Главы 1: жёстко зашитые инструменты

В Главе 1 у агента было 4 инструмента, и каждый был «вшит» в три места:

```python
# 1. Множество известных имён
KNOWN_TOOLS = {"calculator", "list_directory", ...}

# 2. Описание в системном промпте (вручную)
SYSTEM_PROMPT = """...
1. calculator — безопасно считает...
2. list_directory — показывает файлы...
"""

# 3. Диспетчер if/elif
def execute_tool(name, args):
    if name == "calculator":
        return calculator(...)
    if name == "list_directory":
        return list_directory(...)
    # ... каждый новый инструмент = правка в трёх местах
```

Чтобы добавить один инструмент, нужно изменить три разных места и нигде
не ошибиться. Это не масштабируется.

**Цель Главы 4:** добавление нового инструмента должно занимать 5 строк кода,
и ничего больше менять не нужно.

## 4.2 Решение: реестр инструментов через декораторы

Паттерн «реестр» (registry): заводим словарь, в котором ключ — имя инструмента,
значение — функция и её описание. Декоратор `@tool` автоматически кладёт
функцию в реестр в момент определения.

```python
import inspect

# name -> {"func": callable, "description": str}
TOOL_REGISTRY = {}

def tool(name: str, description: str):
    """Декоратор: регистрирует функцию как инструмент агента."""
    def decorator(func):
        TOOL_REGISTRY[name] = {"func": func, "description": description}
        return func
    return decorator
```

Вспомогательные функции реестра:

```python
def known_tools() -> set:
    """Множество имён зарегистрированных инструментов."""
    return set(TOOL_REGISTRY)


def execute_tool(name: str, args: dict) -> str:
    """Вызывает инструмент по имени, фильтруя лишние аргументы."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"Неизвестный инструмент: {name}"
    try:
        func = entry["func"]
        sig = inspect.signature(func)
        # Берём только те аргументы, которые функция реально принимает
        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        return str(func(**valid_args))
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


def render_tools_for_prompt() -> str:
    """Собирает нумерованный список инструментов для системного промпта."""
    lines = []
    for number, (name, entry) in enumerate(TOOL_REGISTRY.items(), 1):
        lines.append(f"{number}. {name} — {entry['description']}")
    return "\n".join(lines)
```

Обратите внимание: `execute_tool` больше не содержит `if/elif` по именам —
он один и тот же для любого количества плагинов. А `render_tools_for_prompt`
генерирует описание инструментов для промпта автоматически из реестра.

### Защита от лишних аргументов через `inspect.signature`

LLM иногда «галлюцинирует» и передаёт инструментам аргументы, которых нет
в сигнатуре функции. Например:

```json
{"name": "calculator", "arguments": {"expression": "2+2", "reason": "нужно посчитать"}}
```

Если просто сделать `func(**args)`, Python упадёт с `TypeError: unexpected keyword argument 'reason'`.

Чтобы этого избежать, мы используем `inspect.signature` и передаём функции
**только те аргументы, которые она реально принимает**:

```python
sig = inspect.signature(func)
valid_args = {k: v for k, v in args.items() if k in sig.parameters}
return str(func(**valid_args))
```

Лишние аргументы просто отбрасываются, и агент продолжает работать.

## 4.3 Как модель сама выбирает нужный инструмент

Никакой магии нет — выбор делает LLM по описаниям. Схема такая:

```text
TOOL_REGISTRY
    ↓ render_tools_for_prompt()
Системный промпт: "У тебя есть инструменты: 1. calculator — ... 2. ..."
    ↓
Модель читает запрос пользователя + описания
    ↓
Модель сама решает, какой инструмент подходит, и возвращает JSON:
{"name": "search_in_directory", "arguments": {"query": "TODO"}}
    ↓
Ядро агента парсит JSON (Глава 1.5) и вызывает execute_tool()
    ↓
Реестр находит функцию по имени и передаёт аргументы через **args
```

Ключевое правило: **описание в декораторе — это то, что видит модель.**
Чем точнее описание, тем правильнее модель выбирает инструмент. Сравните:

```python
# ❌ Плохо: модель не поймёт, когда это использовать
@tool("do_stuff", "делает всякое")

# ✅ Хорошо: понятно назначение и формат данных
@tool("search_in_directory", "ищет текст во всех файлах директории (рекурсивно)")
```

## 4.4 Новые инструменты Главы 4

К четырём инструментам из Главы 1 добавляем ещё четыре.

### write_file — запись в файл

```python
@tool("write_file", "записывает текст в файл (создаёт или перезаписывает)")
def write_file(path: str, content: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        return f"Директория не найдена: {directory}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Записано {len(content)} символов в {path}"
```

> 💡 Проверка `if directory and ...` нужна для случая, когда файл создаётся
> в текущей директории и `os.path.dirname` возвращает пустую строку.

### search_in_directory — поиск по всей директории

Рекурсивный обход через `os.walk`, пропуская служебные папки
(`.git`, `__pycache__`, `node_modules`, `.venv`):

```python
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}

@tool("search_in_directory", "ищет текст во всех файлах директории (рекурсивно)")
def search_in_directory(path: str = ".", query: str = "", max_results: int = 30) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return f"Директория не найдена: {path}"
    if not query:
        return "Не указан текст для поиска"

    results = []
    query_lower = query.lower()

    for root, dirs, files in os.walk(path):
        # Пропускаем служебные директории
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in sorted(files):
            if len(results) >= max_results:
                break
            full_path = os.path.join(root, name)
            try:
                matches_in_file = 0
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_number, line in enumerate(f, start=1):
                        if query_lower in line.lower():
                            relative = os.path.relpath(full_path, path)
                            results.append(f"{relative}:{line_number}: {line.rstrip()}")
                            matches_in_file += 1
                            # Не больше 3 совпадений на файл
                            if matches_in_file >= 3:
                                break
            except OSError:
                continue

        if len(results) >= max_results:
            break

    if not results:
        return f"Ничего не найдено: {query}"
    if len(results) >= max_results:
        results.append(f"... [показаны первые {max_results}] ...")
    return "\n".join(results)
```

### run_command — выполнение команд с подтверждением

> 🔒 **Главное правило безопасности Главы 4:** агент НИКОГДА не выполняет
> команды без явного подтверждения пользователя. `shell=True` без
> подтверждения — это дыра, через которую модель может удалить файлы.

```python
COMMAND_TIMEOUT = 30  # секунд

@tool("run_command", "выполняет команду в терминале (спрашивает подтверждение пользователя)")
def run_command(command: str) -> str:
    if not command:
        return "Не указана команда"

    # Обязательное подтверждение перед выполнением
    confirm = input(
        f"\n⚠️  Агент хочет выполнить команду:\n   {command}\n"
        f"Разрешить? [y/n]: "
    ).strip().lower()

    if confirm != "y":
        return "Команда отменена пользователем"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.strip()[:2000])
        if result.stderr:
            parts.append(f"[stderr] {result.stderr.strip()[:500]}")
        parts.append(f"[код возврата: {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Команда выполнялась дольше {COMMAND_TIMEOUT} сек и была остановлена"
    except Exception as e:
        return f"Ошибка выполнения команды: {e}"
```

Защитные механизмы здесь:

| Механизм | Зачем |
|---|---|
| `input()` с подтверждением | Человек — последнее звено перед выполнением |
| `timeout=30` | Зависшая команда не заблокирует агента навсегда |
| Обрезка вывода до 2000 символов | Огромный stdout не раздует контекст модели |
| Возврат кода возврата | Модель поймёт, что команда упала, и сможет это учесть |

### http_get — агент ходит в интернет

```python
@tool("http_get", "скачивает страницу по URL и возвращает её текст")
def http_get(url: str, max_chars: int = 4000) -> str:
    if not url.startswith(("http://", "https://")):
        return "URL должен начинаться с http:// или https://"
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "firstAgent/1.0 (tutorial agent)"}
        )
        if response.status_code != 200:
            return f"HTTP {response.status_code} для {url}"
        text = response.text[:max_chars]
        if len(response.text) > max_chars:
            text += "\n... [обрезано] ..."
        return text
    except requests.exceptions.Timeout:
        return f"Превышено время ожидания ответа от {url}"
    except Exception as e:
        return f"Ошибка HTTP-запроса: {e}"
```

Теперь агент может, например: «прочитай README по этой ссылке и перескажи».

> ⚠️ Возвращается сырой HTML/текст страницы. Для продакшена стоит добавить
> извлечение чистого текста (например, библиотекой `beautifulsoup4`).

## 4.5 Добавляем инструмент за 5 строк

Вот и весь процесс. Откройте `chapter4/src/tools.py` и допишите:

```python
@tool("current_time", "возвращает текущие дату и время")
def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Всё. Не нужно менять ни промпт, ни диспетчер, ни ядро агента:

1. Декоратор положил функцию в `TOOL_REGISTRY` при импорте модуля.
2. `render_tools_for_prompt()` сам добавил строку в системный промпт.
3. `execute_tool()` сам найдёт функцию по имени.

Проверить реестр можно запуском самого модуля:

```bash
python -m chapter4.src.tools
```

```text
Зарегистрированные инструменты:
  - calculator: безопасно считает арифметические выражения
  - list_directory: показывает файлы и папки в директории
  - read_file: читает текстовый файл
  - search_in_file: ищет текст в файле
  - write_file: записывает текст в файл (создаёт или перезаписывает)
  - search_in_directory: ищет текст во всех файлах директории (рекурсивно)
  - run_command: выполняет команду в терминале (спрашивает подтверждение пользователя)
  - http_get: скачивает страницу по URL и возвращает её текст
  - current_time: возвращает текущие дату и время

Проверка calculator: 108
Проверка current_time: 2026-08-16 12:00:00
```

## 4.6 Подключение плагинов к ядру агента

Ядро агента взято из Главы 1 без переписывания — мы лишь подменяем
три вещи на версии из реестра. Файл `chapter4/agent.py`:

```python
import requests

from chapter1 import agent as base
from chapter4.src import tools


SYSTEM_PROMPT_TEMPLATE = """
Ты — автономный агент-ассистент для разработчика.

У тебя есть инструменты:
{tools}

Формат вызова инструмента:
{{"name": "имя_инструмента", "arguments": {{...}}}}

Пример:
User: Посчитай 2+2
Assistant: {{"name": "calculator", "arguments": {{"expression": "2+2"}}}}
User: Observation from calculator: 4
Assistant: 2 + 2 = 4

Правила:
1. Если нужен инструмент, ответь ТОЛЬКО JSON-вызовом или несколькими JSON-вызовами.
2. Не добавляй пояснения до или после JSON при вызове инструмента.
3. После получения Observation проанализируй результат.
4. Если нужно вызвать ещё инструмент — снова верни JSON.
5. Когда готов дать финальный ответ пользователю, ответь обычным текстом без JSON.
6. Никогда не выдумывай результаты инструментов.
""".strip()


def install_plugins():
    """Подключает реестр плагинов к ядру агента из Главы 1."""
    base.KNOWN_TOOLS = tools.known_tools()
    base.SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
        tools=tools.render_tools_for_prompt()
    )
    base.execute_tool = tools.execute_tool
```

Что делает `install_plugins()`:

| Было в Главе 1 | Стало в Главе 4 |
|---|---|
| `KNOWN_TOOLS` — множество, прописанное вручную | `tools.known_tools()` — из реестра |
| `SYSTEM_PROMPT` — список инструментов вручную | Шаблон + `render_tools_for_prompt()` |
| `execute_tool` — цепочка `if/elif` | Универсальный диспетчер реестра |

> 💡 Обратите внимание на двойные фигурные скобки `{{...}}` в шаблоне:
> это экранирование для `str.format()`, чтобы JSON-примеры остались
> одинарными скобками в итоговом промпте.

## 4.7 Запуск и тестирование

```bash
# Из корня проекта
python -m chapter4.agent
```

```text
============================================================
🔌 Агент с системой плагинов (Глава 4)
============================================================
🔌 Загружено плагинов: 9
   - calculator: безопасно считает арифметические выражения
   - list_directory: показывает файлы и папки в директории
   - read_file: читает текстовый файл
   - search_in_file: ищет текст в файле
   - write_file: записывает текст в файл (создаёт или перезаписывает)
   - search_in_directory: ищет текст во всех файлах директории (рекурсивно)
   - run_command: выполняет команду в терминале (спрашивает подтверждение пользователя)
   - http_get: скачивает страницу по URL и возвращает её текст
   - current_time: возвращает текущие дату и время
```

Примеры запросов, которые теперь работают:

```text
Вы > Найди все места, где используется KEEP_ALIVE
🤖 Агент > search_in_directory → chapter1/agent.py:24: KEEP_ALIVE = "5m" ...

Вы > Создай файл notes.txt с текстом "привет от агента"
🤖 Агент > write_file → Записано 18 символов в .../notes.txt

Вы > Который час?
🤖 Агент > current_time → Сейчас 2026-08-16 12:00:00

Вы > Выполни команду "python --version"
⚠️  Агент хочет выполнить команду:
   python --version
Разрешить? [y/n]: y
🤖 Агент > Python 3.13.x
```

## 4.8 Итоги главы: что даёт система плагинов

| Критерий | Глава 1 (if/elif) | Глава 4 (реестр) |
|---|---|---|
| Новый инструмент | Правка в 3 местах | 5 строк с декоратором |
| Промпт | Пишется вручную | Генерируется из реестра |
| Диспетчер | Растёт с каждым инструментом | Один на все инструменты |
| Ошибка «забыл добавить» | Легко допустить | Невозможна — декоратор делает всё сам |
| Лишние аргументы от модели | `TypeError` и падение | Фильтруются через `inspect.signature` |

### Чек-лист безопасности новых плагинов

- [ ] Опасные действия (команды, запись, удаление) — только с подтверждением
- [ ] Таймауты на все внешние вызовы (`subprocess`, `requests`)
- [ ] Обрезка длинного вывода, чтобы не раздувать контекст
- [ ] Возврат ошибок строкой, а не исключением — модель должна их увидеть
- [ ] Описание в `@tool` точно объясняет, когда использовать инструмент

---

---
# Глава 5. Долгосрочная память и RAG
## 5.1 Проблема: короткая память из Главы 3
В Главе 3 мы добавили память между запросами через `conversation_history`,
но она ограничена 10 сообщениями. Это создаёт проблемы:
```text
Вы > Прочитай README проекта
🤖 Агент > [читает файл]
Вы > (через 20 сообщений) Какие там были инструкции по установке?
🤖 Агент > Я не помню, что было раньше. Контекст переполнен.
```
Агент не может:
- Помнить информацию из прошлых сессий
- Искать по всем прошлым разговорам
- Использовать знания из документов проекта
**Решение:** RAG (Retrieval-Augmented Generation) — агент сохраняет важную
информацию в векторную базу и извлекает её по запросу.
## 5.2 Что такое эмбеддинги и векторный поиск
**Эмбеддинг** — это числовой вектор, который представляет смысл текста.
Похожие по смыслу тексты имеют похожие векторы:
```text
"Как установить проект?" → [0.12, -0.45, 0.89, ..., 0.23]  (768 чисел)
"Setup instructions"      → [0.11, -0.44, 0.88, ..., 0.22]  (похожий вектор)
"Который час?"            → [-0.89, 0.12, -0.33, ..., 0.67]  (другой вектор)
```
**Векторный поиск:**
1. Превращаем запрос в эмбеддинг
2. Ищем в базе векторы с минимальным косинусным расстоянием
3. Возвращаем соответствующие тексты
```text
Запрос: "инструкции по установке"
↓ эмбеддинг
Вектор: [0.12, -0.45, 0.89, ...]
↓ поиск похожих векторов
Результаты:
- README.md: "Для установки выполните pip install..."
- notes.txt: "Setup: клонируйте репозиторий..."
```
## 5.3 Установка ChromaDB и модели эмбеддингов
**ChromaDB** — локальная векторная база данных. Не требует сервера, хранит всё в файлах.
```bash
pip install chromadb
```
**Модель эмбеддингов** через Ollama:
```bash
ollama pull nomic-embed-text
```
Проверка:
```bash
ollama list
```
Должна появиться:
```text
NAME                    ID              SIZE      MODIFIED
nomic-embed-text:latest abc123...       274 MB    just now
```
## 5.4 Векторная база знаний (`src/vectorstore.py`)
Создайте `chapter5/src/embeddings.py`:
```python
"""Работа с эмбеддингами через Ollama."""
import requests
from chapter1 import agent as base
EMBEDDING_MODEL = "nomic-embed-text"
def get_embedding(text: str) -> list[float]:
"""Получает эмбеддинг для текста через Ollama."""
response = requests.post(
f"{base.OLLAMA_BASE}/api/embeddings",
json={
"model": EMBEDDING_MODEL,
"prompt": text
},
timeout=30
)
response.raise_for_status()
return response.json()["embedding"]
```
Создайте `chapter5/src/vectorstore.py`:
```python
"""Векторная база знаний на ChromaDB."""
import chromadb
from .embeddings import get_embedding
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "agent_memory"
# Новый API ChromaDB (версия 0.4+)
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
def get_or_create_collection():
"""Получает или создаёт коллекцию для хранения документов."""
return client.get_or_create_collection(
name=COLLECTION_NAME,
metadata={"hnsw:space": "cosine"}  # косинусное сходство
)
def add_document(doc_id: str, text: str, metadata: dict = None):
"""Добавляет документ в векторную базу."""
collection = get_or_create_collection()
embedding = get_embedding(text)
collection.add(
ids=[doc_id],
embeddings=[embedding],
documents=[text],
metadatas=[metadata or {}]
)
def search_documents(query: str, n_results: int = 5) -> list[dict]:
"""Ищет похожие документы по запросу."""
collection = get_or_create_collection()
query_embedding = get_embedding(query)
results = collection.query(
query_embeddings=[query_embedding],
n_results=n_results
)
documents = []
for i in range(len(results["ids"][0])):
documents.append({
"id": results["ids"][0][i],
"text": results["documents"][0][i],
"metadata": results["metadatas"][0][i],
"distance": results["distances"][0][i]
})
return documents
def list_documents(limit: int = 10) -> list[dict]:
"""Возвращает список документов в базе (полный текст, без обрезки)."""
collection = get_or_create_collection()
results = collection.get(limit=limit)
documents = []
for i in range(len(results["ids"])):
documents.append({
"id": results["ids"][i],
"text": results["documents"][i],
"metadata": results["metadatas"][i]
})
return documents
```
> ⚠️ **Важно:** используется `chromadb.PersistentClient(path=...)`.
> Старый API с `Settings(chroma_db_impl="duckdb+parquet")` удалён
> в ChromaDB 0.4+ и вызывает `ValueError: deprecated configuration`.
## 5.5 Инструменты remember, recall, list_memory
Создайте `chapter5/src/tools.py`. Все три инструмента регистрируются
через декоратор `@tool` из Главы 4:
```python
"""Плагины для работы с векторной базой знаний."""
import hashlib
from datetime import datetime
from chapter4.src import tools as chapter4_tools
from .vectorstore import add_document, search_documents, list_documents
def _generate_id(text: str) -> str:
"""Генерирует уникальный ID для документа."""
return hashlib.md5(text.encode()).hexdigest()[:12]
@chapter4_tools.tool(
"remember",
"сохраняет важную информацию в долгосрочную память. Используй, когда "
"пользователь просит запомнить факт, или когда узнаёшь новый факт "
"о проекте/настройках/предпочтениях пользователя."
)
def remember(content: str = "", category: str = "general", **kwargs) -> str:
# Fallback для выдуманных параметров
if not content and kwargs:
content = " | ".join(f"{k}: {v}" for k, v in kwargs.items())
if not content:
return "Ошибка: не указан текст для сохранения. Используй параметр content."
# Дедупликация: проверяем, нет ли уже очень похожей записи
try:
existing = search_documents(content, n_results=1)
if existing and existing[0]["distance"] < 0.1:
return f"Эта информация уже есть в памяти (ID: {existing[0]['id']})."
except Exception:
pass
doc_id = _generate_id(content + datetime.now().isoformat())
metadata = {
"category": category,
"timestamp": datetime.now().isoformat()
}
add_document(doc_id, content, metadata)
return f"Сохранено в памяти (ID: {doc_id}, категория: {category})"
@chapter4_tools.tool(
"recall",
"ищет информацию в долгосрочной памяти по ключевым словам или фразе. "
"Используй, когда помнишь, ЧТО ищешь. Параметр query ОБЯЗАТЕЛЬНО должен быть непустым."
)
def recall(query: str, max_results: int = 3) -> str:
if not query or not query.strip():
return "Ошибка: не указан поисковый запрос. Используй параметр query."
results = search_documents(query, n_results=max_results)
if not results:
return "Ничего не найдено в памяти."
output = []
for i, result in enumerate(results, 1):
distance = result["distance"]
confidence = "высокая" if distance < 0.3 else "средняя" if distance < 0.5 else "низкая"
output.append(f"{i}. [{confidence} уверенность] {result['text']}")
if result["metadata"].get("category"):
output.append(f"   Категория: {result['metadata']['category']}")
return "\n".join(output)
@chapter4_tools.tool(
"list_memory",
"показывает ВСЕ сохранённые документы в памяти без поиска. Используй, когда "
"пользователь просит 'покажи что ты помнишь', 'что у тебя в памяти', 'вспомни всё'."
)
def list_memory(limit: int = 10) -> str:
docs = list_documents(limit=limit)
if not docs:
return "Память пуста."
output = [f"Найдено {len(docs)} записей в памяти:\n"]
for i, doc in enumerate(docs, 1):
category = doc["metadata"].get("category", "без категории")
output.append(f"{i}. [{category}] {doc['text']}")
return "\n".join(output)
```
Защитные механизмы в инструментах:
| Механизм | Где | Зачем |
|---|---|---|
| `**kwargs` fallback | `remember` | Модель может выдумать параметры `key`/`value` |
| Дедупликация | `remember` | Не создавать копии уже существующих записей |
| Проверка пустого query | `recall` | ChromaDB падает на пустом запросе |
| Полный текст без обрезки | `list_memory` | Модель не будет "уточнять" обрезанный текст |
## 5.6 Подключение к агенту из Главы 4
Создайте `chapter5/agent.py`. Ядро берём из Главы 4, добавляем правила памяти:
```python
"""Агент с долгосрочной памятью (RAG)."""
import requests
from chapter1 import agent as base
from chapter4 import agent as chapter4_agent
from chapter5.src import tools  # плагины регистрируются при импорте
MEMORY_RULES = r"""
Работа с долгосрочной памятью:
- remember, recall и list_memory — это обычные инструменты, такие же как калькулятор.
- Пользователь просит запомнить → вызови remember.
- Пользователь спрашивает факт, который мог сохранить ранее → вызови recall с конкретным запросом.
- Пользователь просит показать память → вызови list_memory, затем перескажи ВСЕ возвращённые записи.
- Показывать сохранённые записи — это нормальная разрешённая операция. Ограничений безопасности на это НЕТ.
- Сохраняй текст на языке пользователя (по-русски, если пишет по-русски).
Пример 1 (запомнить):
User: Запомни, что мой проект называется firstAgent
Assistant: {"name": "remember", "arguments": {"content": "Мой проект называется firstAgent"}}
User: Observation from remember: Сохранено в памяти
Assistant: Запомнил, что ваш проект называется firstAgent.
Пример 2 (показать память):
User: Покажи, что ты помнишь
Assistant: {"name": "list_memory", "arguments": {}}
User: Observation from list_memory: Найдено 2 записей в памяти:
1. [general] Мой проект называется firstAgent
2. [general] Меня зовут Владимир
Assistant: Вот что я помню:
1. [general] Мой проект называется firstAgent
2. [general] Меня зовут Владимир
""".strip()
def main():
print("=" * 60)
print("🧠 Агент с долгосрочной памятью (Глава 5)")
print("=" * 60)
if not chapter4_agent.ensure_ollama_running():
return
if not chapter4_agent.model_exists(chapter4_agent.MODEL):
print(f"❌ Модель '{chapter4_agent.MODEL}' не найдена.")
return
chapter4_agent.preload_model(chapter4_agent.MODEL)
chapter4_agent.install_plugins()
# Добавляем правила работы с памятью к промпту
base.SYSTEM_PROMPT = base.SYSTEM_PROMPT + "\n" + MEMORY_RULES
# Показываем, сколько документов уже в базе
try:
from chapter5.src.vectorstore import get_or_create_collection
collection = get_or_create_collection()
doc_count = collection.count()
print(f"\n🧠 Модель: {chapter4_agent.MODEL}")
print(f"💾 Векторная база: ./chroma_db ({doc_count} документов)")
except Exception:
print(f"\n🧠 Модель: {chapter4_agent.MODEL}")
print(f"💾 Векторная база: ./chroma_db")
print("🛑 exit / quit / выход — для выхода.\n")
while True:
try:
user_input = input("Вы > ").strip()
if not user_input:
continue
if user_input.lower() in {"exit", "quit", "выход"}:
print("👋 Пока!")
break
if not chapter4_agent.VERBOSE:
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
```
## 5.7 Особенности маленьких моделей (3B)
Мы прошли через несколько итераций отладки, прежде чем агент заработал
стабильно. Все проблемы были связаны с особенностями модели 3B.
### Few-shot примеры работают лучше абстрактных правил
Модель 3B плохо следует длинным спискам правил, но отлично копирует
конкретные примеры:
```text
❌ Плохо (модель игнорирует):
"КРИТИЧЕСКИ ВАЖНО: После получения результатов от list_memory ты ОБЯЗАН
пересказать ВСЕ записи пользователю. НЕ пропускай записи..."
✅ Хорошо (модель следует):
Пример:
User: Покажи, что ты помнишь
Assistant: {"name": "list_memory", "arguments": {}}
User: Observation from list_memory: Найдено 2 записей...
Assistant: Вот что я помню:
1. [general] Мой проект называется firstAgent
2. [general] Меня зовут Владимир
```
> **Эмпирическое правило: для моделей до 7B — 1 пример работает лучше, чем 5 правил.**
### Агрессивные формулировки вызывают откат к базовому поведению
Когда мы добавили в промпт формулировки «КРИТИЧЕСКИ ВАЖНО», «ты ОБЯЗАН»,
«НИКОГДА», модель 3B **перегрузилась** и откатилась к базовому safety-поведению:
```text
Вы > Покажи, что ты помнишь
🤖 Агент > Извините, но я не могу показать вам информацию из моей
долгосрочной памяти. Это ограничение установлено для безопасности
и конфиденциальности.
```
Модель "вспомнила", что она обучена не раскрывать "внутреннюю память",
и проигнорировала системный промпт. Парадокс: чем жёстче мы давили,
тем сильнее модель сопротивлялась.
**Решение:** спокойные формулировки + явное снятие ограничений:
```text
Показывать сохранённые записи — это нормальная разрешённая операция.
Ограничений безопасности на это НЕТ.
```
### Модель выдумывает параметры инструментов
Если в промпте нет сигнатуры функции, модель придумывает собственные параметры:
```text
Модель вызывает: {"name": "remember", "arguments": {"key": "project_name", "value": "firstAgent"}}
Функция ожидает: remember(content: str, category: str)
Результат: ошибка "missing required argument: content" × 8 итераций
```
Решение — извлечение сигнатуры через `inspect.signature` (Глава 4.2)
и добавление параметров в промпт:
```text
remember — сохраняет важную информацию
Параметры: content (string, обязательно), category (string, по умолчанию: 'general')
```
### Модель склонна переводить на английский
Несмотря на инструкцию «сохраняй на языке пользователя», модель 3B часто
переводит текст на английский. Это не критично для поиска (эмбеддинги
мультиязычные), но стоит добавить пример с русским контентом в промпт.
### Итоговая таблица: что работает с моделью 3B
| Приём | Эффективность |
|---|---|
| Few-shot примеры в промпте | ⭐⭐⭐⭐⭐ |
| Короткие спокойные правила | ⭐⭐⭐⭐ |
| Явное снятие ложных ограничений безопасности | ⭐⭐⭐⭐ |
| Сигнатура функций в описании инструментов | ⭐⭐⭐⭐ |
| Агрессивные запреты («НИКОГДА», «ОБЯЗАН») | ⭐ (вредно!) |
| Длинные списки абстрактных правил | ⭐ (вредно!) |
## 5.8 Уроки отладки: что пошло не так
| Проблема | Причина | Решение |
|---|---|---|
| `ValueError: deprecated configuration` | Старый API ChromaDB | `chromadb.PersistentClient(path=...)` |
| Модель выдумывает `key`/`value` | Нет сигнатуры в промпте | `inspect.signature` + параметры в промпт |
| `list index out of range in query` | Пустой query в ChromaDB | Валидация `if not query` |
| Дубли записей в памяти | `auto_save` + `remember` одновременно | Убрать авто-сохранение |
| Модель отказывается показывать память | Safety-отказ из-за агрессивных формулировок | Few-shot примеры + снятие ограничений |
| Модель переводит на английский | Базовое поведение модели | Пример с русским контентом |
| Обрезанный текст в list_memory | `[:100]` в vectorstore | Убрать обрезку |
## 5.9 Итоги главы
| Критерий | Глава 3 (короткая память) | Глава 5 (RAG) |
|---|---|---|
| Объём памяти | 10 сообщений | Неограничен |
| Поиск | Только по последним | Семантический по всем |
| Между сессиями | Нет | Да (ChromaDB на диске) |
| Структура | Список сообщений | Векторная база с категориями |
| Влияние на контекст | Линейный рост | Constant — только релевантные записи |
### Чек-лист для RAG-агента
- [ ] Установлен `chromadb` и модель `nomic-embed-text`
- [ ] Используется `PersistentClient`, а не старый API
- [ ] Инструменты защищены от пустых/выдуманных параметров
- [ ] Дедупликация через косинусное расстояние < 0.1
- [ ] В промпте есть few-shot примеры, а не только правила
- [ ] Формулировки спокойные, без «КРИТИЧЕСКИ ВАЖНО» и «ОБЯЗАН»

---
# Приложение

## Шпаргалка по командам Ollama

```bash
# Список моделей
ollama list

# Информация о модели
ollama show <model_name>

# Запуск сервера вручную
ollama serve

# Загрузка модели
ollama pull <model_name>

# Удаление модели
ollama rm <model_name>

# Загруженные модели (память)
ollama ps
```

## Частые ошибки и решения

| Ошибка | Причина | Решение |
|---|---|---|
| `ConnectionError` | Ollama не запущена | `ollama serve` или автозапуск в коде |
| Модель не вызывает tools | Текстовый формат вместо нативного | Парсить JSON из текста (Глава 1.5) |
| Медленный первый ответ | Модель загружается в память | `preload_model()` + `keep_alive` |
| Out of memory | Большой `num_ctx` + параллельные запросы | Уменьшить `num_ctx`, поднять `OLLAMA_NUM_PARALLEL` осторожно |
| Агент зацикливается | Модель не может завершить задачу | Уменьшить `MAX_ITERATIONS`, улучшить промпт |
| `TypeError: unexpected keyword` | Модель передала лишний аргумент | `inspect.signature` в `execute_tool` (Глава 4.2) |
| `ValueError: deprecated configuration` | Старый API ChromaDB | `PersistentClient(path=...)` (Глава 5.4) |
| `list index out of range in query` | Пустой query в ChromaDB | Валидация query в инструменте (Глава 5.5) |
| Модель отказывается показывать память | Safety-отказ из-за агрессивных формулировок | Few-shot примеры + снятие ограничений (Глава 5.7) |
| Модель выдумывает параметры инструмента | Нет сигнатуры в промпте | `inspect.signature` + параметры в промпт (Глава 4.2, 5.7) |

## Рекомендуемые модели для агентов

| Задача | Модель | Размер |
|---|---|---|
| Перефразировщик | qwen2.5:1.5b | ~1 GB |
| Агент с инструментами | qwen2.5-coder:3b | ~2 GB |
| Сложные задачи | qwen2.5-coder:7b | ~4.5 GB |
| Критик | qwen2.5:3b | ~2 GB |

## Полезные ссылки

- [Документация Ollama](https://github.com/ollama/ollama)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Qwen2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)

---

> **Дата создания туториала:** 16 августа 2026
> **Версия:** 1.3
> **Автор:** Создано на основе практической сессии с Ollama + Qwen2.5 Coder
```
