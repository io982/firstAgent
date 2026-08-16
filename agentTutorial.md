# agentTutorial.md

# 🤖 Туториал: Создаём локального ИИ-агента с Ollama

> Полный курс по созданию агентов с нуля до production-ready системы.

## 📚 Содержание

| Глава | Тема | Код |
|-------|------|-----|
| [Глава 1](chapter1/) | Базовый агент с ReAct-паттерном | [`chapter1/agent.py`](chapter1/agent.py) |
| [Глава 2](chapter2/) | Мульти-агентная система | [`chapter2/paraphraser.py`](chapter2/paraphraser.py) |
| [Глава 3](chapter3/) | Контекст, память и производительность | [`chapter3/agent.py`](chapter3/agent.py) |
| [Глава 4](chapter4/) | Система плагинов: расширяем инструменты | [`chapter4/src/tools.py`](chapter4/src/tools.py) |

## 🎯 Что ты научишься делать

- ✅ Создавать агентов с ReAct-паттерном
- ✅ Работать с инструментами (tools/function calling)
- ✅ Строить мульти-агентные системы
- ✅ Оптимизировать память и производительность
- ✅ Создавать расширяемую систему плагинов
- ✅ Добавлять новые инструменты за 5 строк кода

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

```markdown
# 🤖 Туториал: Создаём локального ИИ-агента с Ollama

> **Цель туториала:** создать работающего локального агента с инструментами, 
> добавить к нему мульти-агентную архитектуру и разобраться в тонкостях 
> контекста, памяти и производительности.

> **Требования:**
> - Установленная [Ollama](https://ollama.com)
> - Python 3.10+
> - Любая модель Qwen2.5 Coder или аналогичная с поддержкой tools

---

## 📑 Содержание

1. [Глава 1. Создание простого агента](#глава-1-создание-простого-агента)
2. [Глава 2. Добавление распознавателя (перефразировщика)](#глава-2-добавление-распознавателя)
3. [Глава 3. Контекст, память и производительность](#глава-3-контекст-память-и-производительность)
4. [Приложение. Шпаргалки и частые ошибки](#приложение)

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
> **Версия:** 1.0
> **Автор:** Создано на основе практической сессии с Ollama + Qwen2.5 Coder
```

Файл готов. Сохраните его как `agentTutorial.md` в папке проекта:

```powershell
notepad agentTutorial.md
```

и вставьте содержимое выше.

Туториал включает:
- **Глава 1** — от проверки Ollama до полного рабочего агента с инструментами
- **Глава 2** — мульти-агентная архитектура с перефразировщиком
- **Глава 3** — контекст, память, параллельность, оптимизация
- **Приложение** — шпаргалки, частые ошибки, рекомендации по моделям