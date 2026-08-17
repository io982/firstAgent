import json
import os
import ast
import operator
import re
import subprocess
import sys
import time
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_BASE = "http://localhost:11434"
# Модель по умолчанию — кастомная Q5-сборка qwen2.5-coder:3b.
# Другая модель подключается через переменную окружения, править код не нужно:
#   Windows (PowerShell): $env:AGENT_MODEL = "qwen2.5-coder:3b"
#   Linux / macOS:        export AGENT_MODEL=qwen2.5-coder:3b
MODEL = os.getenv("AGENT_MODEL", "qwen2_5coder3b_q5:latest")
# Флаг режима вывода
# True  = подробный режим (видно все итерации, вызовы инструментов)
# False = чистый режим (только финальный ответ)
VERBOSE = False

MAX_ITERATIONS = 8

# Сколько держать модель в памяти после последнего запроса
# "5m" = 5 минут, "0" = выгружать сразу, "-1" = никогда не выгружать
KEEP_ALIVE = "5m"

KNOWN_TOOLS = {
    "calculator",
    "list_directory",
    "read_file",
    "search_in_file"
}


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
1. Если нужен инструмент, ответь ТОЛЬКО JSON-вызовом или несколькими JSON-вызовами.
2. Не добавляй пояснения до или после JSON при вызове инструмента.
3. После получения Observation проанализируй результат.
4. Если нужно вызвать ещё инструмент — снова верни JSON.
5. Когда готов дать финальный ответ пользователю, ответь обычным текстом без JSON.
6. Никогда не выдумывай результаты инструментов.
""".strip()


# ====================================================================
# АВТОЗАПУСК OLLAMA
# ====================================================================

def is_ollama_running() -> bool:
    """Проверяет, отвечает ли сервер Ollama на порту 11434."""
    try:
        response = requests.get(OLLAMA_BASE, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server():
    """Запускает ollama serve как фоновый процесс."""
    print("🚀 Запускаю сервер Ollama...")
    
    # В Windows ollama обычно в PATH
    # Используем CREATE_NEW_PROCESS_GROUP, чтобы сервер жил отдельно от нашего скрипта
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        # Скрываем окно процесса
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    
    try:
        # Запускаем в фоне, не блокируя наш скрипт
        process = subprocess.Popen(
            ["ollama", "serve"],
            **kwargs
        )
        print(f"✅ Сервер Ollama запущен (PID: {process.pid})")
        return True
    except FileNotFoundError:
        print("❌ Ошибка: команда 'ollama' не найдена.")
        print("   Убедись, что Ollama установлена и добавлена в PATH.")
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
            print("✅ Сервер Ollama готов к работе!")
            return True
        time.sleep(1)
    
    print("❌ Сервер Ollama не запустился за отведённое время.")
    return False


def ensure_ollama_running() -> bool:
    """Гарантирует, что сервер Ollama запущен."""
    if is_ollama_running():
        print("✅ Сервер Ollama уже работает.")
        return True
    
    if not start_ollama_server():
        return False
    
    return wait_for_ollama()


def list_installed_models() -> list:
    """Возвращает имена моделей, установленных в Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if response.status_code != 200:
            return []
        return [m.get("name", "") for m in response.json().get("models", [])]
    except Exception:
        return []


def model_exists(model_name: str) -> bool:
    """Проверяет, установлена ли модель в Ollama.

    Имя без тега ("qwen2.5-coder") подходит к любому тегу.
    Имя с тегом ("qwen2.5-coder:3b") требует именно этот тег,
    чтобы установленная 7b не выдавала себя за запрошенную 3b.
    """
    installed = list_installed_models()

    for name in installed:
        if name == model_name:
            return True
        # Ollama показывает "модель:latest", запрашивать можно без тега
        if ":" not in model_name and name == f"{model_name}:latest":
            return True
        if ":" not in model_name and name.startswith(f"{model_name}:"):
            return True

    return False


def model_not_found_message(model_name: str) -> str:
    """Подсказка, что делать, если модель не найдена."""
    installed = list_installed_models()
    lines = [f"❌ Модель '{model_name}' не найдена."]
    if installed:
        lines.append("   Установлены: " + ", ".join(installed))
        lines.append("   Выбрать другую: задайте переменную окружения AGENT_MODEL")
        lines.append('   PowerShell: $env:AGENT_MODEL = "имя_модели"')
    lines.append(f"   Установить эту: ollama pull {model_name}")
    return "\n".join(lines)


def preload_model(model_name: str):
    """
    "Прогревает" модель: загружает её в память пустым запросом,
    чтобы первый настоящий запрос был быстрым.
    """
    print(f"📦 Предзагрузка модели '{model_name}' в память...")
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": "",  # пустой промпт
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1}  # генерировать почти ничего
            },
            timeout=120  # первая загрузка может быть долгой
        )
        
        if response.status_code == 200:
            print(f"✅ Модель '{model_name}' готова к работе!")
        else:
            print(f"⚠️ Модель вернула статус {response.status_code}")
    except requests.exceptions.Timeout:
        print("⚠️ Предзагрузка заняла слишком много времени (модель большая).")
        print("   Она продолжит загружаться при первом запросе.")
    except Exception as e:
        print(f"⚠️ Ошибка предзагрузки: {e}")


# ====================================================================
# Безопасный калькулятор и другие инструменты (как раньше)
# ====================================================================

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
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return ALLOWED_BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("Недопустимый унарный оператор")
        operand = _safe_eval_node(node.operand)
        return ALLOWED_UNARY_OPS[op_type](operand)

    raise ValueError("Недопустимое выражение")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree)
        return str(result)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as e:
        return f"Ошибка калькулятора: {e}"


def list_directory(path: str = ".") -> str:
    path = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(path):
        return f"Путь не найден: {path}"

    if not os.path.isdir(path):
        return f"Это не директория: {path}"

    try:
        entries = []
        for name in sorted(os.listdir(path))[:200]:
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                entries.append(f"[DIR]  {name}/")
            else:
                entries.append(f"[FILE] {name}")

        if not entries:
            return "Директория пуста"
        return "\n".join(entries)
    except Exception as e:
        return f"Ошибка чтения директории: {e}"


def read_file(path: str, max_chars: int = 8000) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return f"Файл не найден: {path}"
    if not os.path.isfile(path):
        return f"Это не файл: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... [файл обрезан] ..."
        return content
    except Exception as e:
        return f"Ошибка чтения файла: {e}"


def search_in_file(path: str, query: str, max_results: int = 20) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return f"Файл не найден: {path}"
    if not os.path.isfile(path):
        return f"Это не файл: {path}"
    if not query:
        return "Не указан текст для поиска"
    try:
        results = []
        query_lower = query.lower()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                if query_lower in line.lower():
                    results.append(f"{line_number}: {line.rstrip()}")
                    if len(results) >= max_results:
                        results.append(f"... [показаны первые {max_results}] ...")
                        break
        if not results:
            return f"Ничего не найдено: {query}"
        return "\n".join(results)
    except Exception as e:
        return f"Ошибка поиска в файле: {e}"


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
        return f"Ошибка выполнения инструмента {name}: {e}"


def normalize_potential_tool(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if name in KNOWN_TOOLS:
        return {"name": name, "arguments": obj.get("arguments", {})}
    function_obj = obj.get("function")
    if isinstance(function_obj, dict):
        name = function_obj.get("name")
        if name in KNOWN_TOOLS:
            return {"name": name, "arguments": function_obj.get("arguments", {})}
    return None


def extract_tool_calls(text: str):
    calls = []
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for block in code_blocks:
        try:
            obj = json.loads(block)
            tool_call = normalize_potential_tool(obj)
            if tool_call:
                calls.append(tool_call)
        except Exception:
            pass
    if calls:
        return calls

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            tool_call = normalize_potential_tool(obj)
            if tool_call:
                calls.append(tool_call)
            pos = end
        except json.JSONDecodeError:
            # Fallback: пробуем починить неэкранированные обратные слэши (Windows пути)
            try:
                sub_text = text[start:]
                # Заменяем одинарные слэши на двойные
                fixed_sub = sub_text.replace('\\', '\\\\')
                obj, end = decoder.raw_decode(fixed_sub, 0)
                tc = normalize_potential_tool(obj)
                if tc:
                    calls.append(tc)
                pos = start + end
            except Exception:
                pos = start + 1
    return calls


def request_model(messages: list) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,  # удерживаем модель в памяти!
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096    # ← было 65536, стало 4096
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
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
            print("[Model raw answer]")
            print(content[:500])

        tool_calls = extract_tool_calls(content)

        if not tool_calls:
            return content.strip()

        messages.append({"role": "assistant", "content": content})

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = normalize_arguments(tool_call.get("arguments", {}))

            if VERBOSE:
                print(f"[Agent] Вызываю инструмент: {tool_name}")
                print(f"[Agent] Аргументы: {tool_args}")

            tool_result = execute_tool(tool_name, tool_args)

            if VERBOSE:
                preview = tool_result[:300].replace("\n", " ")
                print(f"[Tool] Результат: {preview}")

            messages.append({
                "role": "user",
                "content": f"Observation from {tool_name}: {tool_result}"
            })

    return "Агент достиг лимита итераций и не смог завершить задачу."

# ====================================================================
# ГЛАВНАЯ ФУНКЦИЯ С АВТОЗАПУСКОМ
# ====================================================================

def main():
    if not ensure_ollama_running():
        print("\n❌ Не удалось запустить Ollama. Завершение работы.")
        return
    
    print(f"\n🔍 Проверяю модель '{MODEL}'...")
    if not model_exists(MODEL):
        print(model_not_found_message(MODEL))
        return
    print(f"✅ Модель готова.")
    
    preload_model(MODEL)
    
    if VERBOSE:
        print("\n" + "=" * 60)
        print(f"🧠 Модель: {MODEL}")
        print(f"💾 Keep alive: {KEEP_ALIVE}")
        print(f"🔊 Режим: подробный (VERBOSE={VERBOSE})")
        print("🛑 exit / quit / выход — для выхода.")
        print("=" * 60 + "\n")
    else:
        print(f"🤖 Агент готов. Модель: {MODEL}")
        print("🛑 exit / quit / выход — для выхода.\n")

    while True:
        try:
            user_input = input("Вы > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "выход"}:
                print("👋 Пока!")
                break

            # В чистом режиме показываем индикатор работы
            if not VERBOSE:
                print("⏳ Думаю...")

            answer = ask_agent(user_input)

            # Чистый вывод: только финальный ответ
            print("\n🤖 Агент >")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\n⚠️ Остановлено.")
            break

        except requests.exceptions.ConnectionError:
            print("\n⚠️ Связь потеряна. Восстанавливаю...")
            if ensure_ollama_running():
                print("✅ Восстановлено. Повтори запрос.")
            else:
                print("❌ Не удалось восстановить.")
                break

        except Exception as e:
            print(f"\nОшибка: {e}")

if __name__ == "__main__":
    main()