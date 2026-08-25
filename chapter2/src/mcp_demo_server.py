"""Минимальный MCP-сервер: два инструмента над одной папкой.

Запуск (обычно его запускает клиент, а не человек):

    python chapter2/src/mcp_demo_server.py <корневая_папка>

Сервер — отдельный процесс, который ничего не знает ни про наш реестр, ни про
Ollama, ни про Главу 2. Он говорит только на JSON-RPC 2.0 и общается через
stdin/stdout: одна строка — одно сообщение.

Отсюда главное правило stdio-транспорта: **в stdout нельзя печатать ничего,
кроме протокола**. Один случайный print("готов") — и клиент получает строку,
которая не парсится как JSON. Всё, что хочется залогировать, идёт в stderr.

Зависимостей нет — только стандартная библиотека. Это тот же сервер, что
`@modelcontextprotocol/server-filesystem`, только на два инструмента и без
единой строчки, которая не нужна для понимания протокола.
"""

import json
import os
import sys

# Версия протокола, которую сервер объявляет в ответ на initialize.
PROTOCOL_VERSION = "2025-06-18"

SERVER_NAME = "chapter2-demo"

# Корень, за пределы которого сервер не выпускает. Песочницу держит СЕРВЕР —
# не агент и не модель. Тот, кто владеет данными, тот и решает, что отдавать.
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())

# Потолок на размер ответа. У клиента есть свой, но полагаться на чужую
# аккуратность в вопросе «сколько текста мне пришлют» — плохая идея.
MAX_CHARS = 4000

# Описания инструментов. Это тот же JSON Schema, который в Главе 2 строит
# декоратор @tool из сигнатуры функции, — только здесь он написан руками,
# потому что источник инструментов внешний.
TOOLS = [
    {
        "name": "list_dir",
        "description": "Показывает список файлов и папок внутри корня сервера.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь относительно корня сервера. Пустая строка — сам корень.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_text_file",
        "description": "Читает текстовый файл внутри корня сервера.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу относительно корня сервера.",
                }
            },
            "required": ["path"],
        },
    },
]


def _resolve(path: str) -> str:
    """Приводит путь к абсолютному и проверяет, что он внутри корня."""
    target = os.path.abspath(os.path.join(ROOT, path or ""))
    # Сравниваем с ROOT + разделитель, иначе `/data_secret` пройдёт проверку
    # на префикс `/data`.
    if target != ROOT and not target.startswith(ROOT + os.sep):
        raise ValueError(f"Путь вне корня сервера: {path}")
    return target


def tool_list_dir(path: str = "") -> str:
    target = _resolve(path)
    if not os.path.isdir(target):
        raise ValueError(f"Не папка: {path}")
    names = sorted(os.listdir(target))
    if not names:
        return "(пусто)"
    lines = []
    for name in names:
        full = os.path.join(target, name)
        mark = "/" if os.path.isdir(full) else ""
        lines.append(f"{name}{mark}")
    return "\n".join(lines)


def tool_read_text_file(path: str) -> str:
    target = _resolve(path)
    if not os.path.isfile(target):
        raise ValueError(f"Файл не найден: {path}")
    with open(target, encoding="utf-8", errors="replace") as f:
        content = f.read(MAX_CHARS + 1)
    if len(content) > MAX_CHARS:
        return content[:MAX_CHARS] + f"\n\n[...обрезано сервером: первые {MAX_CHARS} символов]"
    return content


HANDLERS = {
    "list_dir": tool_list_dir,
    "read_text_file": tool_read_text_file,
}


def handle_initialize(params: dict) -> dict:
    """Рукопожатие: обе стороны называют себя и версию протокола."""
    return {
        "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": "0.1"},
    }


def handle_tools_list(_params: dict) -> dict:
    return {"tools": TOOLS}


def handle_tools_call(params: dict) -> dict:
    """Вызов инструмента.

    Ошибка ИНСТРУМЕНТА — это не ошибка протокола. Файл не найден, путь вне
    корня — обычный result с флагом isError, чтобы модель увидела текст ошибки
    и попробовала иначе. JSON-RPC error остаётся для сломанного запроса.
    """
    name = params.get("name")
    arguments = params.get("arguments") or {}

    handler = HANDLERS.get(name)
    if handler is None:
        return {
            "content": [{"type": "text", "text": f"Неизвестный инструмент: {name}"}],
            "isError": True,
        }

    try:
        text = handler(**arguments)
    except TypeError as e:
        return {
            "content": [{"type": "text", "text": f"Неверные аргументы: {e}"}],
            "isError": True,
        }
    except Exception as e:
        # Флаг isError уже сказал «это ошибка» — в тексте остаётся только суть.
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}

    return {"content": [{"type": "text", "text": text}], "isError": False}


METHODS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "ping": lambda _params: {},
}


def main() -> None:
    # newline="\n" — чтобы на Windows перевод строки не превратился в \r\n:
    # для собеседника это одна строка протокола, а не две.
    sys.stdin.reconfigure(encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue  # мусор в канале — не наша забота, молча пропускаем

        method = message.get("method")
        request_id = message.get("id")

        # У уведомления нет id — на него не отвечают. Самое частое из них:
        # notifications/initialized сразу после рукопожатия.
        if request_id is None:
            continue

        handler = METHODS.get(method)
        if handler is None:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Метод не найден: {method}"},
            }
        else:
            try:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": handler(message.get("params") or {}),
                }
            except Exception as e:  # сломался сам обработчик — это уже протокольная ошибка
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)},
                }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
