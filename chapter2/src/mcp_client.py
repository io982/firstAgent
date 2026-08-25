"""MCP-клиент: чужие инструменты в нашем реестре.

Реестр Главы 2 закрывает свои инструменты — те, что мы написали сами и обернули
декоратором `@tool`. MCP (Model Context Protocol) закрывает чужие: файловую
систему, git, поиск, внутренние API. Протокол один и тот же для всех серверов,
а описание инструмента приходит в том же JSON Schema, который мы строим из
сигнатуры функции.

Ключевая мысль главы: MCP добавляет ИСТОЧНИК инструментов, а не второй механизм.
Модель по-прежнему видит один список, агент по-прежнему зовёт `execute_tool`.

    TOOL_REGISTRY (свои)  ─┐
                           ├──► единый список схем ──► LLM
    MCP servers (чужие)  ──┘

Здесь нет ни одной внешней зависимости: транспорт stdio — это subprocess,
а сообщения — это JSON-RPC 2.0, по одному объекту на строку. Официальный SDK
(`pip install mcp`) делает то же самое плюс SSE/HTTP-транспорт, переподключения
и типы; но пока не видно, что внутри, MCP выглядит магией, а не сокетом.
"""

import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter2.src.tools import READ_FILE_LIMIT, TOOL_REGISTRY

# Версия протокола, которую мы объявляем при рукопожатии. Сервер вправе
# ответить другой — тогда решать нам: работать или разорвать соединение.
PROTOCOL_VERSION = "2025-06-18"

# Сколько ждём ответа на один запрос. Чужой процесс может не ответить никогда,
# и без таймаута агент повиснет молча — худший вид отказа.
DEFAULT_TIMEOUT = 15.0

# Потолок на вывод — тот же, что у read_file. Результат чужого инструмента
# уезжает в тот же контекст, что и результат своего, и лимит у контекста один.
MCP_OUTPUT_LIMIT = READ_FILE_LIMIT

# Путь к демо-серверу из этой же главы — чтобы главу можно было пройти
# без Node.js и без интернета.
DEMO_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_demo_server.py")


class MCPError(RuntimeError):
    """Ошибка транспорта или протокола: процесс умер, таймаут, JSON-RPC error."""


def demo_server_command(root: str) -> list[str]:
    """Команда запуска демо-сервера этой главы над указанной папкой."""
    return [sys.executable, "-u", DEMO_SERVER, os.path.abspath(root)]


def _split_command(text: str) -> list[str]:
    """Разбирает командную строку сервера в список аргументов.

    `shlex.split` по умолчанию работает по правилам POSIX, где обратный слэш —
    это экранирование. На Windows он молча съедает разделители пути, и
    `C:\\Python\\python.exe` превращается в `C:Pythonpython.exe`.
    """
    if os.name == "nt":
        # posix=False сохраняет слэши, но оставляет кавычки внутри токенов
        return [part.strip('"') for part in shlex.split(text, posix=False)]
    return shlex.split(text)


def _resolve_executable(command: list[str]) -> list[str]:
    """Чинит самую частую боль на Windows: `npx` там называется `npx.cmd`.

    Popen без shell=True ищет ровно то имя, что ему дали, и на `npx` падает
    с FileNotFoundError, хотя в PATH лежит `npx.cmd`.
    """
    if not command:
        raise MCPError("Пустая команда запуска MCP-сервера")
    found = shutil.which(command[0])
    if found:
        return [found, *command[1:]]
    return command


class MCPClient:
    """Один MCP-сервер: процесс, канал и три метода, которые нам нужны.

    Из всего протокола агенту достаточно трёх: `initialize` (рукопожатие),
    `tools/list` (что умеешь) и `tools/call` (сделай). Остальное — ресурсы,
    промпты, подписки — существует, но реестру инструментов не нужно.
    """

    def __init__(self, command: list[str], label: str, timeout: float = DEFAULT_TIMEOUT,
                 cwd: str | None = None):
        self.command = command
        self.label = label
        self.timeout = timeout
        self.cwd = cwd
        self.server_info: dict[str, Any] = {}
        self._proc: subprocess.Popen | None = None
        self._stdin: io.TextIOWrapper | None = None
        self._stdout: io.TextIOWrapper | None = None
        self._incoming: queue.Queue = queue.Queue()
        self._next_id = 0

    # --- жизненный цикл процесса -------------------------------------

    def start(self) -> "MCPClient":
        """Запускает сервер и проходит рукопожатие."""
        self._proc = subprocess.Popen(
            _resolve_executable(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr — журнал сервера, и он НЕ часть протокола. Сюда сервер
            # пишет свои логи; если смешать его со stdout, первая же строчка
            # лога сломает разбор сообщений.
            stderr=subprocess.DEVNULL,
            cwd=self.cwd,
            bufsize=0,
        )
        # Оборачиваем каналы вручную, а не через text=True: нужен newline="\n".
        # В текстовом режиме Python на Windows превратил бы наш "\n" в "\r\n",
        # и сервер получил бы лишний символ в конце каждой строки протокола.
        self._stdin = io.TextIOWrapper(self._proc.stdin, encoding="utf-8", newline="\n",
                                       write_through=True)
        self._stdout = io.TextIOWrapper(self._proc.stdout, encoding="utf-8", newline="\n",
                                        errors="replace")

        # Читаем ответы в отдельном потоке: readline() блокирует, а select()
        # на пайпах не работает на Windows. Поток + очередь — самый простой
        # способ получить таймаут, который ведёт себя одинаково везде.
        threading.Thread(target=self._reader_loop, daemon=True).start()

        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "chapter2-agent", "version": "0.1"},
        })
        self.server_info = result.get("serverInfo", {})

        # Уведомление без id: ответа на него не будет и ждать его не нужно.
        self._notify("notifications/initialized", {})
        return self

    def close(self) -> None:
        """Закрывает stdin и ждёт; если сервер не ушёл сам — убивает."""
        if self._proc is None:
            return
        try:
            if self._stdin is not None:
                self._stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- транспорт ----------------------------------------------------

    def _reader_loop(self) -> None:
        """Складывает всё, что приходит из stdout сервера, в очередь."""
        assert self._stdout is not None
        for line in self._stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._incoming.put(json.loads(line))
            except json.JSONDecodeError:
                # Сервер напечатал в stdout что-то не по протоколу.
                # Пропускаем: ответ на наш запрос всё равно придёт отдельной строкой.
                continue

    def _send(self, message: dict) -> None:
        if self._proc is None or self._stdin is None:
            raise MCPError(f"MCP-сервер '{self.label}' не запущен")
        try:
            self._stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        except (BrokenPipeError, ValueError) as e:
            raise MCPError(f"MCP-сервер '{self.label}' закрыл канал: {e}") from e

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict) -> dict:
        """Запрос-ответ по id, с таймаутом и проверкой, что процесс ещё жив."""
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(
                    f"MCP-сервер '{self.label}' не ответил на '{method}' за {self.timeout:g} с"
                )
            try:
                message = self._incoming.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                # Пустая очередь — повод проверить, жив ли вообще собеседник.
                if self._proc is not None and self._proc.poll() is not None:
                    raise MCPError(
                        f"MCP-сервер '{self.label}' завершился с кодом {self._proc.returncode}"
                    ) from None
                continue

            # Чужие сообщения (уведомления сервера, запросы к клиенту) нам
            # неинтересны: ждём строго свой id.
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                raise MCPError(f"{method}: {error.get('message')} (код {error.get('code')})")
            return message.get("result") or {}

    # --- три метода протокола ----------------------------------------

    def list_tools(self) -> list[dict]:
        """Список инструментов сервера. Ответ может приходить страницами."""
        tools: list[dict] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """Вызов инструмента. Возвращает текст — то же, что и наши функции."""
        try:
            result = self._request("tools/call", {"name": name, "arguments": arguments})
        except MCPError as e:
            # Упавший сервер не должен ронять агента: ошибка уходит в контекст
            # тем же способом, что и ошибка своего инструмента.
            return f"Ошибка MCP: {e}"

        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                # Картинки и ресурсы в контекст текстовой модели не положишь —
                # честно говорим, что содержимое пропущено.
                parts.append(f"[{item.get('type')}: содержимое не текстовое, пропущено]")
        text = "\n".join(parts).strip() or "(пустой ответ)"

        if result.get("isError"):
            text = f"Ошибка инструмента MCP: {text}"

        if len(text) > MCP_OUTPUT_LIMIT:
            # Обрезку проговариваем вслух — по той же причине, что и в read_file.
            text = (
                text[:MCP_OUTPUT_LIMIT]
                + f"\n\n[...ответ MCP обрезан: первые {MCP_OUTPUT_LIMIT} символов]"
            )
        return text


# === СКЛЕЙКА С РЕЕСТРОМ ГЛАВЫ 2 ===

def _safe_name(raw: str) -> str:
    """Имя инструмента уходит в enum схемы и в промпт — чистим его.

    У MCP-серверов имена бывают через дефис (`read-file`), а нам нужен
    идентификатор без сюрпризов.
    """
    name = re.sub(r"[^0-9a-zA-Z_]", "_", raw).strip("_")
    return name or "tool"


def _make_caller(client: MCPClient, remote_name: str, schema: dict):
    """Оборачивает удалённый инструмент в обычную функцию реестра.

    Своим инструментам имена аргументов проверяет Python: лишний ключ — TypeError,
    и `execute_tool` превращает его в подсказку модели. У чужого инструмента
    сигнатуры нет, поэтому ту же проверку делаем сами — иначе модель, придумав
    `filename` вместо `path`, получит от сервера невнятное «неверные аргументы»
    вместо списка ожидаемых имён.
    """
    expected = list(schema.get("properties", {}).keys())
    required = list(schema.get("required", []))

    def call(**arguments: Any) -> str:
        unknown = [k for k in arguments if k not in expected]
        missing = [k for k in required if k not in arguments]
        if unknown or missing:
            return (
                f"Ошибка валидации аргументов для '{remote_name}'. "
                f"Ты передал: {list(arguments.keys())}. Ожидались строго: {expected}."
            )
        return client.call_tool(remote_name, arguments)

    return call


def register_mcp_tools(client: MCPClient, prefix: str | None = None,
                       only: list[str] | None = None) -> list[str]:
    """Кладёт инструменты MCP-сервера в общий реестр Главы 2.

    После этого вызова разницы между своим и чужим инструментом нет нигде:
    ни в промпте (`describe_tools`), ни в схеме (`build_response_schema`),
    ни в диспетчере (`execute_tool`).

    `only` — список нужных имён (в терминах сервера). Не придирчивость, а
    экономия: каждый инструмент занимает строку в промпте и позицию в enum,
    а типичный готовый сервер отдаёт их десятками. Модели на 3B незачем
    видеть тридцать способов работы с git, если агенту нужен один.

    Возвращает список имён, под которыми инструменты легли в реестр.
    """
    prefix = _safe_name(prefix or client.label)
    registered = []

    for remote in client.list_tools():
        remote_name = remote.get("name")
        if not remote_name:
            continue
        if only is not None and remote_name not in only:
            continue

        # Префикс сервера — не украшение: `read_file` есть и у нас, и у
        # половины MCP-серверов. Без префикса чужой инструмент молча затрёт свой.
        local_name = f"{prefix}_{_safe_name(remote_name)}"
        if local_name in TOOL_REGISTRY:
            suffix = 2
            while f"{local_name}_{suffix}" in TOOL_REGISTRY:
                suffix += 1
            local_name = f"{local_name}_{suffix}"

        input_schema = dict(remote.get("inputSchema") or {})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        input_schema.setdefault("required", [])

        TOOL_REGISTRY[local_name] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": local_name,
                    "description": remote.get("description") or "Инструмент MCP-сервера",
                    "parameters": input_schema,
                },
            },
            "function": _make_caller(client, remote_name, input_schema),
            # Метка происхождения: по ней видно, что инструмент не наш,
            # и по ней же его можно снять с регистрации.
            "mcp": {"client": client, "remote_name": remote_name},
        }
        registered.append(local_name)

    return registered


def unregister_mcp_tools(client: MCPClient | None = None) -> list[str]:
    """Убирает из реестра инструменты MCP (одного сервера или всех)."""
    removed = []
    for name, info in list(TOOL_REGISTRY.items()):
        origin = info.get("mcp")
        if not origin:
            continue
        if client is None or origin["client"] is client:
            del TOOL_REGISTRY[name]
            removed.append(name)
    return removed


def connect_mcp_servers(spec: str, timeout: float = DEFAULT_TIMEOUT) -> list[MCPClient]:
    """Поднимает серверы по строке из переменной окружения AGENT_MCP.

    Формат — команды через `;`, метка сервера отделяется знаком `=`:

        AGENT_MCP="demo"
        AGENT_MCP="fs=npx -y @modelcontextprotocol/server-filesystem ."
        AGENT_MCP="demo; git=npx -y @modelcontextprotocol/server-git"

    Сервер, который не поднялся, не должен ронять агента: печатаем причину
    и работаем с тем, что есть.
    """
    clients = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue

        label, _, command_text = chunk.partition("=")
        if not command_text:
            label, command_text = "mcp", chunk
        label, command_text = label.strip(), command_text.strip()

        if command_text == "demo":
            command = demo_server_command(os.getcwd())
            label = label if label != "mcp" else "demo"
        else:
            command = _split_command(command_text)

        client = MCPClient(command, label=label, timeout=timeout)
        try:
            client.start()
            names = register_mcp_tools(client)
        except (MCPError, OSError) as e:
            print(f"⚠️ MCP-сервер '{label}' не подключён: {e}")
            client.close()
            continue

        server_name = client.server_info.get("name", "?")
        print(f"🔌 MCP '{label}' ({server_name}): инструментов — {len(names)}: {', '.join(names)}")
        clients.append(client)

    return clients
