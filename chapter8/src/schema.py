"""Описание инструментов в формате JSON Schema для нативного tools API.

Реестр Главы 4 уже хранит параметры каждого инструмента: их извлекает
`_extract_parameters` через `inspect.signature`. Здесь эти же данные
перекладываются в формат, который понимает Ollama:

    {"type": "function", "function": {
        "name": "calculator",
        "description": "безопасно считает арифметические выражения",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}}

Ничего нового собирать не нужно — решение Главы 4 окупается второй раз.
"""

import inspect

from chapter4.src import tools as registry

# Типы реестра почти совпадают с типами JSON Schema. Исключение — "any":
# такого типа в схеме нет, и поле просто остаётся без ограничения.
JSON_TYPES = {"string", "integer", "number", "boolean"}


def _python_type_to_json(type_name: str) -> str | None:
    """Переводит тип из реестра в тип JSON Schema.

    Возвращает None для "any" — значит, ограничение по типу не ставим.
    """
    return type_name if type_name in JSON_TYPES else None


def _skipped_parameters(func) -> set:
    """Имена параметров, которых не должно быть в схеме.

    `_extract_parameters` Главы 4 перечисляет всё, что видит в сигнатуре,
    включая `**kwargs`. Для промпта это безвредно, а в JSON Schema
    превратилось бы в обязательное поле с именем "kwargs", которого модель
    не понимает и которое она начнёт заполнять чем попало.
    """
    skipped = set()
    for name, param in inspect.signature(func).parameters.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            skipped.add(name)
    return skipped


def tool_to_schema(name: str, entry: dict) -> dict:
    """Превращает одну запись реестра в описание инструмента."""
    properties = {}
    required = []
    skipped = _skipped_parameters(entry["func"])

    for param_name, info in entry.get("parameters", {}).items():
        if param_name in skipped:
            continue

        field = {}
        json_type = _python_type_to_json(info.get("type", "any"))
        if json_type:
            field["type"] = json_type

        if not info.get("required", True):
            field["description"] = f"по умолчанию: {info.get('default')!r}"
        else:
            required.append(param_name)

        properties[param_name] = field

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": entry["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def registry_to_schemas() -> list:
    """Собирает описания всех инструментов реестра для поля `tools` запроса."""
    return [tool_to_schema(name, entry) for name, entry in registry.TOOL_REGISTRY.items()]


__all__ = ["registry", "tool_to_schema", "registry_to_schemas"]
