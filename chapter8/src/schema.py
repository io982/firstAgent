"""Описание инструментов в формате JSON Schema для нативного tools API.

Реестр Главы 4 уже хранит параметры каждого инструмента: их извлекает
`_extract_parameters` через `inspect.signature`. Здесь эти же данные
перекладываются в формат, который понимает Ollama.

СТАТУС: каркас Главы 8. Реализации нет.
"""

from chapter4.src import tools as registry


def tool_to_schema(name: str, entry: dict) -> dict:
    """Превращает одну запись реестра в JSON Schema одного инструмента.

    Ожидаемый результат:
        {"type": "function", "function": {
            "name": ..., "description": ...,
            "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
    """
    raise NotImplementedError("Глава 8: перевод записи реестра в JSON Schema")


def registry_to_schemas() -> list:
    """Собирает описания всех инструментов реестра для поля `tools` запроса."""
    raise NotImplementedError("Глава 8: сборка списка схем из TOOL_REGISTRY")


def _python_type_to_json(type_name: str) -> str:
    """Переводит тип из реестра ("string", "integer", ...) в тип JSON Schema.

    Реестр Главы 4 уже хранит типы почти в нужном виде — отдельная функция
    нужна для случая "any", которого в JSON Schema нет.
    """
    raise NotImplementedError("Глава 8: сопоставление типов")


__all__ = ["registry", "tool_to_schema", "registry_to_schemas"]
