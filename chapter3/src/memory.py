"""
Долгосрочная память агента (пункт 3.4 ROADMAP).

Простое персистентное хранилище фактов в формате ключ-значение.
Агент использует инструменты remember/recall/forget для управления памятью.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Инструменты памяти регистрируются в том же реестре, что и инструменты Главы 2.
# Никакого второго реестра и второго диспетчера: один @tool на всех.
from chapter2.src.tools import tool

# Путь к файлу памяти (по умолчанию в директории главы)
DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "memory.json"


class LongTermMemory:
    """
    Персистентное хранилище фактов.

    Пример использования:
        memory = LongTermMemory()
        memory.remember("user_name", "Алексей")
        print(memory.recall("user_name"))  # "Алексей"
        memory.forget("user_name")
    """

    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_MEMORY_PATH
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Загружает память из файла. Если файла нет — создаёт пустую память."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"⚠️ Ошибка загрузки памяти: {e}. Создаю новую память.")
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Сохраняет память в файл."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠️ Ошибка сохранения памяти: {e}")

    def remember(self, key: str, value: Any) -> str:
        """
        Сохраняет факт в память.

        Args:
            key: Уникальный ключ (например, "user_name", "favorite_color")
            value: Значение (строка, число, список и т.д.)

        Returns:
            Сообщение о результате операции.
        """
        if not key or not isinstance(key, str):
            return "❌ Ошибка: ключ должен быть непустой строкой."

        self._data[key] = value
        self._save()
        return f"✅ Запомнил: {key} = {value}"

    def recall(self, key: str) -> str:
        """
        Извлекает факт из памяти.

        Args:
            key: Ключ для поиска.

        Returns:
            Значение или сообщение об отсутствии.
        """
        if key in self._data:
            return f"📖 Найдено: {key} = {self._data[key]}"
        return f"❌ Не найдено: {key}"

    def forget(self, key: str) -> str:
        """
        Удаляет факт из памяти.

        Args:
            key: Ключ для удаления.

        Returns:
            Сообщение о результате операции.
        """
        if key in self._data:
            del self._data[key]
            self._save()
            return f"🗑️ Забыл: {key}"
        return f"❌ Не найдено: {key}"

    def list_memories(self) -> str:
        """
        Возвращает список всех сохранённых фактов.

        Returns:
            Строка со списком ключей или сообщение об отсутствии.
        """
        if not self._data:
            return "📭 Память пуста."

        items = [f"  - {k}: {v}" for k, v in self._data.items()]
        return "📚 Сохранённые факты:\n" + "\n".join(items)

    def clear_all(self) -> str:
        """Очищает всю память."""
        self._data = {}
        self._save()
        return "🧹 Вся память очищена."


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР (для использования в инструментах)
# ====================================================================

_memory_instance: LongTermMemory | None = None


def get_memory() -> LongTermMemory:
    """Возвращает глобальный экземпляр памяти (singleton)."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance


# ====================================================================
# ИНСТРУМЕНТЫ ДЛЯ АГЕНТА (интеграция с реестром из Главы 2)
# ====================================================================

# Имя функции становится именем инструмента для модели, а первая строка
# docstring — его описанием в системном промпте. Единый источник истины.

@tool
def remember(key: str, value: str) -> str:
    """Сохраняет факт в долгосрочную память под коротким ключом.

    Используй, когда пользователь сообщает важную информацию о себе или задаче.

    Args:
        key: Краткий ключ (например, "user_name", "project_deadline")
        value: Значение факта
    """
    return get_memory().remember(key, value)


@tool
def recall(key: str) -> str:
    """Извлекает факт из долгосрочной памяти по точному ключу.

    Используй, когда нужно вспомнить ранее сохранённую информацию.

    Args:
        key: Ключ для поиска
    """
    return get_memory().recall(key)


@tool
def forget(key: str) -> str:
    """Удаляет один факт из долгосрочной памяти по ключу.

    Используй, когда информация устарела или неверна.

    Args:
        key: Ключ для удаления
    """
    return get_memory().forget(key)


@tool
def list_memories() -> str:
    """Показывает все сохранённые факты. Вызывай, чтобы узнать, что уже известно."""
    return get_memory().list_memories()


@tool
def clear_all() -> str:
    """Полностью очищает всю долгосрочную память.

    Используй ТОЛЬКО при прямом и явном запросе пользователя удалить ВСЕ факты.
    """
    return get_memory().clear_all()
