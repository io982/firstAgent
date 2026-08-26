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

# Потолки на вывод list_memories. Смысл тот же, что у лимита read_file:
# результат инструмента попадает прямо в контекст, а число фактов на диске
# ничем не ограничено — без потолка память однажды вытеснит сам разговор.
LIST_TOTAL_LIMIT = 1500   # символов на весь список
LIST_LINE_LIMIT = 200     # символов на одну строку «ключ: значение»


# ====================================================================
# ОДИН ФАКТ — ОДИН КЛЮЧ
# ====================================================================
# Реальный лог: в памяти лежит `user_name: io982`, а модель зовёт
# `recall(key="user")` и получает «не найдено». Второй заход — и она кладёт
# рядом `user: io`. Два ключа, два разных ответа на один вопрос, и агент
# уверенно отвечает тем, который попался первым.
#
# Лечится не промптом, а нормализацией: ключ приводится к канону ДО записи
# и ДО чтения. Тогда `имя`, `name`, `user` и `user_name` — это один факт.
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "user_name": ("name", "username", "user", "имя", "имя_пользователя", "как_зовут"),
    "user_email": ("email", "mail", "e_mail", "почта", "электронная_почта"),
}

# Обратный индекс собирается один раз: синоним -> канонический ключ
_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in KEY_ALIASES.items()
    for alias in (canonical, *aliases)
}

# Ключи из few-shot примера в системном промпте. Модель однажды приняла
# пример за данные и сохранила его в память как настоящий факт — это не
# гипотеза, такие строки нашлись в боевом memory.json.
PROMPT_EXAMPLE_KEYS = ("fact1", "fact2")


def normalize_key(key: str) -> str:
    """Приводит ключ к сравнимому виду: регистр, пробелы, дефисы."""
    normalized = str(key).strip().lower()
    for char in (" ", "-", ".", "/"):
        normalized = normalized.replace(char, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def canonical_key(key: str) -> str:
    """Канонический ключ факта: `имя`, `name`, `user` — всё это `user_name`."""
    normalized = normalize_key(key)
    return _ALIAS_TO_CANONICAL.get(normalized, normalized)


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

        stored_key = canonical_key(key)
        if not stored_key:
            return "❌ Ошибка: ключ должен быть непустой строкой."

        self._data[stored_key] = value
        self._save()

        if stored_key != key:
            # Проговариваем подмену: иначе модель считает, что факт лежит под
            # её ключом, и на следующем шаге не найдёт его.
            return f"✅ Запомнил: {stored_key} = {value} (ключ '{key}' приведён к '{stored_key}')"
        return f"✅ Запомнил: {key} = {value}"

    def recall(self, key: str) -> str:
        """
        Извлекает факт из памяти.

        Args:
            key: Ключ для поиска.

        Returns:
            Значение или сообщение об отсутствии.
        """
        stored_key = self._resolve(key)
        if stored_key is not None:
            return f"📖 Найдено: {stored_key} = {self._data[stored_key]}"
        return f"❌ Не найдено: {key}"

    def forget(self, key: str) -> str:
        """
        Удаляет факт из памяти.

        Args:
            key: Ключ для удаления.

        Returns:
            Сообщение о результате операции.
        """
        stored_key = self._resolve(key)
        if stored_key is not None:
            del self._data[stored_key]
            self._save()
            return f"🗑️ Забыл: {stored_key}"
        return f"❌ Не найдено: {key}"

    def list_memories(self) -> str:
        """
        Возвращает список всех сохранённых фактов.

        Вывод ограничен: он уходит прямо в контекст, а память растёт без
        предела. read_file обрезан по той же причине — разница лишь в том,
        что файл вы видите глазами, а список фактов копится незаметно.

        Returns:
            Строка со списком ключей или сообщение об отсутствии.
        """
        if not self._data:
            return "📭 Память пуста."

        lines: list[str] = []
        used = 0
        for key, value in self._data.items():
            line = f"  - {key}: {value}"
            if len(line) > LIST_LINE_LIMIT:
                line = line[:LIST_LINE_LIMIT] + " …(значение обрезано)"
            if lines and used + len(line) > LIST_TOTAL_LIMIT:
                break
            lines.append(line)
            used += len(line)

        result = "📚 Сохранённые факты:\n" + "\n".join(lines)

        hidden = len(self._data) - len(lines)
        if hidden:
            # Не молчим о пропущенном: иначе модель считает показанное
            # всей памятью и отвечает «больше я о вас ничего не знаю».
            result += (
                f"\n  […ещё {hidden} фактов не показано: список не помещается "
                f"в контекст. Доставай их по ключу через recall]"
            )
        return result

    def _resolve(self, key: str) -> str | None:
        """Находит настоящий ключ факта: точный, потом канонический.

        Точное совпадение проверяется первым — в памяти могли остаться ключи,
        записанные до нормализации, и терять их из-за неё было бы обидно.
        """
        if key in self._data:
            return key
        target = canonical_key(key)
        for stored in self._data:
            if canonical_key(stored) == target:
                return stored
        return None

    def duplicates(self) -> dict[str, list[str]]:
        """Ключи, которые означают одно и то же: {канон: [ключи]}.

        Нормализация закрывает будущие записи, но старые файлы уже разъехались.
        Показать расхождение — работа для человека, автоматически «починить»
        такое нельзя: неизвестно, какое из двух значений верное.
        """
        groups: dict[str, list[str]] = {}
        for key in self._data:
            groups.setdefault(canonical_key(key), []).append(key)
        return {canon: keys for canon, keys in groups.items() if len(keys) > 1}

    def suspicious_keys(self) -> list[str]:
        """Факты, похожие на строки из few-shot примера, а не на данные."""
        return [key for key in self._data if normalize_key(key) in PROMPT_EXAMPLE_KEYS]

    def items(self) -> dict[str, Any]:
        """Возвращает копию всех фактов.

        В отличие от list_memories(), здесь нет ни потолков, ни форматирования:
        это данные для кода, а не текст для модели. Копия, а не сам словарь —
        чтобы посторонний код не менял память в обход remember/forget.
        """
        return dict(self._data)

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
