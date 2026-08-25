"""
Core memory — короткий блок фактов, который агент правит сам (пункт 3.4).

Разница с долгосрочной памятью из memory.py не в хранилище, а в том, кто и
когда решает, что лежит в контексте:

    Working memory   — история диалога, пишет оркестратор автоматически
    Archival memory  — LongTermMemory, агент достаёт по вызову recall
    Core memory      — этот файл: всегда в контексте, правит сам агент

Зачем ещё один уровень, если память на диске уже есть: чтобы вспомнить имя
пользователя, агент обязан потратить итерацию на `recall` — а имя нужно
почти в каждом ответе. Core-блок избавляет от этой итерации.

Цена — место в контексте, которое занято ВСЕГДА. Отсюда все ограничения ниже:
блок фиксированного размера, поля заранее известны, значение проверяется
до записи, а каждая правка пишется в журнал.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter2.src.tools import tool

from .security import looks_like_instruction

# ====================================================================
# СХЕМА БЛОКА
# ====================================================================

# Поля заранее известны и их три. Это не бедность, а главное ограничение:
# свободный текст, который модель переписывает целиком, на 3B деградирует
# за несколько правок — сначала «уплотняется», потом теряет чужие факты.
# Фиксированный набор полей превращает «перепиши блок» в «замени поле».
CORE_FIELDS: dict[str, str] = {
    "user": "Кто пользователь: имя и как к нему обращаться",
    "project": "Над чем пользователь сейчас работает",
    "style": "Как пользователь просил отвечать",
}

# Потолок на одно поле. 120 символов — это примерно строка текста: имя,
# название проекта, короткое пожелание. Всё, что длиннее, — уже не факт,
# а пересказ, и ему место в долгосрочной памяти.
FIELD_LIMIT = 120

# Потолок на сумму полей. Нужен отдельно от FIELD_LIMIT: три поля по 120
# символов дали бы 360, и бюджет истории «плавал» бы вслед за памятью.
BLOCK_LIMIT = 300

EMPTY_MARK = "(пусто)"

BLOCK_HEADER = "ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ (core memory, правится инструментом update_core):"

DEFAULT_CORE_PATH = Path(__file__).parent.parent / "core_memory.json"
DEFAULT_LOG_PATH = Path(__file__).parent.parent / "core_memory.log"


class CoreMemory:
    """Блок фактов фиксированного размера, который всегда виден модели.

    Все проверки собраны в одном методе `update`, и это осознанно: единственный
    способ изменить блок — заменить одно поле, пройдя все проверки. Метода
    «записать блок целиком» здесь нет, потому что именно он и ломается.
    """

    def __init__(self, storage_path: Path | str | None = None,
                 log_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_CORE_PATH
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self._fields: dict[str, str] = {name: "" for name in CORE_FIELDS}
        self._load()

    # ---------------------------------------------------------------- диск

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠️ Ошибка загрузки core memory: {e}. Начинаю с пустого блока.")
            return

        # Читаем только известные поля: файл могли поправить руками, а лишний
        # ключ оттуда — это лишняя строка в КАЖДОМ запросе к модели.
        for name in CORE_FIELDS:
            value = saved.get(name, "")
            self._fields[name] = str(value)[:FIELD_LIMIT] if value else ""

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._fields, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠️ Ошибка сохранения core memory: {e}")

    def _log(self, entry: str) -> None:
        """Пишет строку в журнал правок.

        Журнал — не отладочная роскошь. Блок редактирует модель, и без записи
        «что было → что стало» деградация памяти незаметна: вы видите только
        последнее состояние и не видите, как имя пользователя по дороге
        превратилось в пересказ разговора.
        """
        line = f"{datetime.now().isoformat(timespec='seconds')}  {entry}"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"⚠️ Ошибка записи журнала core memory: {e}")

    # ---------------------------------------------------------------- чтение

    def get(self, field: str) -> str:
        return self._fields.get(field, "")

    def as_dict(self) -> dict[str, str]:
        return dict(self._fields)

    def used_chars(self) -> int:
        """Сколько символов блока занято сейчас."""
        return sum(len(v) for v in self._fields.values())

    def render(self) -> str:
        """Собирает блок для контекста.

        Форма постоянная: все поля перечислены всегда, пустые — с пометкой.
        Так модель видит, что поле существует и его можно заполнить, а размер
        блока не скачет от реплики к реплике.
        """
        lines = [BLOCK_HEADER]
        for name in CORE_FIELDS:
            value = self._fields[name] or EMPTY_MARK
            lines.append(f"- {name}: {value}")
        return "\n".join(lines)

    @staticmethod
    def worst_case_block() -> str:
        """Самый большой блок, который вообще может получиться.

        Нужен для бюджета контекста: место под core-память резервируется по
        верхней границе, иначе бюджет истории пришлось бы пересчитывать после
        каждой правки — а он должен быть предсказуем.
        """
        lines = [BLOCK_HEADER]
        for name in CORE_FIELDS:
            lines.append(f"- {name}: ")
        return "\n".join(lines) + "x" * BLOCK_LIMIT

    def log_tail(self, limit: int = 10) -> list[str]:
        """Последние строки журнала правок."""
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            return []
        return lines[-limit:]

    # ---------------------------------------------------------------- запись

    def update(self, field: str, value: str) -> str:
        """Заменяет ОДНО поле блока. Единственный способ его изменить.

        Порядок проверок повторяет логику всей главы: любая ошибка возвращается
        текстом, чтобы модель прочитала её в Observation и переделала вызов.
        """
        value = "" if value is None else str(value).strip()

        # 1. Поле должно существовать. Модель придумывает имена полей ровно
        # так же, как придумывала `name`/`fact` вместо `key`/`value`.
        if field not in CORE_FIELDS:
            self._log(f"ОТКАЗ  update {field!r}: неизвестное поле")
            return (
                f"❌ Неизвестное поле '{field}'. Доступны строго: "
                f"{', '.join(CORE_FIELDS)}. Для остальных фактов есть remember."
            )

        # 2. Пустое значение — это очистка поля, а не ошибка.
        if not value:
            old = self._fields[field]
            self._fields[field] = ""
            self._save()
            self._log(f"очищено {field}: {old!r} -> ''")
            return f"🧹 Поле '{field}' очищено."

        # 3. Значение не должно быть инструкцией. Блок виден модели в КАЖДОМ
        # запросе — инъекция, попавшая сюда, действует постоянно, в отличие
        # от разовой подставы в выводе инструмента.
        if looks_like_instruction(value):
            self._log(f"ОТКАЗ  update {field}: значение похоже на инструкцию: {value[:60]!r}")
            return (
                f"❌ Значение для '{field}' похоже на инструкцию, а не на факт. "
                f"В core-память пишут факты о пользователе, а не правила поведения."
            )

        # 4. Потолок на поле. Не обрезаем молча: обрезанный факт остаётся
        # в контексте навсегда и выглядит как настоящий.
        if len(value) > FIELD_LIMIT:
            self._log(f"ОТКАЗ  update {field}: {len(value)} символов > {FIELD_LIMIT}")
            return (
                f"❌ Слишком длинно для '{field}': {len(value)} символов при лимите "
                f"{FIELD_LIMIT}. Сократи до сути или сохрани подробности через remember."
            )

        # 5. Потолок на весь блок — считаем с учётом замены, а не сложением.
        would_use = self.used_chars() - len(self._fields[field]) + len(value)
        if would_use > BLOCK_LIMIT:
            self._log(f"ОТКАЗ  update {field}: блок {would_use} символов > {BLOCK_LIMIT}")
            return (
                f"❌ Блок памяти переполнен: {would_use} символов при лимите {BLOCK_LIMIT}. "
                f"Сначала сократи или очисти другое поле."
            )

        old = self._fields[field]
        self._fields[field] = value
        self._save()
        self._log(f"update {field}: {old!r} -> {value!r}")

        if old and old != value:
            # Затирание чужого факта проговариваем вслух: на 3B «обновление»
            # часто оказывается потерей предыдущего значения.
            return f"✅ Поле '{field}' обновлено: было «{old}», стало «{value}»."
        return f"✅ Запомнил в core-память: {field} = {value}"

    def clear(self) -> str:
        """Очищает весь блок (только по явной просьбе пользователя)."""
        self._fields = {name: "" for name in CORE_FIELDS}
        self._save()
        self._log("очищен весь блок")
        return "🧹 Core-память очищена."


# ====================================================================
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ====================================================================

_core_instance: CoreMemory | None = None


def get_core_memory() -> CoreMemory:
    """Возвращает глобальный экземпляр core-памяти (singleton)."""
    global _core_instance
    if _core_instance is None:
        _core_instance = CoreMemory()
    return _core_instance


# ====================================================================
# ИНСТРУМЕНТ ДЛЯ АГЕНТА
# ====================================================================

@tool
def update_core(field: str, value: str) -> str:
    """Записывает ОДНО поле короткой памяти, которая всегда видна тебе в контексте.

    Поля: user (имя пользователя), project (над чем работает), style (как просил
    отвечать). Пустое value очищает поле. Всё остальное сохраняй через remember.

    Args:
        field: Одно из: user, project, style
        value: Короткий факт, до 120 символов
    """
    return get_core_memory().update(field, value)
