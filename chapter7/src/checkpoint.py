"""
Чекпоинт: остановиться посреди графа и продолжить потом (пункт 7.7).

Состояние графа сериализуемо целиком — из этого сразу следуют три вещи,
и все три нужны дальше по курсу:

  * остановиться перед узлом и продолжить позже. Это основа
    Human-in-the-Loop: «покажи, что собираешься сделать, и жди меня»;
  * откатиться к шагу N после сбоя вместо перезапуска с нуля. На 3B
    один узел — это десятки секунд, и терять их из-за упавшего
    последнего шага обидно;
  * воспроизвести прогон по сохранённым состояниям при разборе.

Хранение — JSON-файл. Не потому, что этого достаточно навсегда (не
достаточно: одновременных прогонов файл не переживёт), а потому, что
интерфейс save/load/clear от хранилища не зависит. Замена файла
на SQLite и дальше на базу — тема Главы 14.

**Идемпотентность.** Снимок указывает на узел, который ЕЩЁ НЕ выполнен
(см. Graph.run), поэтому обычное продолжение ничего не переигрывает.
Но окно всё равно есть: узел успел записать факт в память, а сохраниться
программа не успела — машину выключили между двумя строчками. Тогда при
продолжении узел выполнится второй раз.

Отсюда требование к узлам с побочным действием: повтор должен быть
безобиден. Проверить это на глаз нельзя, поэтому в State есть `steps` —
узел смотрит, был ли он уже пройден, и решает сам. Оба случая покрыты
тестами (TestCheckpoint).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .graph import END, Graph, State

# Файл лежит рядом с главой и в .gitignore: внутри — реплики пользователя
# и найденные фрагменты, то есть ровно то, что Глава 5 училась не пускать
# в индекс. Путь переопределяется аргументом — тесты пишут в tmp_path.
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoint.json"

# Версия формата. Пишется в файл и проверяется при чтении: молча поднять
# чекпоинт чужого формата — это получить непонятную ошибку через три шага
# вместо понятной сразу.
FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    """Снимок прогона: состояние, узел и история сообщений."""

    state: State
    # Узел, который выполнится при продолжении. Дублирует state.node
    # намеренно: чекпоинт читают руками, и позиция должна быть видна
    # в файле сразу, а не выковыриваться из вложенного объекта.
    node: str = ""
    # История разговора со специалистом. Хранится отдельно от State,
    # потому что это не данные графа, а контекст модели: граф отработает
    # и без неё, а вот продолжить разговор — нет.
    messages: list[dict[str, str]] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "created": self.created,
            "node": self.node,
            "state": self.state.to_dict(),
            "messages": list(self.messages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        version = data.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Чекпоинт версии {version}, а код понимает {FORMAT_VERSION}. "
                "Удалите файл и начните прогон заново."
            )
        state = State.from_dict(data.get("state", {}))
        return cls(
            state=state,
            node=data.get("node", "") or state.node,
            messages=list(data.get("messages", [])),
            created=data.get("created", ""),
        )

    def describe(self) -> str:
        """Одна строка для человека: что сохранено и где остановились."""
        where = self.node or "(не начат)"
        return (
            f"чекпоинт от {self.created or 'неизвестно когда'}: "
            f"следующий узел «{where}», пройдено {len(self.state.steps)} "
            f"({self.state.trace()})"
        )


def save(
    state: State,
    node: str | None = None,
    messages: list[dict[str, str]] | None = None,
    path: Path | str | None = None,
) -> Path:
    """Сохраняет снимок прогона. Возвращает путь к файлу.

    Пишем через временный файл и os.replace: прерванная запись не должна
    оставить наполовину написанный JSON вместо рабочего чекпоинта.
    Замена файла на большинстве систем атомарна, дописывание — нет.
    """
    target = Path(path) if path else CHECKPOINT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(
        state=state,
        node=node if node is not None else state.node,
        messages=list(messages or []),
        created=datetime.now().isoformat(timespec="seconds"),
    )

    handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def load(path: Path | str | None = None) -> Checkpoint | None:
    """Поднимает снимок. None — файла нет.

    Отсутствие файла — не ошибка: «продолжать нечего» это обычное
    состояние первого запуска. А вот битый файл — ошибка, и она
    выбрасывается: молча начать с нуля значило бы потерять прогон
    и не сказать об этом.
    """
    target = Path(path) if path else CHECKPOINT_PATH
    if not target.exists():
        return None
    with open(target, encoding="utf-8") as f:
        return Checkpoint.from_dict(json.load(f))


def clear(path: Path | str | None = None) -> bool:
    """Удаляет снимок. True — файл был и удалён."""
    target = Path(path) if path else CHECKPOINT_PATH
    if not target.exists():
        return False
    target.unlink()
    return True


def resume(graph: Graph, checkpoint: Checkpoint, **kwargs: Any) -> State:
    """Продолжает прогон с сохранённого узла.

    Законченный прогон продолжать нечего — возвращаем состояние как есть.
    Это не проверка «на дурака»: чекпоинт последнего шага сохраняется
    в норме, и продолжить с END попросят рано или поздно.
    """
    node = checkpoint.node or checkpoint.state.node
    if not node or node == END:
        return checkpoint.state
    return graph.run(checkpoint.state, start=node, **kwargs)


def checkpointer(path: Path | str | None = None) -> Any:
    """Готовый `on_step` для Graph.run: сохранять после каждого узла.

    Собирается функцией, а не пишется в каждом вызове, потому что
    сохранение после каждого шага — самый частый режим, а `on_step`
    у графа один, и занимать его руками не хочется.
    """

    def on_step(name: str, state: State) -> None:  # noqa: ARG001 — имя узла уже в steps
        save(state, path=path)

    return on_step
