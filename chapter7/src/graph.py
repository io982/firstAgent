"""
State Graph: поток управления агента, записанный кодом (пункт 7.6).

Три понятия и ничего больше:

  * **State** — общий объект, который течёт через весь прогон. Узлы его
    заполняют, узлы его читают. Сериализуем целиком — из этого сразу
    следует чекпоинт (см. checkpoint.py);
  * **узел** — функция `State -> State`. Один шаг работы: выбрать
    специалиста, найти фрагменты, спросить модель;
  * **ребро** — куда идти после узла. Безусловное (`edge`) или условное
    (`conditional`): функция `State -> имя следующего узла`.

Главное правило, ради которого всё это пишется:

    Модель заполняет поля State. Следующий узел выбирает КОД.

В свободном цикле ReAct Главы 1 модель решала и что делать, и что дальше.
На 3B второе решение — самое дорогое место: ошибка в выборе шага стоит
всей ветки. Здесь модель по-прежнему решает, что ответить и какой
инструмент позвать, но маршрут выбирает функция перехода, которую можно
прочитать глазами и покрыть тестом.

Зависимостей нет: ни LangGraph, ни чего-либо ещё. LangGraph смотрели как
референс архитектуры — узлы, рёбра, общий state, — но сотня строк здесь
делает то же самое и целиком помещается в голове.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# Конец прогона. Отдельным именем, а не пустой строкой: «дальше некуда» —
# это решение, и в трейсе оно должно быть видно как решение.
END = "__end__"

Node = Callable[["State"], "State"]
Edge = Callable[["State"], str]


@dataclass
class State:
    """Всё, что прогон знает о себе. Сериализуемо целиком.

    Полей немного, и каждое здесь потому, что его читает больше одного узла.
    Данные, нужные ровно одному узлу, кладутся в `extra` и не расширяют
    общий словарь: State — это то, о чём договорились все узлы, а не свалка.
    """

    user_input: str = ""
    # Кто отвечает. Заполняет узел маршрутизации, читают все следующие.
    agent: str = ""
    # Что положено в контекст к текущей реплике.
    retrieved: str = ""
    answer: str = ""
    # Специалисты, которых уже спрашивали. Это и есть защита от кольца
    # «код -> пусто -> документы -> пусто -> код»: условное ребро смотрит
    # сюда и второй раз того же не предлагает.
    tried: list[str] = field(default_factory=list)
    # Пройденные узлы по порядку — трейс прогона. Печатается человеку,
    # проверяется тестами и пригодится при разборе (Глава 13).
    steps: list[str] = field(default_factory=list)
    # Узел, который будет выполнен СЛЕДУЮЩИМ. Пустая строка — прогон ещё
    # не начинался, END — закончился. Позиция в графе тоже часть состояния:
    # без неё чекпоинт не знает, откуда продолжать.
    node: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Состояние словарём — для json.dump в чекпоинт.

        Из `extra` выбрасывается всё, что не сериализуется. Это не защита
        «на всякий случай»: в extra живут диалоги специалистов — объекты
        SelectiveConversation, — и первая же попытка сохранить прогон
        падала на них с TypeError. Нашёл это интеграционный тест, а не
        рассуждение, хотя в коде уже было написано, что диалоги в чекпоинт
        не едут: написать в комментарии — не то же самое, что обеспечить.

        Выброшенное не пропадает молча: их имена остаются в файле под
        ключом `__dropped__`, и по ним видно, чего в снимке нет.
        """
        kept: dict[str, Any] = {}
        dropped: list[str] = []
        for key, value in self.extra.items():
            if key == "__dropped__":
                continue
            try:
                json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                dropped.append(key)
                continue
            kept[key] = value
        if dropped:
            kept["__dropped__"] = dropped

        return {
            "user_input": self.user_input,
            "agent": self.agent,
            "retrieved": self.retrieved,
            "answer": self.answer,
            "tried": list(self.tried),
            "steps": list(self.steps),
            "node": self.node,
            "error": self.error,
            "extra": kept,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> State:
        """Состояние из словаря. Незнакомые ключи молча отбрасываются.

        Отбрасываются намеренно: чекпоинт, снятый более старой версией
        кода, должен подниматься, а не падать с TypeError. Недостающие
        поля берут значения по умолчанию — по той же причине.
        """
        known = {
            key: value
            for key, value in (data or {}).items()
            if key in cls.__dataclass_fields__
        }
        return cls(**known)

    def copy(self) -> State:
        """Независимая копия — списки и содержимое extra тоже.

        Через deepcopy, а не через to_dict/from_dict: сериализация
        выбрасывает из extra несериализуемое, и копия молча теряла бы
        диалоги специалистов. Копия и снимок — разные вещи, и делать
        одно через другое нельзя.
        """
        return copy.deepcopy(self)

    def trace(self) -> str:
        """Пройденный маршрут одной строкой."""
        return " -> ".join(self.steps) if self.steps else "(пусто)"


class Graph:
    """Узлы, рёбра и цикл, который по ним ходит."""

    def __init__(self, max_steps: int = 12):
        self.nodes: dict[str, Node] = {}
        # Безусловные рёбра: узел -> узел.
        self.edges: dict[str, str] = {}
        # Условные: узел -> функция перехода.
        self.conditions: dict[str, Edge] = {}
        # Куда условное ребро МОЖЕТ привести. Нужно только для validate():
        # статически заглянуть внутрь функции перехода нельзя, а проверять
        # рёбра до запуска хочется.
        self.targets: dict[str, tuple[str, ...]] = {}
        self.entry_point: str = ""
        # Потолок шагов. Цикл в графе — это нормально (в нём вся польза
        # условных рёбер), а вот бесконечный цикл — нет.
        self.max_steps = max_steps

    # ---------------------------------------------------------------
    # СБОРКА
    # ---------------------------------------------------------------

    def node(self, name: str, fn: Node) -> Graph:
        """Добавляет узел. Возвращает себя — чтобы сборка читалась цепочкой."""
        if name == END:
            raise ValueError(f"'{END}' — служебное имя конца прогона, узлом быть не может")
        if name in self.nodes:
            raise ValueError(f"Узел '{name}' уже есть в графе")
        self.nodes[name] = fn
        return self

    def edge(self, src: str, dst: str) -> Graph:
        """Безусловный переход src -> dst."""
        if src in self.conditions:
            raise ValueError(
                f"У узла '{src}' уже есть условное ребро: два выхода из одного узла — "
                "это не граф, а неопределённость"
            )
        self.edges[src] = dst
        return self

    def conditional(
        self, src: str, fn: Edge, targets: Sequence[str] | None = None
    ) -> Graph:
        """Условный переход: куда идти, решает функция по состоянию.

        `targets` — перечисление возможных исходов. Необязательно, но полезно:
        только по нему validate() может заметить переход в несуществующий узел
        ДО запуска. Сама функция перехода при этом ничем не ограничена.
        """
        if src in self.edges:
            raise ValueError(f"У узла '{src}' уже есть безусловное ребро")
        self.conditions[src] = fn
        if targets is not None:
            self.targets[src] = tuple(targets)
        return self

    def entry(self, name: str) -> Graph:
        """Точка входа — узел, с которого начинается прогон."""
        self.entry_point = name
        return self

    # ---------------------------------------------------------------
    # ПРОВЕРКА
    # ---------------------------------------------------------------

    def validate(self) -> list[str]:
        """Список претензий к графу. Пустой список — граф собран верно.

        Проверка отдельным вызовом, а не при сборке: рёбра часто объявляют
        раньше узлов, и ругаться на ещё не добавленный узел было бы вредно.
        """
        problems: list[str] = []

        if not self.entry_point:
            problems.append("не задана точка входа")
        elif self.entry_point not in self.nodes:
            problems.append(f"точка входа '{self.entry_point}' — не узел графа")

        for src, dst in self.edges.items():
            if src not in self.nodes:
                problems.append(f"ребро из несуществующего узла '{src}'")
            if dst != END and dst not in self.nodes:
                problems.append(f"ребро '{src}' -> '{dst}': такого узла нет")

        for src, targets in self.targets.items():
            if src not in self.nodes:
                problems.append(f"условное ребро из несуществующего узла '{src}'")
            for dst in targets:
                if dst != END and dst not in self.nodes:
                    problems.append(f"условное ребро '{src}' -> '{dst}': такого узла нет")

        for src in self.conditions:
            if src not in self.nodes:
                problems.append(f"условное ребро из несуществующего узла '{src}'")

        unreachable = set(self.nodes) - self._reachable()
        for name in sorted(unreachable):
            problems.append(f"узел '{name}' недостижим из точки входа")

        return problems

    def _reachable(self) -> set[str]:
        """Узлы, до которых можно дойти от точки входа.

        Для условных рёбер берутся объявленные `targets`. Не объявили —
        считаем, что оттуда достижимо всё: лучше промолчать, чем назвать
        живой узел недостижимым.
        """
        if not self.entry_point:
            return set()
        if any(src not in self.targets for src in self.conditions):
            return set(self.nodes)

        seen: set[str] = set()
        queue = [self.entry_point]
        while queue:
            current = queue.pop()
            if current in seen or current == END or current not in self.nodes:
                continue
            seen.add(current)
            queue.extend(self._outgoing(current))
        return seen

    def _outgoing(self, name: str) -> Iterable[str]:
        if name in self.edges:
            return [self.edges[name]]
        return self.targets.get(name, ())

    # ---------------------------------------------------------------
    # ПРОГОН
    # ---------------------------------------------------------------

    def next_node(self, current: str, state: State) -> str:
        """Куда идти после узла `current`.

        Узел без исходящих рёбер — конец прогона. Это не забывчивость,
        а соглашение: точка выхода объявляется отсутствием выхода.
        """
        if current in self.conditions:
            chosen = self.conditions[current](state)
            if chosen != END and chosen not in self.nodes:
                raise ValueError(
                    f"Функция перехода из '{current}' вернула '{chosen}' — такого узла нет. "
                    f"Есть: {sorted(self.nodes)}"
                )
            return chosen
        return self.edges.get(current, END)

    def run(
        self,
        state: State,
        start: str | None = None,
        on_step: Callable[[str, State], None] | None = None,
        stop_before: str | None = None,
    ) -> State:
        """Ходит по графу, пока не упрётся в END или в потолок шагов.

        Args:
            state: Состояние. Меняется на месте — узлы возвращают его же.
            start: С какого узла начать. По умолчанию — точка входа.
                Именно этим продолжается прогон из чекпоинта.
            on_step: Зовётся после каждого узла — печать, замер, сохранение.
            stop_before: Остановиться ПЕРЕД этим узлом, не выполняя его.
                Прогон при этом не сломан: `state.node` указывает на него,
                и `run(state, start=state.node)` доведёт дело до конца.

        Потолок шагов не исключение, а поле `error`: наполовину пройденный
        граф всё равно несёт полезное состояние, и выбрасывать его вместе
        с исключением было бы расточительно.
        """
        current = start or self.entry_point
        if not current:
            raise ValueError("Графу не задана точка входа")

        for _ in range(self.max_steps):
            state.node = current
            if current == END:
                return state
            if stop_before is not None and current == stop_before:
                return state
            if current not in self.nodes:
                raise ValueError(f"Узла '{current}' нет в графе")

            state = self.nodes[current](state)
            state.steps.append(current)

            # Следующий узел вычисляется ДО on_step, а не после. Разница
            # видна только в чекпоинте — и она принципиальная: сохранённое
            # состояние должно указывать на узел, который ЕЩЁ НЕ выполнен.
            # Иначе продолжение переигрывает последний шаг, а он мог что-то
            # записать в память.
            following = self.next_node(current, state)
            state.node = following

            if on_step is not None:
                on_step(current, state)

            current = following

        state.error = (
            f"Предел шагов графа ({self.max_steps}) исчерпан. Маршрут: {state.trace()}"
        )
        return state


def run_parallel(
    nodes: Sequence[Node], state: State, workers: int = 2
) -> list[State]:
    """Прогоняет несколько узлов на одном состоянии одновременно.

    Каждый узел получает СВОЮ копию состояния: иначе два узла пишут
    в одно поле и выигрывает тот, кто закончил позже. Результаты
    возвращаются в порядке узлов, а не в порядке завершения.

    Упавший узел не роняет остальные — его беда записывается в `error`
    его же копии состояния. Слить копии обратно в одно состояние должен
    вызывающий: как именно сливать, знает задача, а не эта функция.

    Заранее стоит сказать честно: большого ускорения на одной видеокарте
    ждать не нужно. Ollama обслуживает запросы к одной модели почти
    последовательно, так что «параллельно» здесь означает в основном
    «параллельно ждём». Сколько именно остаётся — меряет TestParallel
    в tests.py, и число там маленькое; первая версия того замера показала
    выигрыш вчетверо больше, но мерила загрузку модели, а не очередь.
    """
    copies = [state.copy() for _ in nodes]

    def call(pair: tuple[Node, State]) -> State:
        fn, own = pair
        try:
            return fn(own)
        except Exception as e:  # noqa: BLE001 — беда одного узла не общая беда
            own.error = f"{type(e).__name__}: {e}"
            return own

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(call, zip(nodes, copies)))
