"""
Тот же конвейер, собранный на LangGraph, — и чего это стоит.

Свой граф Главы 7 сделал свою работу: показал, что узел, ребро
и состояние — это три десятка строк, а не магия. Дальше начинается
вопрос, который встаёт в любом настоящем проекте: писать своё
или взять готовое.

Ответить на него можно только числами, поэтому здесь собран РОВНО ТОТ ЖЕ
конвейер — те же девять узлов и те же функции перехода из `pipeline.py`,
без единой правки, — но ходит по нему LangGraph. Отличие ровно одно:
кто крутит цикл. Сравнивать при этом есть что: цену установки, форму
записи и поведение цикла с возвратом.

Как это устроено технически. LangGraph хочет, чтобы состоянием был
словарь, а узлы возвращали в него частичные обновления. У нас
состояние — объект `State` Главы 7, и переписывать под чужие правила
девять узлов означало бы сравнивать не сборщики графа, а два разных
кода. Поэтому объект кладётся в словарь одним ключом, а узлы
оборачиваются в одну строку. Приём годится не только для учебного
сравнения: так же переносят на LangGraph уже написанный конвейер,
когда переписывать его целиком нет ни времени, ни причины.

Зависимость необязательная. Если LangGraph не установлен, модуль
импортируется и честно об этом сообщает — курс продолжает работать
на своём графе, а `pip install langgraph` остаётся выбором читателя.
"""
from __future__ import annotations

from typing import Any, TypedDict

from chapter7.src.graph import END, State
from chapter8.src import guard
from chapter8.src.pipeline import (
    CONFIRM,
    DEPS,
    DONE,
    EDIT_NODE,
    MAX_GRAPH_STEPS,
    PLAN,
    READ_NODE,
    ROLLBACK,
    STEP,
    VERIFY,
    edge_after_confirm,
    edge_after_edit,
    edge_after_read,
    edge_after_step,
    edge_after_verify,
    node_confirm,
    node_deps,
    node_done,
    node_edit,
    node_plan,
    node_read,
    node_rollback,
    node_step,
    node_verify,
)
from chapter8.src.planner import Plan

try:
    from langgraph.graph import END as LG_END
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
    IMPORT_PROBLEM = ""
except ImportError as exc:  # pragma: no cover — зависит от окружения читателя
    LG_END = END
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False
    IMPORT_PROBLEM = str(exc)


class Carrier(TypedDict):
    """Состояние глазами LangGraph: один ключ, в нём наш объект State.

    Один ключ — потому что сравниваем мы сборщики графа, а не способы
    хранить данные. Разложи мы State по полям словаря, и разница между
    двумя сборками перестала бы быть разницей в одном.
    """

    state: Any


def _wrap(name: str, node):
    """Узел «State -> State» в узел «словарь -> обновление словаря».

    Вся адаптация к чужому интерфейсу помещается сюда, и именно
    поэтому её видно: девять узлов остались нетронутыми, а цена
    переезда — одна функция.

    Вторая строка в ней — про то, что при переезде теряется. Трейс
    прогона (`state.steps`) свой граф Главы 7 вёл сам, потому что это
    было его дело. LangGraph ведёт свой собственный и в наш объект
    не пишет — значит, пишем мы. Первая версия этого модуля строку
    забыла, и `state.trace()` после прогона возвращал «(пусто)»:
    ошибка не в логике, а ровно в том, что чужая библиотека делает
    не всё, что делала своя.
    """
    def run(carrier: Carrier) -> Carrier:
        state = node(carrier["state"])
        state.steps.append(name)
        return {"state": state}

    run.__name__ = name
    return run


def _route(edge):
    """Функция перехода — так же, как узел, но возвращает имя, а не состояние."""
    def choose(carrier: Carrier) -> str:
        chosen = edge(carrier["state"])
        # END у обеих библиотек — одна и та же строка «__end__», но
        # полагаться на совпадение чужих констант нельзя: перевод
        # делается явно, и если LangGraph однажды её сменит, сломается
        # здесь, а не в середине прогона.
        return LG_END if chosen == END else chosen

    return choose


def build_langgraph_pipeline():
    """Собирает тот же конвейер средствами LangGraph.

    Порядок вызовов другой, смысл тот же: узлы, безусловные рёбра,
    условные рёбра со списком возможных исходов, точка входа.
    Читатель, который разобрал `chapter7/src/graph.py`, узнаёт здесь
    каждую строчку — в этом и была цель писать свой граф раньше чужого.
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            f"LangGraph не установлен ({IMPORT_PROBLEM}). "
            "Поставьте: pip install langgraph — или работайте на своём графе."
        )

    builder = StateGraph(Carrier)
    for name, node in (
        (PLAN, node_plan),
        (CONFIRM, node_confirm),
        (STEP, node_step),
        (DEPS, node_deps),
        (VERIFY, node_verify),
        (READ_NODE, node_read),
        (EDIT_NODE, node_edit),
        (ROLLBACK, node_rollback),
        (DONE, node_done),
    ):
        builder.add_node(name, _wrap(name, node))

    builder.set_entry_point(PLAN)
    builder.add_edge(PLAN, CONFIRM)
    builder.add_conditional_edges(CONFIRM, _route(edge_after_confirm), {STEP: STEP, LG_END: LG_END})
    builder.add_conditional_edges(STEP, _route(edge_after_step), {STEP: STEP, DEPS: DEPS})
    builder.add_edge(DEPS, VERIFY)
    builder.add_conditional_edges(
        VERIFY, _route(edge_after_verify), {DONE: DONE, READ_NODE: READ_NODE, ROLLBACK: ROLLBACK}
    )
    builder.add_conditional_edges(READ_NODE, _route(edge_after_read), {EDIT_NODE: EDIT_NODE, DONE: DONE})
    builder.add_conditional_edges(
        EDIT_NODE, _route(edge_after_edit), {VERIFY: VERIFY, READ_NODE: READ_NODE, ROLLBACK: ROLLBACK}
    )
    builder.add_edge(ROLLBACK, DONE)
    builder.add_edge(DONE, LG_END)
    return builder.compile()


def run_langgraph_pipeline(
    task: str, tests: str = "", plan: Plan | None = None, limit: int = MAX_GRAPH_STEPS
) -> State:
    """Прогоняет конвейер под управлением LangGraph. Возвращает то же State.

    То же самое State — и это главное свойство всей затеи: замер,
    отчёт и тесты не знают, кто крутил цикл, и сравнивают одно с одним.

    `limit` — потолок шагов, у LangGraph он называется recursion_limit.
    Своё имя у чужого понятия: цикл с возвратом библиотека считает
    рекурсией, но ограничивает ровно то же, что `max_steps` в Главе 7.
    Значение берётся общее (`MAX_GRAPH_STEPS`), а не своё: две сборки
    с разными потолками — это уже не «тот же конвейер», и сравнение
    между ними перестаёт что-либо означать.
    """
    state = State(user_input=task)
    if tests:
        state.extra["tests"] = tests
    if plan is not None:
        state.extra["plan"] = plan.to_dict()
    guard.forget_changes()

    graph = build_langgraph_pipeline()
    result = graph.invoke({"state": state}, {"recursion_limit": limit})
    return result["state"]
