"""
Компоненты Главы 7: специалисты, граф, маршрутизация, чекпоинт.

⚠️ Импорт этого пакета тянет за собой всю цепочку глав 1-6 — в частности,
поднимает модель эмбеддингов до bge-m3 и замещает search_code, как это
делает Глава 6. Глава 7 своего поиска не заводит: она про то, кто и когда
зовёт поиск Главы 6.

Новых инструментов пакет НЕ регистрирует ни одного — и это не упущение,
а содержание главы: специалист получается не добавлением инструментов,
а выборкой из уже существующих 16.

⚠️ И специалистов пакет тоже не регистрирует: здесь только реестр
и декоратор `@specialist`. Сами пятеро объявлены в `chapter7/agent.py`,
рядом со своими функциями поиска, — как инструменты Главы 3 объявлены
в Главе 3, а не в реестре Главы 2. Пока `chapter7.agent` не импортирован,
`SPECIALISTS` пуст и `Team()` соберётся пустой:

    from chapter7.src import Team
    Team().names()          # []  — специалисты ещё не объявлены

    import chapter7.agent   # noqa: F401
    Team().names()          # ['код', 'документы', 'память', ...]

Если нужен только граф, без всей цепочки, импортируйте подмодуль прямо:

    from chapter7.src.graph import Graph, State

— graph.py не зависит ни от одной главы курса и ни от чего, кроме
стандартной библиотеки.
"""
from .agents import (
    CODE_TOOLS,
    COMMON_RULES,
    DEFAULT_SPECIALIST,
    DOCS_TOOLS,
    FALLBACK_ORDER,
    MEMORY_TOOLS,
    SELF_RULES,
    SPECIALISTS,
    UTILITY_TOOLS,
    AgentSpec,
    Retrieval,
    Retriever,
    Team,
    drop_foreign_rules,
    get_specialist,
    prompt_sizes,
    register_specialist,
    specialist,
    tool_coverage,
    universal_tokens,
)
from .checkpoint import (
    CHECKPOINT_PATH,
    FORMAT_VERSION,
    Checkpoint,
    checkpointer,
    clear,
    load,
    resume,
    save,
)
from .graph import END, Edge, Graph, Node, State, run_parallel
from .models import (
    DEFAULT_MODEL,
    MODEL_BY_AGENT,
    loaded_models,
    model_for,
    set_model_for,
    switch_cost,
    using_model,
)
from .router import (
    AGENT_MARKERS,
    OPEN_MARKERS,
    ROUTER,
    ROUTER_PROMPT,
    Decision,
    clear_router_cache,
    looks_like_file_task,
    looks_like_memory_question,
    looks_like_self_question,
    route,
    route_by_model,
    route_by_words,
    router_schema,
    router_stats,
)

__all__ = [
    # специалисты
    "AgentSpec",
    "Team",
    "SPECIALISTS",
    "specialist",
    "Retrieval",
    "Retriever",
    "COMMON_RULES",
    "SELF_RULES",
    "DEFAULT_SPECIALIST",
    "FALLBACK_ORDER",
    "CODE_TOOLS",
    "DOCS_TOOLS",
    "MEMORY_TOOLS",
    "UTILITY_TOOLS",
    "drop_foreign_rules",
    "get_specialist",
    "register_specialist",
    "prompt_sizes",
    "universal_tokens",
    "tool_coverage",
    # граф
    "Graph",
    "State",
    "Node",
    "Edge",
    "END",
    "run_parallel",
    # маршрутизация
    "Decision",
    "route",
    "route_by_words",
    "route_by_model",
    "router_schema",
    "router_stats",
    "clear_router_cache",
    "looks_like_memory_question",
    "looks_like_self_question",
    "looks_like_file_task",
    "AGENT_MARKERS",
    "OPEN_MARKERS",
    "ROUTER",
    "ROUTER_PROMPT",
    # чекпоинт
    "Checkpoint",
    "CHECKPOINT_PATH",
    "FORMAT_VERSION",
    "save",
    "load",
    "clear",
    "resume",
    "checkpointer",
    # модели
    "DEFAULT_MODEL",
    "MODEL_BY_AGENT",
    "model_for",
    "set_model_for",
    "using_model",
    "loaded_models",
    "switch_cost",
]
