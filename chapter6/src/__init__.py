"""
Компоненты Главы 6: лексический поиск, слияние выдач, реранкер, порог.

⚠️ Импорт этого пакета ПЕРЕОПРЕДЕЛЯЕТ инструмент `search_code` Главы 5
(теперь под ним гибридный поиск) и добавляет в общий реестр Главы 2 один
новый — `grep_code`. Если нужны только BM25 и слияние, без побочного
эффекта, импортируйте подмодуль напрямую:

    from chapter6.src.bm25 import BM25Index
"""
from .bm25 import K1, B, BM25Index
from .fusion import RRF_K, fuse, rrf, weighted_sum
from .hybrid import (
    BM25_CANDIDATES,
    DEFAULT_MODE,
    MODES,
    NO_ANSWER_BM25,
    TOP_K,
    VECTOR_CANDIDATES,
    HybridIndex,
    Signal,
    get_hybrid_index,
    set_hybrid_index,
)
from .lexical import (
    CODE_STOP_TOKENS,
    MIN_TOKEN_LEN,
    RU_STOP_TOKENS,
    STOP_TOKENS,
    keep,
    tokenize,
    tokenize_query,
)
from .reranker import (
    RERANK_BUDGET,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    apply_order,
    clear_rerank_cache,
    parse_order,
    render_candidates,
    rerank,
    rerank_stats,
)
from .tools import MAX_MATCHES, grep, grep_code, search_code

__all__ = [
    # токенизация
    "MIN_TOKEN_LEN",
    "STOP_TOKENS",
    "CODE_STOP_TOKENS",
    "RU_STOP_TOKENS",
    "keep",
    "tokenize",
    "tokenize_query",
    # BM25
    "BM25Index",
    "K1",
    "B",
    # слияние
    "RRF_K",
    "rrf",
    "weighted_sum",
    "fuse",
    # гибридный индекс и порог
    "HybridIndex",
    "Signal",
    "TOP_K",
    "MODES",
    "DEFAULT_MODE",
    "VECTOR_CANDIDATES",
    "BM25_CANDIDATES",
    "NO_ANSWER_BM25",
    "get_hybrid_index",
    "set_hybrid_index",
    # реранкер
    "rerank",
    "rerank_stats",
    "clear_rerank_cache",
    "render_candidates",
    "parse_order",
    "apply_order",
    "RERANK_ENABLED",
    "RERANK_CANDIDATES",
    "RERANK_BUDGET",
    # инструменты
    "search_code",
    "grep_code",
    "grep",
    "MAX_MATCHES",
]
