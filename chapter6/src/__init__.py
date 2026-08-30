"""
Компоненты Главы 6: лексический поиск, слияние выдач, реранкер, порог.

⚠️ Импорт этого пакета ПЕРЕОПРЕДЕЛЯЕТ инструмент `search_code` Главы 5
(теперь под ним гибридный поиск) и добавляет в общий реестр Главы 2 один
новый — `grep_code`. Если нужны только BM25 и слияние, без побочного
эффекта, импортируйте подмодуль напрямую:

    from chapter6.src.bm25 import BM25Index

⚠️ И ещё импорт этого пакета ПОДНИМАЕТ МОДЕЛЬ ЭМБЕДДИНГОВ до bge-m3.
Главы 4 и 5 работают на nomic-embed-text, и все их числа сняты на ней;
Глава 6 замерила, что дело было в модели, и переключает её у себя — так же,
как Глава 5 переопределяет размер окна Главы 1 строкой `base.NUM_CTX = ...`.
Разбор — в тексте главы, раздел «Мы чинили не то».
"""
import os

from chapter4.src import embeddings as _embeddings

# Модель эмбеддингов Главы 6. Двенадцать русских вопросов о коде, попадание
# в первую пятёрку, векторный поиск без обработки запроса:
#
#     nomic-embed-text (274 МБ)    1 из 12
#     bge-m3 (1.2 ГБ)             10 из 12
#
# Явно заданная переменная окружения сильнее: если человек выбрал модель
# сам, глава не должна спорить.
EMBED_MODEL = "bge-m3"

if not os.environ.get("AGENT_EMBED_MODEL"):
    _embeddings.EMBED_MODEL = EMBED_MODEL

# Импорты ниже идут ПОСЛЕ переключения модели, и `noqa: E402` стоит именно
# поэтому. Сейчас ни один из подмодулей не читает EMBED_MODEL на импорте —
# все читают в момент вызова, — но порядок «сначала настроить, потом
# импортировать» верен по построению и не сломается, если кто-нибудь
# однажды закэширует имя модели на уровне модуля.

from .bm25 import K1, B, BM25Index  # noqa: E402
from .fusion import RRF_K, fuse, rrf, weighted_sum  # noqa: E402
from .hybrid import (  # noqa: E402
    BM25_CANDIDATES,
    DEFAULT_MODE,
    MODES,
    NO_ANSWER_BM25,
    NO_ANSWER_SUPPORT,
    TOP_K,
    VECTOR_CANDIDATES,
    HybridIndex,
    Signal,
    get_hybrid_index,
    set_hybrid_index,
)
from .lexical import (  # noqa: E402
    CODE_STOP_TOKENS,
    MIN_TOKEN_LEN,
    QUESTION_FRAME_TOKENS,
    RU_STOP_TOKENS,
    STOP_TOKENS,
    content_tokens,
    keep,
    tokenize,
    tokenize_query,
)
from .reranker import (  # noqa: E402
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
from .tools import MAX_MATCHES, grep, grep_code, search_code  # noqa: E402

__all__ = [
    # токенизация
    "MIN_TOKEN_LEN",
    "STOP_TOKENS",
    "CODE_STOP_TOKENS",
    "QUESTION_FRAME_TOKENS",
    "content_tokens",
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
    "NO_ANSWER_SUPPORT",
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
