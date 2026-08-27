"""
Компоненты Главы 5: обход репозитория, нарезка кода, карточки, индекс, карта.

⚠️ Импорт этого пакета РЕГИСТРИРУЕТ инструменты search_code, find_symbol,
project_map и dependencies в общем реестре Главы 2 — так же, как импорт
chapter4.src регистрирует search_docs. Если нужна только нарезка, без
побочного эффекта, импортируйте подмодуль напрямую:

    from chapter5.src.codechunks import chunk_source
"""
from .cards import EMBED_MODE, EMBED_MODES, build_card, embedding_text, split_identifier
from .codebase import (
    CODE_COLLECTION,
    MIN_SCORE,
    SCORE_GAP,
    TOP_K,
    CodeIndex,
    describe,
    fit_lines,
    get_code_index,
    get_code_store,
    set_code_index,
)
from .codechunks import (
    MAX_CHUNK_CHARS,
    MAX_CHUNK_LINES,
    CodeChunk,
    chunk_braces,
    chunk_code,
    chunk_lines,
    chunk_markdown,
    chunk_python,
    chunk_source,
)
from .languages import (
    LANGUAGES,
    MAX_FILE_BYTES,
    SKIP_DIRS,
    gitignore_dirs,
    iter_sources,
    language_of,
    read_source,
)
from .repomap import (
    DEFAULT_ROOT,
    LANGUAGE_ALIASES,
    ProjectMap,
    Symbol,
    get_project_map,
    module_name,
    resolve_language,
    scan,
    set_project_map,
)
from .rewrite import (
    REWRITE_ENABLED,
    clear_rewrite_cache,
    expand_query,
    rewrite_query,
    rewrite_stats,
)
from .tools import (
    dependencies,
    find_symbol,
    get_code_budget,
    list_symbols,
    project_map,
    search_code,
    set_code_budget,
)

__all__ = [
    # обход репозитория
    "LANGUAGES",
    "SKIP_DIRS",
    "MAX_FILE_BYTES",
    "language_of",
    "iter_sources",
    "read_source",
    "gitignore_dirs",
    # нарезка
    "CodeChunk",
    "MAX_CHUNK_LINES",
    "MAX_CHUNK_CHARS",
    "chunk_python",
    "chunk_braces",
    "chunk_lines",
    "chunk_markdown",
    "chunk_code",
    "chunk_source",
    # карточки
    "EMBED_MODE",
    "EMBED_MODES",
    "split_identifier",
    "build_card",
    "embedding_text",
    # индекс кода
    "CodeIndex",
    "CODE_COLLECTION",
    "TOP_K",
    "MIN_SCORE",
    "SCORE_GAP",
    "describe",
    "fit_lines",
    "get_code_store",
    "get_code_index",
    "set_code_index",
    # карта проекта
    "Symbol",
    "ProjectMap",
    "DEFAULT_ROOT",
    "LANGUAGE_ALIASES",
    "resolve_language",
    "scan",
    "module_name",
    "get_project_map",
    "set_project_map",
    # переписывание запроса
    "REWRITE_ENABLED",
    "rewrite_query",
    "expand_query",
    "rewrite_stats",
    "clear_rewrite_cache",
    # инструменты
    "search_code",
    "find_symbol",
    "list_symbols",
    "project_map",
    "dependencies",
    "get_code_budget",
    "set_code_budget",
]
