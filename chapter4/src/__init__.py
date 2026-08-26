"""
Компоненты Главы 4: эмбеддинги, нарезка, векторное хранилище, база знаний.

⚠️ Импорт этого пакета РЕГИСТРИРУЕТ инструменты search_docs и recall_like
в общем реестре Главы 2 — так же, как импорт chapter3.src регистрирует
инструменты памяти. Если нужны только эмбеддинги или нарезка, без побочного
эффекта, импортируйте подмодуль напрямую:

    from chapter4.src.chunking import chunk_text
"""
from .chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Chunk,
    chunk_file,
    chunk_text,
    iter_documents,
    make_chunk_id,
    split_sections,
)
from .embeddings import (
    DOCUMENT_PREFIX,
    EMBED_MODEL,
    QUERY_PREFIX,
    EmbeddingError,
    cache_stats,
    clear_cache,
    cosine_similarity,
    embed_document,
    embed_documents,
    embed_query,
    embedding_model_available,
    normalize,
)
from .knowledge import (
    MIN_SCORE,
    TOP_K,
    IndexReport,
    KnowledgeBase,
    get_knowledge_base,
    set_knowledge_base,
)
from .selective import KEEP_RECENT, SelectiveConversation
from .semantic_memory import (
    FACTS_MIN_SCORE,
    SemanticMemory,
    get_semantic_memory,
    set_semantic_memory,
)
from .tools import get_retrieval_budget, recall_like, search_docs, set_retrieval_budget
from .vectorstore import (
    ChromaVectorStore,
    Hit,
    MemoryVectorStore,
    VectorStore,
    get_store,
)

__all__ = [
    # эмбеддинги
    "EMBED_MODEL",
    "DOCUMENT_PREFIX",
    "QUERY_PREFIX",
    "EmbeddingError",
    "embed_document",
    "embed_documents",
    "embed_query",
    "embedding_model_available",
    "cosine_similarity",
    "normalize",
    "cache_stats",
    "clear_cache",
    # нарезка
    "Chunk",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "chunk_text",
    "chunk_file",
    "iter_documents",
    "make_chunk_id",
    "split_sections",
    # хранилище
    "VectorStore",
    "MemoryVectorStore",
    "ChromaVectorStore",
    "Hit",
    "get_store",
    # база знаний
    "KnowledgeBase",
    "IndexReport",
    "TOP_K",
    "MIN_SCORE",
    "get_knowledge_base",
    "set_knowledge_base",
    # память по смыслу
    "SemanticMemory",
    "FACTS_MIN_SCORE",
    "get_semantic_memory",
    "set_semantic_memory",
    # история по релевантности
    "SelectiveConversation",
    "KEEP_RECENT",
    # инструменты
    "search_docs",
    "recall_like",
    "get_retrieval_budget",
    "set_retrieval_budget",
]
