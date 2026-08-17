"""Работа с эмбеддингами через Ollama."""

import requests

from chapter1 import agent as base

EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    """Получает эмбеддинг для текста через Ollama."""
    response = requests.post(
        f"{base.OLLAMA_BASE}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Получает эмбеддинги для списка текстов (батчами)."""
    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
    return embeddings


def _get_embedding_with_prefix(text: str, prefix: str) -> list[float]:
    """Внутренняя функция: эмбеддинг с префиксом задачи."""
    response = requests.post(
        f"{base.OLLAMA_BASE}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": f"{prefix}: {text}"
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_document_embedding(text: str) -> list[float]:
    """Эмбеддинг для ДОКУМЕНТА при индексации.

    nomic-embed-text работает значительно лучше с префиксом 'search_document: '.
    """
    return _get_embedding_with_prefix(text, "search_document")


def get_query_embedding(text: str) -> list[float]:
    """Эмбеддинг для ПОИСКОВОГО ЗАПРОСА.

    nomic-embed-text работает значительно лучше с префиксом 'search_query: '.
    """
    return _get_embedding_with_prefix(text, "search_query")