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