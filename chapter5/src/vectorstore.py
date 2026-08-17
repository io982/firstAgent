"""Векторная база знаний на ChromaDB."""

import os

import chromadb

from chapter1 import agent as base
from .embeddings import get_embedding, get_embeddings

# Локальная база в корне проекта.
# Путь абсолютный: с относительным "./chroma_db" запуск из другой
# директории создавал бы вторую, пустую базу.
CHROMA_PERSIST_DIR = os.path.join(base.PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "agent_memory"

_client = None


def _get_client():
    """Клиент создаётся при первом обращении, а не при импорте модуля."""
    global _client
    if _client is None:
        # Новый API ChromaDB (версия 0.4+)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_or_create_collection():
    """Получает или создаёт коллекцию для хранения документов."""
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # косинусное сходство
    )


def add_document(doc_id: str, text: str, metadata: dict = None):
    """Добавляет документ в векторную базу."""
    collection = get_or_create_collection()
    
    embedding = get_embedding(text)
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}]
    )


def search_documents(query: str, n_results: int = 5) -> list[dict]:
    """Ищет похожие документы по запросу."""
    collection = get_or_create_collection()
    
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return documents


def delete_document(doc_id: str):
    """Удаляет документ из базы."""
    collection = get_or_create_collection()
    collection.delete(ids=[doc_id])


def list_documents(limit: int = 10) -> list[dict]:
    collection = get_or_create_collection()
    results = collection.get(limit=limit)
    documents = []
    for i in range(len(results["ids"])):
        documents.append({
            "id": results["ids"][i],
            "text": results["documents"][i],  # ← БЫЛО: [:100] + "..."
            "metadata": results["metadatas"][i]
        })
    return documents