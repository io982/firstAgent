"""Индексация файлов проекта в векторную базу."""

import os

import chromadb

from chapter1 import agent as base

# Используем НОВЫЕ функции с префиксами для nomic-embed-text
from chapter5.src.embeddings import get_document_embedding, get_query_embedding

# Та же база, что у памяти Главы 5, но своя коллекция.
# Путь абсолютный — иначе запуск не из корня создаст вторую базу.
CHROMA_PERSIST_DIR = os.path.join(base.PROJECT_ROOT, "chroma_db")
PROJECT_COLLECTION = "project_files"

INDEX_EXTENSIONS = {
    ".py", ".md", ".txt", ".json",
    ".yml", ".yaml", ".toml", ".rst"
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules",
    ".venv", "venv", "chroma_db", ".idea", ".vscode"
}

CHUNK_SIZE = 1000

# Перекрытие между соседними чанками считается в строках, а не в символах:
# так граница всегда проходит по целой строке кода.
CHUNK_OVERLAP_LINES = 3

# Минимальная длина чанка — короткие чанки не несут смысла
MIN_CHUNK_LENGTH = 20

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_project_collection():
    return _get_client().get_or_create_collection(
        name=PROJECT_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def chunk_text_with_lines(text: str, chunk_size: int = CHUNK_SIZE,
                          overlap_lines: int = CHUNK_OVERLAP_LINES) -> list:
    """Разбивает текст на чанки по строкам.

    Возвращает список пар: (чистый_текст, текст_с_номерами_строк).
    Чистый текст — для эмбеддинга, текст с номерами — для возврата модели.
    """
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_size = 0
    start_line_num = 1

    for line in lines:
        line_size = len(line) + 1

        if current_size + line_size > chunk_size and current_lines:
            clean = "\n".join(current_lines)
            numbered = _format_chunk(current_lines, start_line_num)
            chunks.append((clean, numbered))

            if overlap_lines > 0 and len(current_lines) > overlap_lines:
                keep = current_lines[-overlap_lines:]
                start_line_num = start_line_num + len(current_lines) - overlap_lines
                current_lines = keep
                current_size = sum(len(kept_line) + 1 for kept_line in keep)
            else:
                start_line_num = start_line_num + len(current_lines)
                current_lines = []
                current_size = 0

        current_lines.append(line)
        current_size += line_size

    if current_lines:
        clean = "\n".join(current_lines)
        numbered = _format_chunk(current_lines, start_line_num)
        chunks.append((clean, numbered))

    return chunks


def _format_chunk(lines: list, start_line_num: int) -> str:
    """Добавляет номер к каждой строке чанка."""
    numbered = [f"{start_line_num + i}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def index_file(file_path: str, relative_path: str) -> int:
    """Индексирует один файл: читает, разбивает на чанки, сохраняет."""
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return 0

    # ИСПРАВЛЕНИЕ B: пропускаем пустые файлы (например, __init__.py)
    if not content.strip():
        return 0

    chunks = chunk_text_with_lines(content)
    if not chunks:
        return 0

    collection = get_project_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, (clean_chunk, numbered_chunk) in enumerate(chunks):
        # ИСПРАВЛЕНИЕ B: пропускаем слишком короткие чанки
        if len(clean_chunk.strip()) < MIN_CHUNK_LENGTH:
            continue

        chunk_id = f"{relative_path}::chunk{i}"
        ids.append(chunk_id)
        # ИСПРАВЛЕНИЕ A + C: эмбеддинг от ЧИСТОГО кода с префиксом search_document
        embeddings.append(get_document_embedding(clean_chunk))
        # Храним текст с номерами строк — модель видит реальные номера
        documents.append(numbered_chunk)
        metadatas.append({
            "path": relative_path,
            "chunk_index": i,
        })

    # Если все чанки оказались слишком короткими
    if not ids:
        return 0

    # upsert, а не add: повторная индексация того же файла перезаписывает
    # фрагменты, а не падает на дублирующихся id
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def clear_project_index():
    """Очищает индекс проекта (для переиндексации)."""
    collection = get_project_collection()
    all_data = collection.get()
    if all_data["ids"]:
        collection.delete(ids=all_data["ids"])


def index_project(root_path: str = ".") -> dict:
    """Индексирует весь проект: очищает старый индекс и создаёт новый.

    "." означает корень проекта, а не текущую директорию: агент индексирует
    свой проект, из какой бы папки его ни запустили. Любой другой путь
    используется как есть.
    """
    if root_path in {"", "."}:
        root_path = base.PROJECT_ROOT
    root_path = os.path.abspath(os.path.expanduser(root_path))

    total_files = 0
    total_chunks = 0
    errors = []

    # Очищаем старый индекс — старые эмбеддинги несовместимы с новыми
    clear_project_index()

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in INDEX_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, filename)
            # Прямые слэши на всех системах: в промпте агента пути показаны
            # как chapter1/agent.py, и модель не должна видеть два формата
            relative_path = os.path.relpath(full_path, root_path).replace(os.sep, "/")

            try:
                n_chunks = index_file(full_path, relative_path)
                if n_chunks > 0:
                    total_files += 1
                    total_chunks += n_chunks
            except Exception as e:
                errors.append(f"{relative_path}: {e}")

    return {"files": total_files, "chunks": total_chunks, "errors": errors}


def search_project(query: str, n_results: int = 5) -> list:
    """Ищет релевантные чанки в проекте."""
    collection = get_project_collection()
    if collection.count() == 0:
        return []

    # ИСПРАВЛЕНИЕ C: эмбеддинг запроса с префиксом search_query
    query_embedding = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "path": results["metadatas"][0][i].get("path", ""),
            "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return docs


def project_stats() -> dict:
    """Статистика индекса: количество чанков и уникальных файлов."""
    collection = get_project_collection()
    count = collection.count()

    unique_files = set()
    if count > 0:
        all_data = collection.get(include=["metadatas"])
        for meta in all_data["metadatas"]:
            unique_files.add(meta.get("path", ""))

    return {"chunks": count, "files": len(unique_files)}
