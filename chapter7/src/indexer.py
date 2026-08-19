"""Индексация файлов проекта с гибридным поиском (векторы + BM25)."""
import os
import chromadb
from chapter5.src.embeddings import get_document_embedding, get_query_embedding
from .bm25 import SimpleBM25

CHROMA_PERSIST_DIR = "./chroma_db"
PROJECT_COLLECTION = "project_files"
INDEX_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".rst"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "chroma_db", ".idea", ".vscode"}
CHUNK_SIZE = 1000
MIN_CHUNK_LENGTH = 20

_client = None
_bm25_index = None  # Кэш BM25 индекса


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


def chunk_text_with_lines(text: str, chunk_size: int = CHUNK_SIZE, overlap_lines: int = 3) -> list:
    """Разбивает текст на чанки по строкам.
    
    Возвращает список пар: (чистый_текст, текст_с_номерами_строк).
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
                current_size = sum(len(l) + 1 for l in keep)
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
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return 0
    
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
        if len(clean_chunk.strip()) < MIN_CHUNK_LENGTH:
            continue
        chunk_id = f"{relative_path}::chunk{i}"
        ids.append(chunk_id)
        embeddings.append(get_document_embedding(clean_chunk))
        documents.append(numbered_chunk)
        metadatas.append({
            "path": relative_path,
            "chunk_index": i,
        })
    
    if not ids:
        return 0
    
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    
    # Сбрасываем кэш BM25, так как индекс изменился
    global _bm25_index
    _bm25_index = None
    
    return len(ids)


def clear_project_index():
    """Очищает индекс проекта."""
    collection = get_project_collection()
    all_data = collection.get()
    if all_data["ids"]:
        collection.delete(ids=all_data["ids"])
    global _bm25_index
    _bm25_index = None


def index_project(root_path: str = ".") -> dict:
    """Индексирует весь проект."""
    root_path = os.path.abspath(os.path.expanduser(root_path))
    total_files = 0
    total_chunks = 0
    errors = []
    
    clear_project_index()
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in INDEX_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, root_path)
            try:
                n_chunks = index_file(full_path, relative_path)
                if n_chunks > 0:
                    total_files += 1
                    total_chunks += n_chunks
            except Exception as e:
                errors.append(f"{relative_path}: {e}")
    
    return {"files": total_files, "chunks": total_chunks, "errors": errors}


def _get_bm25_index():
    """Возвращает BM25 индекс (создаёт при первом вызове)."""
    global _bm25_index
    if _bm25_index is None:
        collection = get_project_collection()
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return None
        
        # Формируем документы для BM25
        documents = []
        for i in range(len(all_data["ids"])):
            documents.append({
                "id": all_data["ids"][i],
                "text": all_data["documents"][i],
                "metadata": all_data["metadatas"][i]
            })
        _bm25_index = SimpleBM25(documents)
    return _bm25_index


# Буст по типу файла: код важнее документации для поиска реализации
FILE_TYPE_BOOST = {
    ".py": 1.3,    # Реальный код — повышаем
    ".md": 0.6,    # Документация — понижаем (засоряет поиск)
    ".txt": 0.8,
    ".json": 0.9,
    ".rst": 0.7,
}


def _get_file_boost(path: str) -> float:
    """Множитель веса в зависимости от типа файла."""
    ext = os.path.splitext(path)[1].lower()
    return FILE_TYPE_BOOST.get(ext, 1.0)


def _signature_boost(query: str, text: str) -> float:
    """Буст за точное совпадение сигнатуры (def X / class X).
    НЕ применяется к примерам кода, чтобы промпты не доминировали.
    """
    # Примеры кода не получают буст сигнатуры
    if _is_example_chunk(text):
        return 1.0
    import re
    query_lower = query.lower()
    text_lower = text.lower()
    sig_patterns = re.findall(r'(?:def|class)\s+\w+', query_lower)
    for pattern in sig_patterns:
        if pattern in text_lower:
            return 1.5
    return 1.0


def _is_example_chunk(text: str) -> bool:
    """Определяет, является ли чанк примером кода (а не реальной реализацией).
    Примеры кода в промптах и README засоряют поиск: BM25 не отличает
    пример от реализации. Мы детектим примеры по характерным маркерам
    и понижаем их вес.
    """
    text_lower = text.lower()
    # Маркеры примеров в системных промптах и документации
    example_markers = [
        "пример", "example",
        "user:", "assistant:",          # маркеры диалога в промптах
        "observation from",              # из ReAct-примеров
        "формат вызова инструмента",      # из наших промптов
        "используй только те номера строк",  # из PROJECT_RULES
        "если ask_project вернул",       # из PROJECT_RULES
    ]
    for marker in example_markers:
        if marker in text_lower:
            return True
    return False


def hybrid_search(query: str, n_results: int = 5, vector_weight: float = 0.5) -> list:
    """Гибридный поиск: векторы + BM25 + бусты для кода и сигнатур."""
    collection = get_project_collection()
    if collection.count() == 0:
        return []
    # 1. Векторный поиск
    query_embedding = get_query_embedding(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2
    )
    vector_docs = {}
    for i in range(len(vector_results["ids"][0])):
        doc_id = vector_results["ids"][0][i]
        vector_docs[doc_id] = {
            "path": vector_results["metadatas"][0][i].get("path", ""),
            "chunk_index": vector_results["metadatas"][0][i].get("chunk_index", 0),
            "text": vector_results["documents"][0][i],
            "vector_score": 1.0 - vector_results["distances"][0][i],
            "bm25_score": 0.0,
        }
    # 2. BM25 поиск
    bm25_index = _get_bm25_index()
    if bm25_index:
        bm25_results = bm25_index.search(query, n_results=n_results * 2)
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for result in bm25_results:
                doc_id = result["id"]
                normalized_score = result["score"] / max_bm25 if max_bm25 > 0 else 0
                if doc_id in vector_docs:
                    vector_docs[doc_id]["bm25_score"] = normalized_score
                else:
                    vector_docs[doc_id] = {
                        "path": result["metadata"].get("path", ""),
                        "chunk_index": result["metadata"].get("chunk_index", 0),
                        "text": result["text"],
                        "vector_score": 0.0,
                        "bm25_score": normalized_score,
                    }
    # 3. Объединяем scores С БУСТАМИ
    combined = []
    for doc_id, doc in vector_docs.items():
        base_score = (
            vector_weight * doc["vector_score"] +
            (1.0 - vector_weight) * doc["bm25_score"]
        )
        file_boost = _get_file_boost(doc["path"])
        sig_boost = _signature_boost(query, doc["text"])
        final_score = base_score * file_boost * sig_boost
        # НОВОЕ: понижаем вес примеров кода в промптах
        if _is_example_chunk(doc["text"]):
            final_score *= 0.5
        combined.append({
            "path": doc["path"],
            "chunk_index": doc["chunk_index"],
            "text": doc["text"],
            "vector_score": doc["vector_score"],
            "bm25_score": doc["bm25_score"],
            "score": final_score,
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n_results]


def project_stats() -> dict:
    """Статистика индекса."""
    collection = get_project_collection()
    count = collection.count()
    unique_files = set()
    if count > 0:
        all_data = collection.get(include=["metadatas"])
        for meta in all_data["metadatas"]:
            unique_files.add(meta.get("path", ""))
    return {"chunks": count, "files": len(unique_files)}
