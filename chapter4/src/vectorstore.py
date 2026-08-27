"""
Векторная база: где живут чанки и их векторы (пункт 4.3 ROADMAP).

Рабочее хранилище — ChromaVectorStore: настоящая база, векторы бинарно,
запись по одной записи, HNSW-индекс поверх коллекции. Она стоит здесь
не ради 31 чанка учебного корпуса, а ради того масштаба, на котором
живут настоящие базы знаний: десятки тысяч чанков, перебор линейно
растёт, файл целиком в память не помещается.

MemoryVectorStore рядом — те самые двадцать строк перебора, по которым
видно, что «векторный поиск» есть скалярное произведение в цикле.
Включается через AGENT_VECTOR_STORE=memory; тесты берут его же, потому
что он не трогает диск.

Интерфейс у обоих один: агент не знает, какое хранилище под ним. Мера
близости наружу тоже одна — score примерно в [0, 1], больше значит ближе.
Chroma внутри считает РАССТОЯНИЕ (0 — совпадение), и перевод спрятан
здесь: иначе порог отсечения в одном хранилище значил бы «не ближе чем»,
а в другом — «не дальше чем».
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .embeddings import dot

# Куда ChromaDB кладёт свою базу. Папка уже в .gitignore: индекс —
# производная от документов, его не жалко удалить и пересобрать.
CHROMA_PERSIST_DIR = Path(__file__).parent.parent.parent / "chroma_db"

# Куда MemoryVectorStore кладёт свой JSON, если его попросят сохраниться.
INDEX_DIR = Path(__file__).parent.parent / "index"

# Префикс коллекций Главы 4 в общей базе Chroma. Он нужен: в chroma_db
# может лежать что угодно ещё, а имена коллекций там плоские.
COLLECTION_PREFIX = "chapter4_"

# Какое хранилище используется по умолчанию. Chroma — настоящая база:
# бинарные векторы, инкрементальная запись, HNSW-индекс при росте корпуса.
# Перебор в JSON остаётся под рукой, чтобы посмотреть, как это устроено
# внутри, и чтобы тесты не трогали диск:
#   PowerShell:   $env:AGENT_VECTOR_STORE = "memory"
#   Linux/macOS:  export AGENT_VECTOR_STORE=memory
DEFAULT_BACKEND = os.environ.get("AGENT_VECTOR_STORE", "chroma")


@dataclass
class Hit:
    """Найденный фрагмент вместе с мерой близости."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "?"))

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))

    def label(self) -> str:
        return f"{self.source} › {self.heading}" if self.heading else self.source


class VectorStore:
    """Интерфейс хранилища. Пять операций, больше агенту не нужно."""

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        raise NotImplementedError

    def search(self, embedding: list[float], top_k: int = 3) -> list[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def entries(self) -> dict[str, dict[str, Any]]:
        """id → метаданные для всего, что лежит в базе.

        Нужно, чтобы находить устаревшие чанки: файл поправили, его текст
        нарезался иначе, id изменились — а старые записи остались лежать
        и продолжают находиться. Это самая частая причина, по которой RAG
        отвечает по документу, которого уже нет.
        """
        raise NotImplementedError

    def delete(self, ids: list[str]) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def ids(self) -> set[str]:
        """Все id. Одинаково выводится из entries() для любого хранилища."""
        return set(self.entries())


# ====================================================================
# ПЕРЕБОР
# ====================================================================

class MemoryVectorStore(VectorStore):
    """Векторный поиск полным перебором, с сохранением индекса в JSON.

    Сложность здесь без сюрпризов: O(n·d) на запрос, где n — число чанков,
    d — размерность вектора (768 у nomic-embed-text). Для тысячи чанков
    это миллион умножений — доли секунды на чистом Python и в разы меньше
    времени, чем один запрос к модели эмбеддингов. Полный перебор перестаёт
    быть приемлемым не на тысячах, а на сотнях тысяч документов.

    Векторы хранятся уже нормализованными (см. embeddings.normalize),
    поэтому косинус здесь — просто скалярное произведение.
    """

    def __init__(self, persist_path: Path | str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._records: dict[str, dict[str, Any]] = {}
        if self.persist_path:
            self._load()

    # ------------------------------------------------------------ запись

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not (len(ids) == len(texts) == len(embeddings)):
            raise ValueError("ids, texts и embeddings должны быть одной длины")
        metadatas = metadatas or [{} for _ in ids]

        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            # Запись по id перезаписывается, а не дублируется: id детерминирован
            # (см. make_chunk_id), поэтому повторная индексация того же файла
            # обновляет корпус, а не удваивает его.
            self._records[doc_id] = {
                "text": text,
                "embedding": list(embedding),
                "metadata": dict(metadata),
            }

        self._save()
        return len(ids)

    def delete(self, ids: list[str]) -> int:
        removed = 0
        for doc_id in ids:
            if self._records.pop(doc_id, None) is not None:
                removed += 1
        if removed:
            self._save()
        return removed

    def clear(self) -> None:
        self._records.clear()
        self._save()

    # ------------------------------------------------------------ чтение

    def search(self, embedding: list[float], top_k: int = 3) -> list[Hit]:
        if not self._records or top_k <= 0:
            return []

        scored: list[Hit] = []
        for doc_id, record in self._records.items():
            vector = record["embedding"]
            if len(vector) != len(embedding):
                # Индекс, собранный другой моделью эмбеддингов. Молча
                # пропустить такую запись правильнее, чем упасть: у читателя
                # мог остаться индекс от предыдущей модели.
                continue
            scored.append(Hit(doc_id, record["text"], dot(vector, embedding), record["metadata"]))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._records)

    def entries(self) -> dict[str, dict[str, Any]]:
        return {doc_id: dict(record["metadata"]) for doc_id, record in self._records.items()}

    # ------------------------------------------------------------ диск

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                self._records = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠️ Не удалось прочитать индекс {self.persist_path}: {e}. Начинаю с пустого.")
            self._records = {}

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                # Векторы в JSON — это дорого: 768 чисел на запись, примерно
                # 15 КБ текста на чанк. Настоящие базы хранят их в двоичном
                # виде и по делу; нам важнее, чтобы индекс можно было открыть
                # и посмотреть глазами.
                json.dump(self._records, f, ensure_ascii=False)
        except OSError as e:
            print(f"⚠️ Не удалось сохранить индекс: {e}")


# ====================================================================
# CHROMA
# ====================================================================

class ChromaVectorStore(VectorStore):
    """То же самое поверх ChromaDB.

    Chroma умеет считать эмбеддинги сама, но мы передаём свои: модель
    эмбеддингов в курсе одна и живёт в Ollama, а встроенная в Chroma
    качала бы ещё одну, стороннюю.
    """

    def __init__(
        self,
        collection: str = f"{COLLECTION_PREFIX}docs",
        persist_dir: Path | str = CHROMA_PERSIST_DIR,
    ):
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "ChromaDB не установлена. Поставьте её (pip install chromadb) "
                "или используйте хранилище memory."
            ) from e

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Косинус, а не евклидово расстояние по умолчанию: см. оговорку
            # про длину вектора в cosine_similarity.
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not ids:
            return 0
        # Chroma отвергает пустой словарь метаданных, а MemoryVectorStore
        # его принимает. Разницу гасим здесь, чтобы интерфейс оставался одним:
        # запись без источника получает тот же прочерк, что Hit.source
        # показывает по умолчанию.
        prepared = [dict(m) if m else {"source": "?"} for m in (metadatas or [{}] * len(ids))]

        # upsert, а не add: повторная индексация того же файла должна
        # обновлять записи, а не падать на дубликатах id.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=prepared,
        )
        return len(ids)

    def search(self, embedding: list[float], top_k: int = 3) -> list[Hit]:
        if top_k <= 0 or self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
        )

        hits: list[Hit] = []
        for i, doc_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][i]
            # Chroma возвращает косинусное РАССТОЯНИЕ (0 — совпадение).
            # Переводим в ту же меру, что и у перебора, чтобы порог
            # отсечения означал одно и то же в обоих хранилищах.
            hits.append(
                Hit(
                    id=doc_id,
                    text=result["documents"][0][i],
                    score=1.0 - float(distance),
                    metadata=result["metadatas"][0][i] or {},
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def entries(self) -> dict[str, dict[str, Any]]:
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return {
            doc_id: dict(metadatas[i] or {}) if i < len(metadatas) else {}
            for i, doc_id in enumerate(result["ids"])
        }

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        existing = list(self.ids())
        if existing:
            self._collection.delete(ids=existing)


# ====================================================================
# ВЫБОР ХРАНИЛИЩА
# ====================================================================

def get_store(backend: str | None = None, name: str = "docs", **kwargs: Any) -> VectorStore:
    """Создаёт хранилище для корпуса `name`: "chroma" (по умолчанию) или "memory".

    Корпусов в главе два — документы и факты, — и жить они должны врозь:
    в одной коллекции короткая строка «сервер: prod-01» конкурировала бы
    с абзацем документации, а потолок выдачи у них разный.

    Имя корпуса превращается в имя коллекции Chroma или в имя JSON-файла —
    звать хранилище отсюда одинаково, независимо от того, что под ним.
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "memory":
        kwargs.setdefault("persist_path", INDEX_DIR / f"{name}.json")
        return MemoryVectorStore(**kwargs)
    if backend == "chroma":
        kwargs.setdefault("collection", f"{COLLECTION_PREFIX}{name}")
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"Неизвестное хранилище: {backend}. Доступны: memory, chroma")
