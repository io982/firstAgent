"""
Эмбеддинги: текст → вектор (пункт 4.2 ROADMAP).

Здесь всего три идеи:

  1. Модель эмбеддингов превращает текст в список чисел так, что близкие
     по смыслу тексты дают близкие векторы. Это отдельная модель, не та,
     что генерирует ответ: nomic-embed-text весит 274 МБ против 1.9 ГБ
     у qwen2.5:3b и умеет ровно одно — считать вектор.
  2. Близость измеряется косинусом угла между векторами, а не совпадением
     слов. Поэтому «как меня зовут» находит факт, записанный под ключом
     «имя_пользователя», на котором точный поиск Главы 3 промахивался.
  3. Вектор одного и того же текста не меняется, значит его можно кэшировать.
     Это не микрооптимизация: без кэша каждая переиндексация заново гоняет
     через модель весь корпус.
"""

import hashlib
import math
import os
import sys

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter1.agent import OLLAMA_BASE

# ====================================================================
# НАСТРОЙКИ
# ====================================================================

# Отдельная модель под эмбеддинги. Менять через переменную окружения:
#   PowerShell:   $env:AGENT_EMBED_MODEL = "bge-m3"
#   Linux/macOS:  export AGENT_EMBED_MODEL=bge-m3
EMBED_MODEL = os.environ.get("AGENT_EMBED_MODEL", "nomic-embed-text")

# Эмбеддинг считается быстрее генерации, но первый запрос ещё и грузит
# модель в память — отсюда запас по таймауту.
EMBED_TIMEOUT = 60

# Сколько текстов уходит в один запрос. Батч экономит не столько сеть,
# сколько накладные расходы на запрос: 32 коротких чанка одним вызовом
# считаются заметно быстрее, чем 32 отдельными.
BATCH_SIZE = 32

# nomic-embed-text обучен АСИММЕТРИЧНО: вопрос и абзац-ответ — тексты
# разной формы, и модель умеет мерить не «похожи ли они», а «отвечает ли
# второй на первый». Роль текста сообщается префиксом.
#
# Замер на корпусе главы, 16 вопросов с известным файлом-ответом:
#
#     правильные префиксы   10/16   средняя близость лучшего 0.742
#     без префиксов          8/16                            0.740
#     перепутанные           9/16                            0.701
#     всё как запросы       11/16                            0.744
#
# То есть на таком корпусе разница в пределах шума, и это честнее написать,
# чем повторить «так велит карточка модели». Префиксы оставлены: карточка
# действительно велит, на больших корпусах эффект описан, а стоят они ноль.
# Разбор — в разделе «Почему вопрос и документ кодируются по-разному».
DOCUMENT_PREFIX = "search_document"
QUERY_PREFIX = "search_query"


class EmbeddingError(RuntimeError):
    """Не удалось получить вектор: Ollama недоступна или модель не скачана."""


# ====================================================================
# КЭШ
# ====================================================================

# Ключ — хэш от (модель, префикс, текст). Не сам текст: чанк на 800
# символов в роли ключа словаря — это лишняя копия всего корпуса в памяти.
_cache: dict[str, list[float]] = {}

# Потолок на кэш. Без него долгая сессия с переиндексациями съедает память:
# один вектор nomic-embed-text — это 768 чисел, примерно 6 КБ на запись.
CACHE_LIMIT = 4096

_stats = {"hits": 0, "misses": 0, "requests": 0}


def _cache_key(prefix: str, text: str) -> str:
    raw = f"{EMBED_MODEL}\x00{prefix}\x00{text}".encode()
    return hashlib.sha1(raw).hexdigest()


def cache_stats() -> dict[str, int]:
    """Сколько векторов взято из кэша, сколько посчитано, сколько было HTTP-запросов."""
    return dict(_stats, size=len(_cache))


def clear_cache() -> None:
    """Забывает посчитанные векторы (нужно тестам и при смене модели)."""
    _cache.clear()
    _stats.update(hits=0, misses=0, requests=0)


# ====================================================================
# ЗАПРОС К OLLAMA
# ====================================================================

def _request_embeddings(prompts: list[str]) -> list[list[float]]:
    """Считает векторы для готовых строк (префикс уже подставлен).

    Сначала пробуем `/api/embed` — он принимает список и возвращает список.
    На Ollama старее 0.1.39 этой ручки нет, и тогда работает `/api/embeddings`,
    который умеет ровно один текст за раз. Фоллбэк оставлен потому, что
    у читателя курса может стоять любая версия.
    """
    if not prompts:
        return []

    _stats["requests"] += 1

    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": prompts},
            timeout=EMBED_TIMEOUT,
        )
    except requests.RequestException as e:
        raise EmbeddingError(f"Ollama недоступна: {e}") from e

    if response.status_code == 404:
        return [_request_single_embedding(p) for p in prompts]

    if response.status_code != 200:
        raise EmbeddingError(
            f"Ollama вернула {response.status_code}: {response.text[:200]}. "
            f"Проверьте, что модель скачана: ollama pull {EMBED_MODEL}"
        )

    vectors = response.json().get("embeddings")
    if not vectors or len(vectors) != len(prompts):
        raise EmbeddingError(f"Ollama вернула {len(vectors or [])} векторов вместо {len(prompts)}")
    return vectors


def _request_single_embedding(prompt: str) -> list[float]:
    """Старая ручка Ollama: один текст — один вектор."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": prompt},
            timeout=EMBED_TIMEOUT,
        )
    except requests.RequestException as e:
        raise EmbeddingError(f"Ollama недоступна: {e}") from e

    if response.status_code != 200:
        raise EmbeddingError(
            f"Ollama вернула {response.status_code}: {response.text[:200]}. "
            f"Проверьте, что модель скачана: ollama pull {EMBED_MODEL}"
        )

    vector = response.json().get("embedding")
    if not vector:
        raise EmbeddingError("Ollama вернула пустой вектор")
    return vector


# ====================================================================
# ВЕКТОРНАЯ АРИФМЕТИКА
# ====================================================================

def normalize(vector: list[float]) -> list[float]:
    """Приводит вектор к единичной длине.

    После этого косинусная близость — это обычное скалярное произведение,
    без двух квадратных корней на каждое сравнение. При переборе тысяч
    документов разница уже заметна, а хранить нормализованные векторы
    ничего не стоит.
    """
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return list(vector)
    return [x / length for x in vector]


def dot(a: list[float], b: list[float]) -> float:
    """Скалярное произведение. Для нормализованных векторов это и есть косинус."""
    if len(a) != len(b):
        raise ValueError(f"Разная размерность векторов: {len(a)} и {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинус угла между векторами: 1.0 — то же самое, 0.0 — ничего общего.

    Почему именно косинус, а не евклидово расстояние: длина вектора зависит
    в том числе от длины текста, а нас интересует направление — то есть
    смысл, — а не то, абзац это или одно предложение.

    Формально косинус живёт в диапазоне [-1, 1], но у эмбеддингов текста
    отрицательные значения почти не встречаются: «противоположных по смыслу»
    направлений модель не строит. На практике диапазон — примерно [0, 1].
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Разная размерность векторов: {len(a)} и {len(b)}")

    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        return 0.0
    return dot(a, b) / (length_a * length_b)


# ====================================================================
# ПУБЛИЧНОЕ API
# ====================================================================

def _embed_many(texts: list[str], prefix: str) -> list[list[float]]:
    """Считает векторы с кэшем и батчами, сохраняя порядок входа."""
    result: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        cached = _cache.get(_cache_key(prefix, text))
        if cached is None:
            _stats["misses"] += 1
            missing.append((i, text))
        else:
            _stats["hits"] += 1
            result[i] = cached

    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        prompts = [f"{prefix}: {text}" for _, text in batch]
        vectors = _request_embeddings(prompts)
        for (i, text), vector in zip(batch, vectors):
            unit = normalize(vector)
            if len(_cache) < CACHE_LIMIT:
                _cache[_cache_key(prefix, text)] = unit
            result[i] = unit

    # Ни один элемент не остаётся None: _request_embeddings либо вернул
    # ровно столько векторов, сколько просили, либо бросил EmbeddingError.
    return [vector for vector in result if vector is not None]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Векторы для ДОКУМЕНТОВ (индексация)."""
    return _embed_many(list(texts), DOCUMENT_PREFIX)


def embed_document(text: str) -> list[float]:
    """Вектор одного документа."""
    return _embed_many([text], DOCUMENT_PREFIX)[0]


def embed_query(text: str) -> list[float]:
    """Вектор ПОИСКОВОГО ЗАПРОСА.

    Отдельная функция не ради симметрии: префикс здесь другой, и перепутать
    его — самая незаметная ошибка главы. Ничего не ломается, поиск продолжает
    работать и выдавать правдоподобные результаты — просто хуже.
    """
    return _embed_many([text], QUERY_PREFIX)[0]


def embedding_model_available() -> bool:
    """Скачана ли модель эмбеддингов. Нужно REPL и тестам, чтобы не падать зря."""
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        names = [m.get("name", "") for m in response.json().get("models", [])]
    except requests.RequestException:
        return False

    # В /api/tags модель зовётся "nomic-embed-text:latest", а в конфиге
    # обычно пишут без тега — сравниваем по началу имени.
    base = EMBED_MODEL.split(":")[0]
    return any(name.split(":")[0] == base for name in names)
