"""
Реранкер: второй взгляд на выдачу, читающий вопрос и код вместе (пункт 6.3).

И векторный поиск, и BM25 устроены одинаково в одном отношении: вопрос и
фрагмент они обрабатывают ПОРОЗНЬ. Вектор фрагмента посчитан заранее, при
индексации, и о вопросе ничего не знает; список вхождений слова тоже собран
заранее. Сравнение происходит уже между готовыми числами. Это и делает
поиск быстрым — тысяча фрагментов, и ни один из них не пришлось читать
в момент запроса.

Реранкер работает наоборот: он берёт пару «вопрос + фрагмент» целиком
и оценивает её как одно целое. Поэтому он видит то, чего не видит поиск:
что `estimate_tokens` в найденном фрагменте не определяется, а вызывается;
что фрагмент — это тест, а спрашивали про реализацию; что имя совпало,
но речь совсем о другом. И поэтому же он медленный: каждая пара — отдельная
работа, и на тысяче фрагментов это неприменимо. Отсюда порядок: сначала
поиск сужает тысячу до двадцати, потом реранкер разбирается с двадцатью.

**Чем это делают обычно и почему не здесь.** Стандартный инструмент —
cross-encoder: небольшая модель вроде `ms-marco-MiniLM`, обученная ровно
на эту задачу, она даёт оценку пары за миллисекунды. Ставить её пришлось бы
вместе с `torch` и `sentence-transformers` — около 2.5 ГБ на курс, который
весь про 6 ГБ видеопамяти, при том что видеопамять уже занята основной
моделью и моделью эмбеддингов. Поэтому здесь второй взгляд — та же LLM,
что отвечает пользователю: один запрос, в котором она видит вопрос и всех
кандидатов разом и расставляет их по порядку.

Это не замена cross-encoder, а его заменитель на том железе, которое есть.
Разницу видно в замере главы: один запрос к модели на вопрос о коде
занимает секунды, а не миллисекунды.

**Отказ не должен ничего ронять.** Не ответила модель, вернула мусор,
назвала номера, которых нет, — порядок остаётся тем, что дал поиск. Ровно
как с переписыванием запроса в Главе 5: приём либо улучшает, либо ничего
не меняет.
"""

import json
import os
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter1.agent import request_model
from chapter3.src.context import estimate_tokens
from chapter4.src.vectorstore import Hit
from chapter5.src.codebase import describe

# ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН — и это результат замера, а не осторожность.
#
# На двенадцати вопросах главы гибрид без реранкера дал 7 попаданий из 12
# при MRR 0.40, с реранкером — те же 7 при MRR 0.38. Единственное, что
# он поменял: вопрос про нарезку документа уехал со второго места
# на четвёртое. Двенадцать запросов к модели, 2.8 с на вопрос — и ноль
# в качестве, при том что разбор ответа не сбоил ни разу.
#
# Код остаётся: приём стандартный, на другой модели и другом корпусе
# он работает, и включается одной переменной.
#   PowerShell:   $env:AGENT_RERANK = "1"
#   Linux/macOS:  export AGENT_RERANK=1
RERANK_ENABLED = os.environ.get("AGENT_RERANK", "0") != "0"

# Сколько кандидатов показываем модели. Двадцать не влезают в разумный
# бюджет, а на пяти переставлять уже нечего: поиск и так вернул пять.
RERANK_CANDIDATES = 8

# Потолок на описание кандидатов. Всё, что сверху, обрезается по строкам —
# как выдача в Главе 5, и по той же причине: оборванная скобка сбивает
# модель сильнее, чем недостающие строки функции.
RERANK_BUDGET = 900

# Сколько строк фрагмента показываем. Начало определения — сигнатура
# и докстрока — решает почти всё; тело нужно реранкеру редко.
RERANK_LINES = 6

SYSTEM_PROMPT = """Ты упорядочиваешь найденные фрагменты кода по тому, насколько они отвечают на вопрос.

Правила:
1. Ответ — только JSON: {"order": [номера фрагментов от самого подходящего к наименее]}.
2. Перечисляй ВСЕ номера, которые видишь, и ни одного лишнего.
3. Реализация важнее вызова: фрагмент, где функция ОПРЕДЕЛЯЕТСЯ, идёт выше фрагмента, где она вызывается.
4. Тест — не реализация, если только не спрашивали про тесты.
5. Ничего не объясняй и не отвечай на сам вопрос."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"order": {"type": "array", "items": {"type": "integer"}}},
    "required": ["order"],
}

_cache: dict[str, list[int]] = {}
_stats = {"calls": 0, "hits": 0, "failures": 0, "seconds": 0.0}

_NUMBER = re.compile(r"-?\d+")


def rerank_stats() -> dict[str, float]:
    """Сколько раз звали модель, сколько взяли из кэша и сколько это заняло."""
    return dict(_stats)


def clear_rerank_cache() -> None:
    _cache.clear()
    _stats.update(calls=0, hits=0, failures=0, seconds=0.0)


def render_candidates(hits: list[Hit], budget_tokens: int = RERANK_BUDGET) -> str:
    """Собирает кандидатов в нумерованный список под бюджет.

    Показывается шапка с адресом и начало фрагмента. Полный код сюда класть
    незачем: восемь функций целиком — это половина окна, а решение
    принимается по сигнатуре и докстроке.
    """
    parts: list[str] = []
    used = 0

    for number, hit in enumerate(hits, 1):
        head = f"[{number}] {describe(hit)}"
        body = "\n".join(hit.text.splitlines()[:RERANK_LINES])
        block = f"{head}\n{body}"
        cost = estimate_tokens(block) + 1
        if used + cost > budget_tokens:
            break
        parts.append(block)
        used += cost

    return "\n\n".join(parts)


def parse_order(content: str, count: int) -> list[int]:
    """Достаёт из ответа модели порядок номеров. Пустой список — не разобрали.

    Разбирается и правильный JSON, и то, что 3B выдаёт вместо него: список
    без кавычек, номера через запятую, лишний текст вокруг. Схема
    constrained decoding это почти всегда предотвращает, но «почти»
    здесь недостаточно — от разбора зависит порядок выдачи.
    """
    order: list[int] = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            order = [int(value) for value in parsed.get("order", []) if isinstance(value, int)]
    except (ValueError, TypeError):
        order = [int(found) for found in _NUMBER.findall(content or "")]

    # Номера вне диапазона и повторы выбрасываются: модель на 3B умеет
    # назвать девятый фрагмент из восьми и назвать третий дважды.
    seen: set[int] = set()
    return [
        number
        for number in order
        if 1 <= number <= count and not (number in seen or seen.add(number))
    ]


def apply_order(hits: list[Hit], order: list[int]) -> list[Hit]:
    """Переставляет выдачу. Неназванные фрагменты уезжают в хвост в своём порядке.

    Именно в хвост, а не в мусор: молчание модели о фрагменте — это не
    «фрагмент плохой», а «до него не дошли». Выбрасывать по такому признаку
    значит терять ответ там, где реранкер просто поленился.
    """
    named = [hits[number - 1] for number in order]
    rest = [hit for index, hit in enumerate(hits, 1) if index not in set(order)]
    return named + rest


def rerank(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """Переставляет найденное по решению модели, читающей вопрос и код вместе."""

    def cut(ordered: list[Hit]) -> list[Hit]:
        return ordered[:top_k] if top_k else ordered

    if not RERANK_ENABLED or len(hits) < 2:
        return cut(hits)

    # Переставляются первые кандидаты, хвост едет следом как есть: показывать
    # модели двадцать фрагментов — это половина окна, а переставлять то,
    # что и так не попадёт в выдачу, незачем.
    candidates, tail = hits[:RERANK_CANDIDATES], hits[RERANK_CANDIDATES:]

    key = query + " :: " + "|".join(hit.id for hit in candidates)
    if key in _cache:
        _stats["hits"] += 1
        return cut(apply_order(candidates, _cache[key]) + tail)

    listing = render_candidates(candidates)
    if not listing:
        return cut(hits)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Вопрос: {query}\n\nФрагменты:\n\n{listing}"},
    ]

    started = time.time()
    content = ""
    try:
        content = request_model(messages, response_format=RESPONSE_SCHEMA)
    except Exception:
        content = ""
    finally:
        _stats["calls"] += 1
        _stats["seconds"] += time.time() - started

    order = parse_order(content, len(candidates))
    if not order:
        # Ни ответа модели, ни разбираемых номеров — оставляем порядок поиска.
        _stats["failures"] += 1
        return cut(hits)

    _cache[key] = order
    return cut(apply_order(candidates, order) + tail)
