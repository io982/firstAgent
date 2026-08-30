"""
Перевод вопроса на английский перед поиском — второй мост со стороны запроса.

Глава 5 строила мост так: просила модель назвать вероятные ИМЕНА из кода
и дописывала их к вопросу (см. rewrite.py). Здесь другой ход к той же цели —
перевести весь вопрос на язык, на котором написан код.

Замер типов вопроса показал, что ход небезоснователен: один и тот же вопрос,
заданный по-английски, находится лучше, чем по-русски, — 0.64 против 0.56
на bge-m3 и 0.44 против 0.02 на nomic-embed-text. Разрыв между русским
вопросом и английским кодом, с которого начиналась Глава 5, никуда не делся;
хороший эмбеддер его сокращает, но не закрывает.

Оба приёма стоят одного обращения к модели, и оба вставляются в одно и то же
место конвейера. Значит выбрать надо один — что и меряет
test_query_preparation в tests.py.

Имена из кода при переводе не трогаются: `is_safe_query` остаётся
`is_safe_query`, а не превращается в «is safe request».
"""

import os
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter1.agent import request_model

# Выключатель для замера «с переводом и без»:
#   PowerShell:   $env:AGENT_TRANSLATE = "1"
#   Linux/macOS:  export AGENT_TRANSLATE=1
TRANSLATE_ENABLED = os.environ.get("AGENT_TRANSLATE", "0") != "0"

SYSTEM_PROMPT = """Ты переводишь вопрос о коде на английский язык.

Правила:
1. Ответ — только перевод вопроса, одной короткой фразой.
2. Имена из кода не переводи и не меняй: is_safe_query остаётся is_safe_query.
3. Не отвечай на вопрос и ничего не объясняй.

Примеры:
вопрос: где вычисляется арифметическое выражение
перевод: where is the arithmetic expression evaluated

вопрос: как история разговора обрезается по бюджету
перевод: how is the conversation history trimmed by budget

вопрос: что делает estimate_tokens
перевод: what does estimate_tokens do"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

_cache: dict[str, str] = {}
_stats = {"calls": 0, "hits": 0, "seconds": 0.0}


def translate_stats() -> dict[str, float]:
    """Сколько раз звали модель, сколько взяли из кэша и сколько это заняло."""
    return dict(_stats)


def clear_translate_cache() -> None:
    _cache.clear()
    _stats.update(calls=0, hits=0, seconds=0.0)


def translate_query(question: str) -> str:
    """Переводит вопрос на английский. Пустая строка — не вышло.

    Ошибка ничего не роняет: не ответила модель — ищем по исходному
    вопросу, как искали до этого.
    """
    question = (question or "").strip()
    if not question:
        return ""

    if question in _cache:
        _stats["hits"] += 1
        return _cache[question]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"вопрос: {question}\nперевод:"},
    ]

    started = time.time()
    try:
        content = request_model(messages, response_format=RESPONSE_SCHEMA)
    except Exception:
        content = ""
    finally:
        _stats["calls"] += 1
        _stats["seconds"] += time.time() - started

    text = content or ""
    if '"query"' in text:
        match = re.search(r'"query"\s*:\s*"([^"]*)"', text)
        text = match.group(1) if match else ""

    translated = " ".join(text.split()).strip()
    _cache[question] = translated
    return translated


def english_query(question: str, enabled: bool | None = None) -> str:
    """Английский вопрос, если получилось; иначе исходный.

    В отличие от expand_query Главы 5, перевод ЗАМЕЩАЕТ вопрос, а не
    дописывается к нему. Дописать перевод к оригиналу значило бы оставить
    в запросе ту самую русскую половину, из-за которой он и находится хуже.
    """
    if not (TRANSLATE_ENABLED if enabled is None else enabled):
        return question

    translated = translate_query(question)
    return translated or question
