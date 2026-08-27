"""
Переписывание запроса: мост со стороны вопроса (пункт 5.3).

Карточка достраивает мост со стороны документа — приписывает к коду
человеческий текст. Замер главы показал, что этого мало: русский вопрос
без единого имени из кода находит нужное определение в первой пятёрке
один раз из двенадцати, а тот же вопрос на языке кода — двенадцать из
двенадцати.

Причина видна на прямом измерении близости. Вопрос «как оценивается
количество токенов в тексте» против трёх текстов:

    та самая докстрока по-русски            0.676
    посторонняя русская фраза про ветку     0.643
    код `def estimate_tokens(...)`          0.471

nomic-embed-text на русском меряет «это вообще русская проза?», а не тему.
Английский код от русского вопроса далеко, и никакая нарезка этого не
чинит.

Зато чинит одно слово в запросе. На настоящем индексе курса:

    «калькулятор»                                    → мусор, 0.708
    «вычисление арифметического выражения calculator» → chapter2/src/tools.py:53
                                                        calculator, 0.727

Отсюда приём: перед поиском попросить ту же самую LLM превратить вопрос
в набор вероятных имён из кода. Это один лишний запрос к модели на вопрос
о коде — и он окупается, потому что без него поиск отвечает не на тот
вопрос.

Полученные слова ДОПИСЫВАЮТСЯ к вопросу, а не заменяют его: если модель
угадала имя — работает имя, если промахнулась — работает исходный вопрос,
и хуже, чем было, не становится.
"""

import os
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chapter1.agent import request_model

# Выключатель для замера «с переписыванием и без»:
#   PowerShell:   $env:AGENT_CODE_REWRITE = "0"
#   Linux/macOS:  export AGENT_CODE_REWRITE=0
REWRITE_ENABLED = os.environ.get("AGENT_CODE_REWRITE", "1") != "0"

# Сколько слов берём из ответа модели. Длинный «запрос» перестаёт быть
# запросом: двадцать случайных английских слов размывают вектор так же,
# как размывал бы лишний абзац.
MAX_REWRITE_WORDS = 8

SYSTEM_PROMPT = """Ты переводишь вопрос о коде в поисковый запрос по исходникам.

Правила:
1. Ответ — от двух до шести слов НА АНГЛИЙСКОМ: вероятные имена функций, классов, переменных и ключевые слова языка.
2. НЕ отвечай на вопрос и ничего не объясняй.
3. Пиши имена так, как их пишут в коде: calculator, estimate_tokens, chunk_text.

Примеры:
вопрос: где реализован калькулятор
запрос: calculator eval arithmetic expression

вопрос: как история обрезается по бюджету токенов
запрос: trim history tokens budget

вопрос: где хранятся факты о пользователе между запусками
запрос: memory store facts save json

вопрос: как проверяется, что Ollama запущена
запрос: ollama running check request"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

# Тот же вопрос переписывается в тот же запрос — держим ответы под рукой.
# Иначе повторный вопрос в разговоре стоит ещё одного запроса к модели.
_cache: dict[str, str] = {}
_stats = {"calls": 0, "hits": 0, "seconds": 0.0}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def rewrite_stats() -> dict[str, float]:
    """Сколько раз звали модель, сколько взяли из кэша и сколько это стоило."""
    return dict(_stats)


def clear_rewrite_cache() -> None:
    _cache.clear()
    _stats.update(calls=0, hits=0, seconds=0.0)


def rewrite_query(question: str) -> str:
    """Просит модель назвать вероятные имена из кода. Пустая строка — не вышло.

    Ошибка здесь не должна ничего ронять: не ответила модель — ищем по
    исходному вопросу, как искали до этой главы.
    """
    question = (question or "").strip()
    if not question:
        return ""

    if question in _cache:
        _stats["hits"] += 1
        return _cache[question]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"вопрос: {question}\nзапрос:"},
    ]

    started = time.time()
    try:
        content = request_model(messages, response_format=RESPONSE_SCHEMA)
    except Exception:
        return ""
    finally:
        _stats["calls"] += 1
        _stats["seconds"] += time.time() - started

    # Ответ приходит по схеме, но модель на 3B умеет вернуть и просто
    # строку — разбираем оба случая, как parse_agent_response Главы 2.
    text = content or ""
    if '"query"' in text:
        match = re.search(r'"query"\s*:\s*"([^"]*)"', text)
        text = match.group(1) if match else ""

    words = _WORD.findall(text)[:MAX_REWRITE_WORDS]
    rewritten = " ".join(dict.fromkeys(words))  # порядок сохраняем, повторы убираем

    _cache[question] = rewritten
    return rewritten


def expand_query(question: str, enabled: bool | None = None) -> str:
    """Вопрос плюс его «кодовый» перевод — то, с чем идём в индекс.

    Дописываем, а не заменяем. Замена опаснее: модель, промахнувшаяся мимо
    имени, увела бы поиск в сторону совсем — а так исходный вопрос остаётся
    в запросе и продолжает работать.
    """
    if not (REWRITE_ENABLED if enabled is None else enabled):
        return question

    rewritten = rewrite_query(question)
    return f"{question} {rewritten}".strip() if rewritten else question
