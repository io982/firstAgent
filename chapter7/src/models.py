"""
Разные модели разным агентам — и сколько это стоит на слабом железе.

Типовая схема выглядит соблазнительно: маршрутизатор раздаёт задачи
маленькой модели, общей, рассуждающей и кодовой. На машине с большой
видеопамятью так и делают.

У нас 6 ГБ, и вопрос стоит иначе: помещаются ли две модели одновременно.
Если да — переключение бесплатно, это просто другое поле в запросе.
Если нет — Ollama выгружает одну и загружает другую, и цена переключения
это секунды на КАЖДОМ чередовании. Ответ на вопрос даёт не рассуждение,
а замер — `switch_cost()` и тест TestModelSwitch.

Поэтому по умолчанию здесь пусто: все специалисты работают на одной
модели. Разные модели — не «правильнее», а дороже, и включать их стоит,
только когда замер на вашей машине сказал, что они уживаются.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chapter1.agent as base
from chapter1.agent import OLLAMA_BASE, request_model

# Модель по умолчанию — та же, что у всего курса (qwen2.5:3b, если
# не переопределена переменной AGENT_MODEL).
DEFAULT_MODEL = base.MODEL

# Имя специалиста -> имя модели. Пусто намеренно, см. модульную строку.
MODEL_BY_AGENT: dict[str, str] = {}


def model_for(agent: str) -> str:
    """Какая модель отвечает за этого специалиста."""
    return MODEL_BY_AGENT.get(agent) or base.MODEL


def set_model_for(agent: str, model: str | None) -> None:
    """Назначает специалисту модель. None — вернуть на общую."""
    if model is None:
        MODEL_BY_AGENT.pop(agent, None)
    else:
        MODEL_BY_AGENT[agent] = model


@contextmanager
def using_model(name: str | None) -> Iterator[str]:
    """Временно подменяет модель на время вызова.

    Подменяется модуль-уровневая переменная Главы 1, потому что через неё
    ходят ВСЕ запросы курса — и запрос агента, и запрос реранкера,
    и запрос маршрутизатора. Возврат в try/finally: исключение внутри
    не должно оставить курс на чужой модели до конца сессии.
    """
    previous = base.MODEL
    if name:
        base.MODEL = name
    try:
        yield base.MODEL
    finally:
        base.MODEL = previous


def loaded_models() -> list[str]:
    """Что Ollama держит в памяти прямо сейчас.

    Это и есть ответ на вопрос «уживаются ли две модели»: список длиной
    два означает, что уживаются, длиной один — что каждое переключение
    стоит загрузки. Ollama не отвечает — пустой список, а не исключение:
    справка о состоянии не должна ронять агента.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def switch_cost(first: str, second: str, rounds: int = 3) -> dict[str, float]:
    """Цена чередования двух моделей: секунды на один и тот же запрос.

    Меряется самый короткий запрос, какой можно придумать, — чтобы
    в измеренное время попало переключение, а не генерация. Сначала обе
    модели прогреваются, потом идут `rounds` чередований.

    Возвращает медианы: «своя» (тот же вопрос той же модели подряд)
    и «чужая» (после работы другой модели). Их разница и есть цена
    переключения на этой машине.
    """
    probe = [{"role": "user", "content": "Ответь одним словом: да"}]

    def ask(model: str) -> float:
        started = time.time()
        with using_model(model):
            request_model(probe)
        return time.time() - started

    # Прогрев: первый запрос к модели включает загрузку с диска, и мерить
    # её вместе с переключением — значит смешать два разных числа.
    ask(first)
    ask(second)

    same: list[float] = []
    switched: list[float] = []
    for _ in range(max(1, rounds)):
        ask(first)
        same.append(ask(first))
        switched.append(ask(second))
        switched.append(ask(first))

    return {
        "same": round(statistics.median(same), 2),
        "switched": round(statistics.median(switched), 2),
        "loaded": len(loaded_models()),
    }
