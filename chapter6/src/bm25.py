"""
BM25: поиск, у которого есть ноль (пункт 6.1).

Векторный поиск отвечает всегда. Он ранжирует, а не находит: даже на вопрос
про борщ он вернёт пять самых похожих фрагментов кода с близостью 0.7, и
отличить их от настоящего ответа по числу нельзя — этим кончились и Глава 4,
и Глава 5. BM25 устроен так, что на вопрос, слов которого в корпусе нет,
он возвращает пустой список. Не «плохие результаты», а ничего.

Формула складывается из трёх частей, и каждая чинит свою беду.

**TF — частота слова в документе.** Чем чаще слово встречается, тем документ
релевантнее. Но не линейно: десятое упоминание `calculator` не делает
фрагмент в десять раз лучше, поэтому рост насыщается — за это отвечает
`k1`.

**IDF — редкость слова в корпусе.** Главная часть. `def` есть почти в каждом
фрагменте и не значит ничего; `is_safe_query` — в двух, и его совпадение
весит много. Именно IDF делает лексический поиск полезным на коде, где
половина текста состоит из ключевых слов языка.

**Нормализация по длине.** В длинном фрагменте слово встречается «само
собой» чаще, чем в коротком. Без поправки поиск всегда вытаскивал бы
файлы-простыни — за неё отвечает `b`.

Внутри — обратный индекс: слово → в каких фрагментах оно есть. Разница
не косметическая. Перебор всех фрагментов на каждый запрос — это O(N·|Q|);
обратный индекс трогает только те фрагменты, где слово запроса вообще
встречается, — на редком имени это два фрагмента из тысячи.
"""

import math
from typing import Any

from chapter4.src.vectorstore import Hit

from .lexical import tokenize, tokenize_query

# Насыщение частоты. При k1 = 1.5 второе вхождение слова добавляет заметно
# меньше первого, десятое — почти ничего. Значение из литературы; менять
# его без замера незачем, как и k в RRF.
K1 = 1.5

# Сила поправки на длину: 0 — длину не учитываем вовсе, 1 — учитываем
# полностью. 0.75 — общепринятая середина.
B = 0.75


class BM25Index:
    """Лексический индекс: слова фрагментов и поиск по ним.

    Векторов здесь нет ни одного, поэтому индекс собирается без модели
    эмбеддингов и без сети — на корпусе этого проекта за доли секунды.
    Отсюда и решение не хранить его на диске: пересобрать быстрее, чем
    следить за тем, чтобы сохранённый не разъехался с векторным.
    """

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._lengths: list[int] = []
        # слово → {номер фрагмента: сколько раз встретилось}
        self._postings: dict[str, dict[int, int]] = {}
        self._total_length = 0

    # ------------------------------------------------------------ сборка

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Кладёт фрагменты в индекс. Возвращает, сколько добавилось.

        Токенизация делается здесь, один раз. Если разбирать текст на слова
        в момент поиска, каждый запрос заново читает весь корпус — на тысяче
        фрагментов это заметно, на десятках тысяч неприемлемо.
        """
        if not ids:
            return 0
        if len(ids) != len(texts):
            raise ValueError("ids и texts должны быть одной длины")

        metadatas = metadatas or [{} for _ in ids]
        known = set(self._ids)
        added = 0

        for doc_id, text, metadata in zip(ids, texts, metadatas):
            if doc_id in known:
                continue
            known.add(doc_id)

            index = len(self._ids)
            self._ids.append(doc_id)
            self._texts.append(text)
            self._metadatas.append(dict(metadata))

            length = 0
            for token in tokenize(text):
                posting = self._postings.setdefault(token, {})
                posting[index] = posting.get(index, 0) + 1
                length += 1

            self._lengths.append(length)
            self._total_length += length
            added += 1

        return added

    def clear(self) -> None:
        self.__init__(k1=self.k1, b=self.b)

    # ------------------------------------------------------------ веса

    def count(self) -> int:
        return len(self._ids)

    @property
    def average_length(self) -> float:
        """Средняя длина фрагмента в токенах — знаменатель поправки на длину."""
        return self._total_length / self.count() if self.count() else 0.0

    def document_frequency(self, token: str) -> int:
        """В скольких фрагментах встречается слово. Ноль — не встречается нигде."""
        return len(self._postings.get(token, ()))

    def idf(self, token: str) -> float:
        """Вес слова: чем реже встречается, тем больше.

        Форма с `+1` под логарифмом выбрана не случайно: у классической
        формулы слово, встречающееся больше чем в половине фрагментов,
        получает ОТРИЦАТЕЛЬНЫЙ вес и начинает штрафовать документы за то,
        что оно у них есть. Здесь вес частого слова стремится к нулю,
        но ниже не опускается.

        Слово, которого в корпусе нет вовсе, получает по этой формуле
        максимальный вес — и это не ошибка: до умножения на частоту дело
        не доходит, списка вхождений у него нет. Проверять «есть ли слово
        в корпусе» надо через document_frequency, а не через вес.
        """
        total = self.count()
        if not total:
            return 0.0
        frequency = self.document_frequency(token)
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    # ------------------------------------------------------------ поиск

    def scores(self, query: str) -> dict[int, float]:
        """Номер фрагмента → его вес по запросу. Только для тех, где есть совпадение.

        Обходятся списки вхождений слов запроса, а не все фрагменты подряд:
        по редкому имени это два документа из тысячи вместо тысячи.
        """
        total = self.count()
        if not total:
            return {}

        average = self.average_length
        found: dict[int, float] = {}

        for token in tokenize_query(query):
            posting = self._postings.get(token)
            if not posting:
                continue
            weight = self.idf(token)
            for index, frequency in posting.items():
                length = self._lengths[index] or 1
                # Знаменатель и есть поправка на длину: у фрагмента длиннее
                # среднего то же число вхождений весит меньше.
                saturation = frequency + self.k1 * (1 - self.b + self.b * length / (average or 1))
                found[index] = found.get(index, 0.0) + weight * frequency * (self.k1 + 1) / saturation

        return found

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        """Находит фрагменты по словам запроса.

        Пустой список здесь — содержательный ответ, а не сбой: ни одного
        слова запроса в корпусе нет. Векторный поиск такого сказать не умеет.
        """
        if not query or not query.strip():
            return []

        found = self.scores(query)
        if not found:
            return []

        best = sorted(found.items(), key=lambda item: -item[1])[:top_k]
        return [
            Hit(
                id=self._ids[index],
                text=self._texts[index],
                score=score,
                metadata=dict(self._metadatas[index]),
            )
            for index, score in best
        ]

    # ------------------------------------------------------------ статистика

    def stats(self) -> dict[str, object]:
        """Что внутри индекса: фрагменты, словарь, средняя длина."""
        return {
            "chunks": self.count(),
            "vocabulary": len(self._postings),
            "tokens": self._total_length,
            "average_length": round(self.average_length, 1),
        }
