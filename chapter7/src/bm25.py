"""BM25 — классический алгоритм поиска по ключевым словам.
BM25 (Best Matching 25) — это алгоритм ранжирования документов по запросу.
Он учитывает:
- Частоту слова в документе (TF — Term Frequency)
- Редкость слова во всей коллекции (IDF — Inverse Document Frequency)
- Длину документа (короткие документы с совпадением ранжируются выше)
Мы используем упрощённую версию без полной формулы BM25, но с ключевыми идеями.
"""
import math
import re
from collections import Counter


class SimpleBM25:
    """Упрощённая реализация BM25 для поиска по чанкам кода."""

    def __init__(self, documents: list[dict]):
        """
        Инициализирует индекс BM25.
        Args:
            documents: список словарей {"id": str, "text": str, "metadata": dict}
        """
        self.documents = documents
        self.doc_count = len(documents)
        # Токенизируем каждый документ один раз при построении индекса.
        # Если делать это в search(), каждый запрос заново разбирает весь
        # индекс на слова — на 250 фрагментах это заметно, на 10 000 неприемлемо.
        self.doc_tokens = [Counter(self._tokenize(doc["text"])) for doc in documents]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_tokens]
        # Подсчитываем IDF для каждого слова
        self.idf = self._compute_idf()
        # Средняя длина документа
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.doc_count)

    def _tokenize(self, text: str) -> list[str]:
        """Разбивает текст на токены (слова).

        Для кода важно сохранять специальные символы как отдельные токены:
        def, class, =, (, ), {, } и т.д.
        """
        # Разбиваем по пробелам и пунктуации, но сохраняем символы кода
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens

    def _compute_idf(self) -> dict[str, float]:
        """Вычисляет IDF (Inverse Document Frequency) для каждого слова.

        IDF показывает, насколько редко слово встречается в коллекции:
        - Редкие слова (например, "calculator") имеют высокий IDF
        - Частые слова (например, "def", "the") имеют низкий IDF
        """
        df = Counter()  # Document Frequency — в скольких документах встречается слово
        for counter in self.doc_tokens:
            for token in counter:
                df[token] += 1

        idf = {}
        for term, freq in df.items():
            # Формула IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            # N — общее количество документов
            # df — количество документов с этим словом
            idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Ищет документы по запросу и возвращает топ-n результатов.

        Args:
            query: поисковый запрос
            n_results: количество результатов
        Returns:
            Список словарей {"id", "text", "metadata", "score"}
        """
        query_tokens = self._tokenize(query)
        scores = []
        for index, doc in enumerate(self.documents):
            score = self._score_document(index, query_tokens)
            if score > 0:
                scores.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        # Сортируем по убыванию score
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:n_results]

    def _score_document(self, index: int, query_tokens: list[str]) -> float:
        """Вычисляет релевантность документа запросу.

        Упрощённая формула BM25:
        score = sum(IDF(term) * TF(term, doc)) для всех термов в запросе
        TF (Term Frequency) — частота терма в документе, нормализованная по длине.
        """
        term_freq = self.doc_tokens[index]
        doc_len = self.doc_lengths[index]

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            # Нормализованная TF: учитываем длину документа
            # Короткие документы с совпадением получают бонус
            normalized_tf = tf / (1 + 0.75 * (doc_len / self.avg_doc_len - 1))
            # BM25-подобная формула
            idf = self.idf[term]
            score += idf * normalized_tf * (1 + 0.75 * tf / (tf + 1))
        return score
