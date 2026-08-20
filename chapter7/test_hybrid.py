"""Замер качества поиска: векторы против ключевых слов против гибрида.

Запуск (нужны Ollama и проиндексированный проект):
    python -m chapter7.test_hybrid

Смысл главы 7 — не "добавить BM25", а показать, что поиск стал лучше.
"Лучше" здесь означает конкретное число: в какой доле запросов нужный файл
попал в первые три результата.
"""

from chapter7.src.indexer import hybrid_search, project_stats

# Пары "запрос → файлы, которые считаются правильным ответом".
# Правильных файлов может быть несколько: calculator, например, определён
# и в Главе 1, и в реестре плагинов Главы 4.
CASES = [
    ("def calculator", {"chapter1/agent.py", "chapter4/src/tools.py"}),
    ("где реализован калькулятор", {"chapter1/agent.py", "chapter4/src/tools.py"}),
    ("автозапуск сервера ollama", {"chapter1/agent.py"}),
    ("keep_alive сколько модель живёт в памяти", {"chapter1/agent.py"}),
    ("парсер вызовов инструментов из текста", {"chapter1/agent.py"}),
    ("перефразировщик запроса пользователя", {"chapter2/paraphraser.py"}),
    ("conversation_history история диалога", {"chapter3/agent.py"}),
    ("декоратор tool реестр плагинов", {"chapter4/src/tools.py"}),
    ("выполнение команды с подтверждением", {"chapter4/src/tools.py"}),
    ("http_get скачать страницу", {"chapter4/src/tools.py"}),
    ("remember recall долгосрочная память", {"chapter5/src/tools.py"}),
    ("префикс search_document для эмбеддингов", {"chapter5/src/embeddings.py"}),
    ("PersistentClient коллекция chromadb", {"chapter5/src/vectorstore.py",
                                             "chapter6/src/indexer.py",
                                             "chapter7/src/indexer.py"}),
    ("разбиение файла на чанки с номерами строк", {"chapter6/src/indexer.py",
                                                   "chapter7/src/indexer.py"}),
    ("idf inverse document frequency", {"chapter7/src/bm25.py"}),
    ("reciprocal rank fusion слияние", {"chapter7/src/indexer.py"}),
    ("настройки линтера ruff", {"ruff.toml"}),
    # Запрос из отчёта об ошибке: определение инструмента, а не его вызовы
    ("где определен инструмент агента калькулятор", {"chapter4/src/tools.py",
                                                     "chapter1/agent.py"}),
]

TOP_N = 3
MODES = ["vector", "bm25", "hybrid"]


def evaluate(mode: str) -> dict:
    """Прогоняет весь набор в одном режиме и считает попадания."""
    hits = 0
    reciprocal_ranks = 0.0
    misses = []

    for query, expected in CASES:
        results = hybrid_search(query, n_results=TOP_N, mode=mode)
        paths = [r["path"] for r in results]

        rank = next((i for i, path in enumerate(paths, start=1) if path in expected), None)
        if rank:
            hits += 1
            reciprocal_ranks += 1.0 / rank
        else:
            misses.append((query, paths))

    total = len(CASES)
    return {
        "mode": mode,
        "hits": hits,
        "total": total,
        "hit_rate": hits / total,
        "mrr": reciprocal_ranks / total,
        "misses": misses,
    }


def main():
    stats = project_stats()
    print("=" * 64)
    print(f"Индекс: {stats['files']} файлов, {stats['chunks']} фрагментов")
    print(f"Набор: {len(CASES)} запросов, засчитываем попадание в топ-{TOP_N}")
    print("=" * 64)

    reports = []
    for mode in MODES:
        report = evaluate(mode)
        reports.append(report)
        print(f"  {mode:<8} попаданий {report['hits']:>2}/{report['total']}"
              f"   hit@{TOP_N} {report['hit_rate']:.0%}   MRR {report['mrr']:.2f}")

    print()
    baseline = next(r for r in reports if r["mode"] == "vector")
    hybrid = next(r for r in reports if r["mode"] == "hybrid")
    delta = hybrid["hit_rate"] - baseline["hit_rate"]
    print(f"Гибрид против чистых векторов: {delta:+.0%} по hit@{TOP_N}, "
          f"{hybrid['mrr'] - baseline['mrr']:+.2f} по MRR")

    if hybrid["misses"]:
        print(f"\nЧего гибрид не нашёл ({len(hybrid['misses'])}):")
        for query, paths in hybrid["misses"]:
            print(f"  '{query}'")
            print(f"     вернул: {', '.join(paths) if paths else 'ничего'}")


if __name__ == "__main__":
    main()
