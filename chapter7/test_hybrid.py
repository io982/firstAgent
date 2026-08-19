"""Сравнение векторного и гибридного поиска."""
from chapter7.src.indexer import hybrid_search, project_stats

print("=" * 60)
print("Статистика:", project_stats())
print("=" * 60)

queries = [
    "def calculator",
    "calculator",
    "где реализован калькулятор",
    "keep_alive",
]

for query in queries:
    print(f"\nЗапрос: '{query}'")
    results = hybrid_search(query, n_results=3)
    if not results:
        print("  ❌ Ничего не найдено")
        continue
    for r in results:
        print(f"  📄 [{r['path']}] (score={r['score']:.2f}, vec={r['vector_score']:.2f}, bm25={r['bm25_score']:.2f})")
        preview = r["text"][:100].replace("\n", " | ")
        print(f"     {preview}")
