"""Диагностика: работает ли поиск по индексу проекта."""
from chapter6.src.indexer import project_stats, search_project

print("=" * 60)
print("Статистика индекса:", project_stats())
print("=" * 60)

queries = ["калькулятор", "calculator", "def calculator", "функция вычисления"]
for query in queries:
    print(f"\nЗапрос: '{query}'")
    results = search_project(query, n_results=3)
    if not results:
        print("  ❌ Ничего не найдено!")
        continue
    for r in results:
        print(f"  📄 [{r['path']}] (distance={r['distance']:.3f})")
        # Показываем первые 150 символов чанка
        preview = r["text"][:150].replace("\n", " | ")
        print(f"     {preview}")
