"""Плагины для гибридного поиска по проекту."""
from chapter4.src import tools as chapter4_tools
from .indexer import index_project, hybrid_search, project_stats


@chapter4_tools.tool(
    "index_project",
    "индексирует все файлы проекта для гибридного поиска (векторы + ключевые слова). "
    "Используй, когда пользователь просит 'проиндексируй проект', 'прочитай весь проект'."
)
def index_project_tool(path: str = ".") -> str:
    result = index_project(path)
    msg = f"Проиндексировано {result['files']} файлов, {result['chunks']} фрагментов."
    if result["errors"]:
        msg += f"\nОшибки ({len(result['errors'])}): " + "; ".join(result["errors"][:5])
    return msg


@chapter4_tools.tool(
    "ask_project",
    "ищет информацию в коде проекта используя гибридный поиск (векторы + ключевые слова). "
    "Используй для вопросов о структуре проекта, функциях, классах, настройках. "
    "Проект должен быть предварительно проиндексирован через index_project."
)
def ask_project(query: str, max_results: int = 5) -> str:
    if not query or not query.strip():
        return "Ошибка: не указан поисковый запрос."
    results = hybrid_search(query, n_results=max_results)
    if not results:
        return "В проекте ничего не найдено. Возможно, проект ещё не проиндексирован."
    
    output = []
    for i, r in enumerate(results, 1):
        # Показываем оба score для отладки
        scores = f"vector={r['vector_score']:.2f}, bm25={r['bm25_score']:.2f}"
        output.append(f"{i}. [{r['path']}] (score={r['score']:.2f}, {scores})\n{r['text'][:800]}")
    return "\n\n".join(output)


@chapter4_tools.tool(
    "project_stats",
    "показывает статистику индекса проекта."
)
def project_stats_tool() -> str:
    stats = project_stats()
    if stats["chunks"] == 0:
        return "Проект ещё не проиндексирован."
    return f"Проиндексировано {stats['files']} файлов, {stats['chunks']} фрагментов."
