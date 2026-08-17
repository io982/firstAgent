"""Плагины для RAG по документам проекта."""

from chapter4.src import tools as chapter4_tools
from .indexer import index_project, search_project, project_stats


@chapter4_tools.tool(
    "index_project",
    "индексирует все файлы проекта в векторную базу для последующего поиска. "
    "Используй, когда пользователь просит 'проиндексируй проект', 'прочитай весь проект', "
    "'изучи код проекта'. Может занять время на больших проектах."
)
def index_project_tool(path: str = ".") -> str:
    result = index_project(path)
    msg = f"Проиндексировано {result['files']} файлов, {result['chunks']} фрагментов."
    if result["errors"]:
        msg += f"\nОшибки ({len(result['errors'])}): " + "; ".join(result["errors"][:5])
    return msg


@chapter4_tools.tool(
    "ask_project",
    "ищет информацию в коде и документах проекта. "
    "Используй для вопросов о структуре проекта, функциях, классах, настройках. "
    "Проект должен быть предварительно проиндексирован через index_project. "
    "Параметр query ОБЯЗАТЕЛЬНО должен быть непустым."
)
def ask_project(query: str, max_results: int = 5) -> str:
    if not query or not query.strip():
        return "Ошибка: не указан поисковый запрос. Используй параметр query."

    results = search_project(query, n_results=max_results)

    if not results:
        return ("В проекте ничего не найдено. "
                "Возможно, проект ещё не проиндексирован — используй index_project.")

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"{i}. [{r['path']}]\n{r['text'][:800]}")

    return "\n\n".join(output)


@chapter4_tools.tool(
    "project_stats",
    "показывает статистику индекса проекта: сколько файлов и фрагментов проиндексировано."
)
def project_stats_tool() -> str:
    stats = project_stats()
    if stats["chunks"] == 0:
        return "Проект ещё не проиндексирован. Используй index_project для индексации."
    return f"Проиндексировано {stats['files']} файлов, {stats['chunks']} фрагментов."