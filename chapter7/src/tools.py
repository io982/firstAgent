"""Плагины для гибридного поиска по проекту."""

from chapter4.src import tools as chapter4_tools

from .indexer import hybrid_search, index_project, project_stats

# Бюджет выдачи ask_project. Всё, что вернёт инструмент, попадает в контекст
# модели, а он не резиновый — см. base.NUM_CTX в chapter7/agent.py.
MAX_FRAGMENT_CHARS = 600
MAX_TOTAL_CHARS = 3000


def _trim_to_whole_lines(text: str, limit: int) -> str:
    """Обрезает текст по границе строк, а не посреди неё.

    Каждая строка фрагмента начинается с номера ("27: def calculator").
    Обрывок без номера модель принимает за отдельную строку кода
    и потом ссылается на несуществующие места в файле.
    """
    if len(text) <= limit:
        return text

    kept = []
    used = 0
    for line in text.split("\n"):
        if used + len(line) + 1 > limit:
            break
        kept.append(line)
        used += len(line) + 1

    if not kept:
        return text[:limit]

    kept.append("... [фрагмент обрезан] ...")
    return "\n".join(kept)


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
    "ищет информацию в коде проекта гибридным поиском (векторы + ключевые слова). "
    "Используй для вопросов о структуре проекта, функциях, классах, настройках. "
    "Проект должен быть предварительно проиндексирован через index_project. "
    "Параметр query ОБЯЗАТЕЛЬНО должен быть непустым."
)
def ask_project(query: str, max_results: int = 5) -> str:
    if not query or not query.strip():
        return "Ошибка: не указан поисковый запрос. Используй параметр query."

    results = hybrid_search(query, n_results=max_results)
    if not results:
        return ("В проекте ничего не найдено. "
                "Возможно, проект ещё не проиндексирован — используй index_project.")

    # Служебные scores модели не показываем: они занимают место в контексте
    # и провоцируют рассуждать о числах вместо ответа на вопрос. Для отладки
    # они доступны напрямую из hybrid_search — см. chapter7/test_hybrid.py.
    output = []
    total = 0
    for i, r in enumerate(results, 1):
        fragment = f"{i}. [{r['path']}]\n{_trim_to_whole_lines(r['text'], MAX_FRAGMENT_CHARS)}"
        if total + len(fragment) > MAX_TOTAL_CHARS:
            output.append(f"... [показаны {i - 1} из {len(results)} найденных фрагментов] ...")
            break
        output.append(fragment)
        total += len(fragment)

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
