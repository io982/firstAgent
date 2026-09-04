"""
Git глазами агента: посмотреть, что изменилось, и зафиксировать (8.2).

Зачем кодинг-агенту git — вопрос не праздный, потому что править файлы
он умеет и без него. Ответ в том, что git даёт агенту две вещи, которых
у него иначе нет.

Первая — **память о том, что было до него**. Журнал правок из `guard`
знает только про файлы, которые трогал агент; `git diff` показывает
разницу целиком, даже если правки делал человек.

Вторая — **точка, к которой можно вернуться**. Откат из `guard`
отменяет работу агента (снимком в git, см. history.py). Коммит —
это отметка, к которой возвращаются через неделю, и в неё попадает
то, что человек одобрил, а не всё подряд.

Разделение то же, что и везде в этой главе: пять инструментов, три
из них только смотрят и не спрашивают ничего, два меняют состояние
репозитория и проходят через `guard`.

Отдельное решение — **что именно попадает в коммит**. Не `git add -A`,
а только те файлы, которые агент действительно трогал: их список ведёт
журнал `guard.changed_files()`. Разница не косметическая. `git add -A`
забирает в коммит и то, что человек правил параллельно в соседнем окне,
и временные файлы, и вывод тестов, — а отвечать за содержимое коммита
будет человек, чьё имя в нём стоит.
"""
from __future__ import annotations

from chapter2.src.tools import tool
from chapter8.src import guard
from chapter8.src.shell import clip, execute

# Потолок на вывод git. Отдельный от общего: `git diff` без потолка —
# это весь код проекта в контексте модели.
GIT_OUTPUT_LIMIT = 2000

# Сколько коммитов показывать без явной просьбы.
LOG_LIMIT = 5


def is_repo() -> bool:
    """Является ли рабочий каталог репозиторием git.

    Проверяется через саму программу, а не наличием папки `.git`:
    каталог внутри репозитория своей `.git` не имеет, а git там
    работает — и агент, запущенный в подпапке, должен это видеть.
    """
    run = execute("git rev-parse --is-inside-work-tree", timeout=10.0)
    return run.ok and run.out.strip() == "true"


def _need_repo() -> str:
    """Сообщение об отсутствии репозитория — или пустая строка."""
    if is_repo():
        return ""
    return (
        f"Каталог {guard.get_workspace()} — не репозиторий git. "
        "Инструменты git здесь работать не будут."
    )


def current_branch() -> str:
    """Имя текущей ветки. Пустая строка, если определить не удалось."""
    run = execute("git rev-parse --abbrev-ref HEAD", timeout=10.0)
    return run.out.strip() if run.ok else ""


# ====================================================================
# ТОЛЬКО СМОТРЯТ
# ====================================================================

@tool
def git_status() -> str:
    """Показывает, какие файлы изменены, добавлены или не отслеживаются."""
    problem = _need_repo()
    if problem:
        return problem

    run = execute("git status --short --branch")
    if not run.ok:
        return f"git status не отработал (код {run.code}):\n{clip(run.text(), GIT_OUTPUT_LIMIT)}"
    body = run.out.strip()
    if len(body.splitlines()) <= 1:
        return f"{body}\nИзменений нет: рабочий каталог совпадает с последним коммитом."
    return clip(body, GIT_OUTPUT_LIMIT)


@tool
def git_diff(path: str = "") -> str:
    """Показывает изменения: без пути — сводку по файлам, с путём — сам diff."""
    problem = _need_repo()
    if problem:
        return problem

    if not path.strip():
        # Без пути показывается СВОДКА, а не изменения целиком. Полный
        # diff по проекту — это тысячи строк, и в контексте на 4096
        # токенов он не оставляет места ни вопросу, ни ответу.
        run = execute("git diff --stat")
        body = run.out.strip()
        return clip(body, GIT_OUTPUT_LIMIT) if body else "Изменений нет."

    try:
        where = guard.relative(guard.resolve_path(path))
    except guard.OutsideWorkspace as exc:
        return str(exc)

    run = execute(["git", "diff", "--", where])
    body = run.out.strip()
    return clip(body, GIT_OUTPUT_LIMIT) if body else f"Изменений в {where} нет."


@tool
def git_log(limit: str = "5") -> str:
    """Показывает последние коммиты — по одной строке на каждый."""
    problem = _need_repo()
    if problem:
        return problem

    try:
        count = max(1, min(int(str(limit).strip()), 50))
    except (TypeError, ValueError):
        count = LOG_LIMIT

    run = execute(f"git log --oneline -n {count}")
    if not run.ok:
        return f"git log не отработал (код {run.code}):\n{clip(run.text(), GIT_OUTPUT_LIMIT)}"
    return clip(run.out.strip() or "История пуста: коммитов ещё нет.", GIT_OUTPUT_LIMIT)


# ====================================================================
# МЕНЯЮТ СОСТОЯНИЕ
# ====================================================================

@tool
def git_branch(name: str = "") -> str:
    """Без имени показывает текущую ветку, с именем — создаёт её и переключается."""
    problem = _need_repo()
    if problem:
        return problem

    if not name.strip():
        run = execute("git branch --list")
        return f"Текущая ветка: {current_branch()}\n\n{clip(run.out.strip(), GIT_OUTPUT_LIMIT)}"

    branch = name.strip()
    # Имя ветки уезжает в командную строку, поэтому проверяется до того,
    # а не «git разберётся». Пробел или дефис в начале превращают имя
    # в ключ команды — самый простой способ попросить git не о том.
    if (branch.startswith(("-", "."))
            or branch.endswith((".", ".lock"))
            or ".." in branch
            or any(ch in branch for ch in " ~^:?*[\\")):
        return f"Недопустимое имя ветки: {branch!r}"

    verdict = guard.check(f"создать ветку {branch}", f"из текущей: {current_branch()}")
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, f"создать ветку {branch}")

    run = execute(["git", "checkout", "-b", branch])
    if not run.ok:
        return f"Ветка не создана (код {run.code}):\n{clip(run.text(), GIT_OUTPUT_LIMIT)}"
    return f"Создана ветка {branch} и выполнено переключение на неё."


@tool
def git_commit(message: str) -> str:
    """Фиксирует в коммите только те файлы, которые менял агент."""
    problem = _need_repo()
    if problem:
        return problem

    text = (message or "").strip()
    if not text:
        return "Пустое сообщение коммита. Коммит без сообщения бесполезен через неделю."

    touched = [guard.relative(p) for p in guard.changed_files() if p.exists()]
    if not touched:
        return (
            "Агент в этой сессии файлов не менял — коммитить нечего. "
            "Файлы, изменённые вручную, агент не фиксирует намеренно."
        )

    detail = "Сообщение: " + text + "\nФайлы:\n" + "\n".join(f"  {p}" for p in touched)
    verdict = guard.check(f"зафиксировать {len(touched)} файл(ов) в коммите", detail)
    if verdict != guard.ALLOW:
        return guard.verdict_message(verdict, "коммит")

    added = execute(["git", "add", "--", *touched])
    if not added.ok:
        return f"git add не отработал (код {added.code}):\n{clip(added.text(), GIT_OUTPUT_LIMIT)}"

    # Пути перечислены и в commit тоже: между add и commit кто-то мог
    # добавить в индекс своё, и без явного списка это уехало бы
    # в коммит агента.
    run = execute(["git", "commit", "-m", text, "--", *touched])
    if not run.ok:
        return f"Коммит не создан (код {run.code}):\n{clip(run.text(), GIT_OUTPUT_LIMIT)}"

    # Журнал очищается: зафиксированное больше не подлежит откату
    # средствами агента — для этого теперь есть git.
    guard.forget_changes()
    return f"Коммит создан ({len(touched)} файл(ов)).\n{clip(run.out.strip(), GIT_OUTPUT_LIMIT)}"


# Имена git-инструментов — для выборки специалиста и схемы ответа.
GIT_TOOLS = ["git_status", "git_diff", "git_log", "git_branch", "git_commit"]
