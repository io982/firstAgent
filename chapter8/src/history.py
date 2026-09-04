"""
Откат через git: снимок до работы агента и журнал, переживающий перезапуск.

Первая версия отката жила в оперативной памяти: перед каждой записью
`guard` запоминал прежний текст файла в список, а «откат» проходил
по списку назад. Это работает ровно до первой неприятности. Агент
закрыт, упал, потерял связь с Ollama — списка нет, и файлы остаются
в том виде, в каком их застали. Единственная настоящая защита главы
оказалась самой недолговечной её частью.

Здесь та же работа поручена git, и он даёт две вещи, которых у списка
в памяти не было.

**Точка, к которой можно вернуться.** Перед первой записью агент
делает СНИМОК рабочего каталога и вешает на него ветку с временем
в имени — `agent-snapshot/2026-09-04-16-40-12`. Ветка живёт в `.git`
и переживает что угодно: человек может посмотреть `git diff
agent-snapshot/...`, вернуть отдельный файл или всё сразу, спустя
неделю и без агента.

**Хранилище прежнего содержимого.** Перед каждой записью прежний текст
файла кладётся в объектное хранилище git (`git hash-object -w`), а его
идентификатор — в журнал на диске. Откат читает журнал и достаёт текст
обратно. Перезапуск агента этому не мешает.

Чего этот механизм НЕ делает, и это важнее того, что он делает.

**Он не трогает вашу работу в git.** Ни `HEAD`, ни рабочий каталог,
ни индекс — то место, где лежит `git add`, сделанный человеком, — не
меняются. Снимок собирается во ВРЕМЕННОМ индексе (`GIT_INDEX_FILE`),
дерево пишется из него, коммит создаётся `commit-tree` напрямую, и
ветка ставится на него `git branch`. Ни одна из этих команд не двигает
то, над чем работает человек. Обычный `git checkout -b`, которым такое
делают на первый взгляд, переключил бы `HEAD` — и агент, начавший
задачу, унёс бы человека на другую ветку.

**Он не восстанавливает то, чего не трогал.** Откат идёт по журналу,
файл за файлом. Правка, сделанная человеком в соседнем окне, в журнал
не попадает и остаётся на месте. Полный откат каталога к снимку —
это `git checkout agent-snapshot/...`, и делает его человек, а не агент.

**Он не работает вне репозитория.** Каталог без git — обычный случай:
человек попросил агента написать программу в пустой папке. Там
остаётся прежний список в памяти, и агент говорит об этом вслух,
а не делает вид, что защищён.

Модуль намеренно не импортирует ничего из главы. `guard` зовёт его,
а он зовёт только git — иначе получилось бы кольцо: `guard` → история
→ запуск процессов → снова `guard`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

# Имя ветки со снимком. Со слэшем и временем: ветки агента тогда
# складываются в отдельную «папку» и не мешаются среди веток человека,
# а по времени видно, к какому запуску снимок относится.
BRANCH_PREFIX = "agent-snapshot/"

# Журнал: что агент изменил с последнего снимка. Лежит рядом с главой,
# а не в рабочем каталоге, по той же причине, что и память сессии
# (см. session.py): каталог агент меняет по команде человека, и журнал,
# лежащий внутри, при первой же смене стал бы другим журналом.
DEFAULT_JOURNAL = Path(__file__).parent.parent / "history.json"

# Сколько ждать git. Снимок каталога — единственная тяжёлая операция
# здесь: на большом репозитории `add -A` читает всё дерево.
GIT_TIMEOUT = 60.0

# Кем подписан снимок, если в репозитории не настроен пользователь.
# Без этого `commit-tree` откажется работать на чистой машине —
# и откат сломался бы там, где он нужнее всего.
IDENTITY = {
    "GIT_AUTHOR_NAME": "coding agent",
    "GIT_AUTHOR_EMAIL": "agent@localhost",
    "GIT_COMMITTER_NAME": "coding agent",
    "GIT_COMMITTER_EMAIL": "agent@localhost",
}


def journal_path() -> Path:
    """Где лежит журнал. Переменная окружения сильнее умолчания.

    Переменная нужна не для гибкости, а для проверок: без подмены пути
    тесты писали бы в журнал разработчика и откатывали его файлы.
    """
    return Path(os.environ.get("AGENT_HISTORY_FILE") or DEFAULT_JOURNAL)


def _git(root: Path, args: list[str], feed: bytes | None = None,
         extra_env: dict[str, str] | None = None) -> tuple[bool, bytes]:
    """Запускает git в каталоге `root`. Возвращает «получилось» и вывод.

    Байты, а не текст: через эту функцию проходит содержимое файлов,
    и декодировать его в строку значит испортить всё, что не UTF-8.
    """
    program = shutil.which("git")
    if not program:
        return False, b""
    env = dict(os.environ)
    env.update(IDENTITY)
    if extra_env:
        env.update(extra_env)
    try:
        done = subprocess.run(
            [program, *args],
            cwd=str(root),
            capture_output=True,
            input=feed,
            timeout=GIT_TIMEOUT,
            env=env,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, b""
    return done.returncode == 0, done.stdout


# Ответ «репозиторий или нет» по каталогам. Кэш нужен не ради скорости,
# а чтобы не запускать git на КАЖДУЮ запись в файл:
# в пустой папке без репозитория ответ один и тот же, а процессов
# набежало бы по три на каждый написанный файл.
_IS_REPO: dict[str, bool] = {}


def available(root: Path) -> bool:
    """Есть ли git и является ли каталог репозиторием.

    Спрашивается у самой программы, а не по наличию папки `.git`:
    подкаталог репозитория своей `.git` не имеет, а git там работает.
    Ответ запоминается: каталог не становится репозиторием посреди
    прогона, а если человек сделал `git init` при живом агенте —
    поможет смена каталога, она кэш и сбрасывает.
    """
    key = str(Path(root))
    if key not in _IS_REPO:
        ok, out = _git(root, ["rev-parse", "--is-inside-work-tree"])
        _IS_REPO[key] = ok and out.strip() == b"true"
    return _IS_REPO[key]


def forget_available() -> None:
    """Забывает, какие каталоги были репозиториями."""
    _IS_REPO.clear()


def _read_journal() -> dict:
    """Журнал с диска. Битый или отсутствующий — пустой."""
    try:
        data = json.loads(journal_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_journal(data: dict) -> None:
    """Журнал на диск. Молча, если писать некуда: откат — не главная
    работа агента, и ронять прогон из-за него нельзя."""
    try:
        path = journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def pending(root: Path) -> list[str]:
    """Файлы, которые журнал считает изменёнными в этом каталоге.

    Пусто, если журнал от другого каталога: чужой откат применять
    нельзя, а молча его стирать — терять чужую страховку.
    """
    data = _read_journal()
    if not data or Path(data.get("root", "")) != Path(root):
        return []
    return [item["path"] for item in data.get("changes", [])]


def branch(root: Path) -> str:
    """Имя ветки со снимком текущего журнала — или пустая строка."""
    data = _read_journal()
    if not data or Path(data.get("root", "")) != Path(root):
        return ""
    return str(data.get("branch", ""))


def start(root: Path) -> str:
    """Делает снимок каталога и вешает на него ветку. Возвращает её имя.

    Порядок команд выбран так, чтобы не задеть ничего человеческого:

      * `add -A` во ВРЕМЕННЫЙ индекс — настоящий индекс с тем, что
        человек уже добавил к следующему коммиту, остаётся нетронутым;
      * `write-tree` превращает временный индекс в дерево;
      * `commit-tree` делает из дерева коммит с родителем `HEAD`
        (или без родителя, если репозиторий пуст);
      * `branch` ставит на этот коммит имя.

    Ни одна команда не двигает `HEAD` и не трогает файлы на диске.
    """
    index = root / ".git" / "agent-snapshot-index"
    env = {"GIT_INDEX_FILE": str(index)}
    try:
        if index.exists():
            index.unlink()
    except OSError:
        pass

    ok, _ = _git(root, ["add", "-A"], extra_env=env)
    if not ok:
        return ""
    ok, out = _git(root, ["write-tree"], extra_env=env)
    if not ok:
        return ""
    tree = out.decode("utf-8", "replace").strip()

    args = ["commit-tree", tree, "-m", "снимок до работы агента"]
    has_head, head = _git(root, ["rev-parse", "HEAD"])
    if has_head:
        args += ["-p", head.decode("utf-8", "replace").strip()]
    ok, out = _git(root, args, extra_env=env)
    if not ok:
        return ""
    commit = out.decode("utf-8", "replace").strip()

    name = BRANCH_PREFIX + time.strftime("%Y-%m-%d-%H-%M-%S")
    ok, _ = _git(root, ["branch", "-f", name, commit])
    if not ok:
        return ""
    _write_journal({"root": str(root), "commit": commit, "branch": name, "changes": []})
    try:
        index.unlink()
    except OSError:
        pass
    return name


def remember(root: Path, path: Path, before: str | None) -> bool:
    """Кладёт прежнее содержимое файла в git и путь — в журнал.

    Снимок делается лениво, при первой же записи: запуск, в котором
    агент ничего не изменил, не должен оставлять после себя веток.

    `before is None` означает «файла не было» — откат такой файл
    удалит, а не восстановит.
    """
    data = _read_journal()
    if not data or Path(data.get("root", "")) != Path(root):
        if not start(root):
            return False
        data = _read_journal()
        if not data:
            return False

    blob = None
    if before is not None:
        ok, out = _git(root, ["hash-object", "-w", "--stdin"], feed=before.encode("utf-8"))
        if not ok:
            return False
        blob = out.decode("utf-8", "replace").strip()

    try:
        name = str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return False

    # Первая запись о файле — самая ценная: в ней его состояние ДО
    # работы агента. Повторные правки того же файла журнал не удлиняют.
    if any(item["path"] == name for item in data["changes"]):
        return True
    data["changes"].append({"path": name, "blob": blob})
    _write_journal(data)
    return True


def restore(root: Path) -> list[str]:
    """Возвращает файлы журнала в состояние до работы агента.

    Ветку со снимком не трогает: она остаётся человеку — посмотреть,
    что происходило, или вернуть каталог целиком.
    """
    data = _read_journal()
    if not data or Path(data.get("root", "")) != Path(root):
        return []

    done: list[str] = []
    for item in reversed(data.get("changes", [])):
        name = item["path"]
        target = Path(root) / name
        try:
            if item["blob"] is None:
                if target.exists():
                    target.unlink()
                    done.append(f"удалён {name}")
                continue
            ok, out = _git(root, ["cat-file", "blob", item["blob"]])
            if not ok:
                done.append(f"НЕ ОТКАЧЕН {name}: снимок содержимого не найден")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(out)
            done.append(f"восстановлен {name}")
        except OSError as exc:
            done.append(f"НЕ ОТКАЧЕН {name}: {exc}")

    clear()
    return done


def clear() -> None:
    """Забывает журнал. Ветку со снимком оставляет на месте."""
    try:
        journal_path().unlink()
    except OSError:
        pass
