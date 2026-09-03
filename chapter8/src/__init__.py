"""
Компоненты Главы 8: границы, файлы, правки, запуск, git, план, конвейер.

⚠️ Импорт этого пакета РЕГИСТРИРУЕТ 20 новых инструментов в общем реестре
Главы 2 — семь файловых, четыре запуска, пять git, четыре про окружение.
Это отличает главу от Главы 7, которая не добавила ни одного: там
специалист получался выборкой из существующих, здесь у агента
появляются права, которых раньше не было ни у кого.

⚠️ Специалиста пакет НЕ регистрирует: единственный исполнитель главы
объявлен в `chapter8/agent.py` — рядом со своими правилами, как
инструменты Главы 3 объявлены в Главе 3, а не в реестре Главы 2.

Порядок модулей внутри пакета — это порядок, в котором они писались,
и он не случаен:

  * `guard` — рабочий каталог, белый список, время, сухой прогон,
    подтверждение, журнал отката. Первым, потому что до ответа на вопрос
    «что будет, если модель попросит переписать чужой файл» писать
    `write_file` рано;
  * `edits` — четыре формы правки и проверки после неё: синтаксис
    и пропавшие определения.
    Единственный модуль главы, который ничего не знает ни о диске,
    ни о моделях: на входе текст, на выходе текст;
  * `fs`, `shell`, `vcs`, `env` — инструменты: файлы, запуск процессов,
    git, виртуальное окружение и зависимости. Каждый действует
    через `guard`;
  * `planner` — задача человека в шаги, которые можно показать
    до первой правки;
  * `pipeline` — конвейер на графе Главы 7: цикл по шагам плана
    и цикл починки с ходом назад от провалившейся проверки
    к перечитыванию файла;
  * `pipeline_lg` — тот же конвейер на LangGraph. Импортируется
    отдельно и только по требованию: библиотека необязательная.

Если нужны только формы правки, без реестра инструментов и без цепочки
глав, импортируйте подмодуль прямо:

    from chapter8.src.edits import apply_anchor, apply_lines

— edits.py не зависит ни от одной главы курса и ни от чего, кроме
стандартной библиотеки. То же верно для guard.py.
"""
from .edits import (
    ANCHOR,
    APPEND,
    EDIT_FORMS,
    FULL,
    LINES,
    REPLACING_FORMS,
    EditResult,
    apply_anchor,
    apply_append,
    apply_full,
    apply_lines,
    definitions,
    describe_forms,
    edit_schema,
    lost_definitions,
    missing_fields,
    syntax_ok,
    unified,
)
from .env import (
    ENV_TOOLS,
    REQUIREMENTS,
    check_imports,
    create_venv,
    env_report,
    has_venv,
    imported_modules,
    install,
    missing_imports,
    package_for,
    write_requirements,
)
from .fs import (
    FS_TOOLS,
    MAX_LINES,
    OUTPUT_LIMIT,
    SKIP_DIRS,
    append_to_file,
    edit_file,
    list_dir,
    put_file,
    read_lines,
    replace_lines,
    search_files,
    write_file,
)
from .guard import (
    ALLOW,
    ANY_COMMAND,
    ASK,
    AUTO,
    DENIED,
    DENY,
    DRY,
    NARROW_ALLOWED,
    Change,
    OutsideWorkspace,
    Policy,
    change_count,
    changed_files,
    check,
    command_allowed,
    forget_changes,
    get_policy,
    get_workspace,
    program_of,
    record,
    relative,
    reset_policy,
    resolve_path,
    rollback,
    set_policy,
    set_workspace,
    split_command,
    verdict_message,
)
from .pipeline import (
    CODER_MODEL,
    CREATE_RULES,
    EDIT_RULES,
    MAX_ATTEMPTS,
    apply_edit,
    build_pipeline,
    coder_model,
    file_schema,
    normalize_edit,
    run_pipeline,
)
from .planner import (
    FROM_FALLBACK,
    FROM_MODEL,
    MAX_STEPS,
    PLAN_ACTIONS,
    PLANNER,
    PLANNER_MODEL,
    Plan,
    Step,
    fallback_plan,
    looks_like_fix,
    looks_like_one_file,
    make_plan,
    named_file,
    parse_plan,
    plan_kind,
    plan_schema,
    planner_model,
    render_plan,
    split_target,
    validate_plan,
)
from .shell import (
    RUN_TOOLS,
    Run,
    execute,
    first_error,
    run_command,
    run_lint,
    run_python,
    run_tests,
    suite_passed,
)
from .vcs import (
    GIT_TOOLS,
    current_branch,
    git_branch,
    git_commit,
    git_diff,
    git_log,
    git_status,
    is_repo,
)

__all__ = [
    # guard
    "ALLOW", "ANY_COMMAND", "ASK", "AUTO", "DENIED", "DENY", "DRY", "NARROW_ALLOWED",
    "Change", "OutsideWorkspace", "Policy", "change_count", "changed_files", "check",
    "command_allowed", "forget_changes", "get_policy", "get_workspace",
    "program_of", "record", "relative", "reset_policy", "resolve_path",
    "rollback", "set_policy", "set_workspace", "split_command", "verdict_message",
    # edits
    "ANCHOR", "APPEND", "EDIT_FORMS", "FULL", "LINES", "REPLACING_FORMS", "EditResult",
    "apply_anchor", "apply_append",
    "apply_full", "apply_lines", "definitions", "describe_forms", "edit_schema",
    "lost_definitions", "missing_fields", "syntax_ok", "unified",
    # fs
    "FS_TOOLS", "MAX_LINES", "OUTPUT_LIMIT", "SKIP_DIRS", "edit_file",
    "append_to_file", "list_dir", "put_file", "read_lines", "replace_lines", "search_files",
    "write_file",
    # shell
    "RUN_TOOLS", "Run", "execute", "first_error", "run_command", "run_lint",
    "run_python", "run_tests", "suite_passed",
    # vcs
    "GIT_TOOLS", "current_branch", "git_branch", "git_commit", "git_diff",
    "git_log", "git_status", "is_repo",
    # planner
    "FROM_FALLBACK", "FROM_MODEL", "MAX_STEPS", "PLANNER", "PLANNER_MODEL", "looks_like_fix",
    "PLAN_ACTIONS", "Plan", "Step", "fallback_plan", "looks_like_one_file", "make_plan",
    "named_file", "parse_plan", "plan_kind",
    "plan_schema", "planner_model", "render_plan", "split_target", "validate_plan",
    # pipeline
    "CODER_MODEL", "CREATE_RULES", "EDIT_RULES", "MAX_ATTEMPTS", "apply_edit",
    "build_pipeline", "coder_model", "file_schema", "normalize_edit", "run_pipeline",
    # env
    "ENV_TOOLS", "REQUIREMENTS", "check_imports", "create_venv", "env_report",
    "has_venv", "imported_modules", "install", "missing_imports", "package_for",
    "write_requirements",
]
