from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .state.config import (
    ConfigError,
    active_model_name,
    agent_dir,
    default_system_config_path,
    load_system_config,
    load_project_config,
    project_config_path,
    project_config_template,
    resolve_model_profile,
    system_config_template,
    write_json,
)
from .runtime.agent import AgentOptions, run_agent, run_chat
from .tools.shell import shell_name
from .state.tasks import TaskManager, TodoManager
from .state.team import MessageBus, TeamManager
from .ue import discover_ue, render_build_result, render_doctor, render_run_result, run_ue_build, run_ue_python
from .tools.worktrees import WorktreeManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uedev",
        description="Agentic CLI for coding and automation workflows.",
    )
    parser.add_argument("--version", action="version", version=f"uedev {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create JSON configuration files")
    init_parser.add_argument("--force", action="store_true", help="overwrite existing JSON config files")

    subparsers.add_parser("doctor", help="check local configuration")

    run_parser = subparsers.add_parser("run", help="run one agent task")
    run_parser.add_argument("task", nargs="+", help="task for the agent")
    run_parser.add_argument("--max-iterations", "--max-steps", dest="max_steps", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--timeout", type=int, default=120, help="shell command timeout in seconds")
    run_parser.add_argument("-y", "--yes", action="store_true", help="start this run in full-access permission mode")
    run_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory for shell commands")
    run_parser.add_argument("--verbose", action="store_true", help="show internal iteration and tool diagnostics")
    run_parser.add_argument(
        "--context-threshold",
        type=int,
        default=None,
        help="estimated token threshold for auto compact; defaults to 90%% of the active model context_window",
    )

    chat_parser = subparsers.add_parser("chat", help="start an interactive agent chat session")
    chat_parser.add_argument("--max-iterations", "--max-steps", dest="max_steps", type=int, default=None, help=argparse.SUPPRESS)
    chat_parser.add_argument("--timeout", type=int, default=120, help="shell command timeout in seconds")
    chat_parser.add_argument("-y", "--yes", action="store_true", help="start this chat in full-access permission mode")
    chat_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory for shell commands")
    chat_parser.add_argument("--verbose", action="store_true", help="show internal iteration and tool diagnostics")
    chat_parser.add_argument(
        "--context-threshold",
        type=int,
        default=None,
        help="estimated token threshold for auto compact; defaults to 90%% of the active model context_window",
    )
    chat_parser.add_argument("--plain", action="store_true", help="use the script-friendly plain chat renderer")

    task_parser = subparsers.add_parser("tasks", help="show the persisted agent todo list")
    task_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .agent state")
    task_parser.add_argument("--graph", action="store_true", help="show persistent task graph instead of short todos")

    team_parser = subparsers.add_parser("team", help="show persistent teammate roster")
    team_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .agent state")

    worktree_parser = subparsers.add_parser("worktrees", help="show managed task worktrees")
    worktree_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .agent state")

    ue_parser = subparsers.add_parser("ue", help="Unreal Engine helper commands")
    ue_subparsers = ue_parser.add_subparsers(dest="ue_command", required=True)

    ue_doctor = ue_subparsers.add_parser("doctor", help="find .uproject and Unreal Editor executables")
    ue_doctor.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")

    ue_build = ue_subparsers.add_parser("build", help="compile the UE Editor target with Build.bat")
    ue_build.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")
    ue_build.add_argument("--timeout", type=int, default=1800, help="UE build timeout in seconds")

    ue_run = ue_subparsers.add_parser("run-python", help="prepare or execute a UE Python script")
    ue_run.add_argument("script", help="path to a Python script to run inside UE")
    ue_run.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")
    ue_run.add_argument("--mode", choices=["commandlet", "full_editor"], default="full_editor", help="UE Python execution mode")
    ue_run.add_argument("--execute", action="store_true", help="actually launch UE; omitted means dry-run only")
    ue_run.add_argument("--timeout", type=int, default=300, help="UE process timeout in seconds")

    ue_list = ue_subparsers.add_parser("list-assets", help="prepare or execute a /Game asset listing script")
    ue_list.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")
    ue_list.add_argument("--mode", choices=["commandlet", "full_editor"], default="commandlet", help="UE Python execution mode")
    ue_list.add_argument("--execute", action="store_true", help="actually launch UE; omitted means dry-run only")
    ue_list.add_argument("--timeout", type=int, default=300, help="UE process timeout in seconds")

    ue_validate = ue_subparsers.add_parser("validate-assets", help="prepare or execute UE Data Validation")
    ue_validate.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")
    ue_validate.add_argument("--mode", choices=["commandlet", "full_editor"], default="commandlet", help="UE Python execution mode")
    ue_validate.add_argument("--execute", action="store_true", help="actually launch UE; omitted means dry-run only")
    ue_validate.add_argument("--timeout", type=int, default=300, help="UE process timeout in seconds")

    return parser


def main() -> None:
    try:
        parser = build_parser()
        args = parser.parse_args()

        if args.command == "init":
            init_config(Path.cwd().resolve(), force=args.force)
        elif args.command == "doctor":
            doctor()
        elif args.command == "run":
            max_steps = _resolve_max_steps(args.max_steps)
            run_agent(
                AgentOptions(
                    task=" ".join(args.task),
                    max_steps=max_steps,
                    auto_approve=args.yes,
                    cwd=Path(args.cwd).resolve(),
                    timeout_seconds=args.timeout,
                    verbose=args.verbose,
                    context_threshold=args.context_threshold,
                )
            )
        elif args.command == "chat":
            max_steps = _resolve_max_steps(args.max_steps)
            run_chat(
                AgentOptions(
                    task="",
                    max_steps=max_steps,
                    auto_approve=args.yes,
                    cwd=Path(args.cwd).resolve(),
                    timeout_seconds=args.timeout,
                    verbose=args.verbose,
                    context_threshold=args.context_threshold,
                    plain=args.plain,
                )
            )
        elif args.command == "tasks":
            cwd = Path(args.cwd).resolve()
            state_dir = agent_dir(cwd)
            if args.graph:
                print(TaskManager(state_dir / "tasks").list_all())
            else:
                print(TodoManager(state_dir).render_current())
        elif args.command == "team":
            cwd = Path(args.cwd).resolve()
            state_dir = agent_dir(cwd)
            task_manager = TaskManager(state_dir / "tasks")
            bus = MessageBus(state_dir / "team")
            print(TeamManager(state_dir / "team", task_manager, bus).list_all())
        elif args.command == "worktrees":
            cwd = Path(args.cwd).resolve()
            state_dir = agent_dir(cwd)
            task_manager = TaskManager(state_dir / "tasks")
            print(WorktreeManager(cwd, state_dir / "worktrees", task_manager).list_all())
        elif args.command == "ue":
            handle_ue(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


def init_config(cwd: Path, force: bool = False) -> None:
    system_path = default_system_config_path()
    project_path = project_config_path(cwd)

    if system_path.exists() and not force:
        print(f"System config already exists: {system_path}")
    else:
        write_json(system_path, system_config_template())
        print(f"Created system config: {system_path}")

    if project_path.exists() and not force:
        print(f"Project config already exists: {project_path}")
    else:
        write_json(project_path, project_config_template())
        print(f"Created project config: {project_path}")


def _resolve_max_steps(cli_value: int | None) -> int:
    if cli_value is not None:
        if cli_value <= 0:
            raise ConfigError("--max-steps must be a positive integer")
        return cli_value
    return load_system_config().runtime_default_max_steps


def doctor() -> None:
    cwd = Path.cwd().resolve()
    print(f"Version: {__version__}")
    print(f"Working directory: {cwd}")
    print(f"Shell: {shell_name()}")
    print(f"System config: {default_system_config_path()}")
    print(f"Project config: {project_config_path(cwd)}")
    print(f"Permission mode: {load_project_config(cwd).permission_mode.replace('_', '-')}")

    try:
        config = load_system_config()
        active = active_model_name(cwd, config)
        profile = resolve_model_profile(cwd, config)
    except ConfigError as error:
        print(f"Config error: {error}")
        return

    print(f"Active model profile: {active}")
    print(f"Model: {profile.model or '(missing)'}")
    print(f"API mode: {'Responses' if profile.gpt_model else 'Chat Completions'}")
    print(f"Base URL: {profile.base_url}")
    print(f"Timeout: {profile.timeout_seconds}s")
    print(f"API key: {'set' if profile.api_key else '(missing)'}")
    if not config.ue_engines:
        print("UE engines: (none)")
        return

    print("UE engines:")
    for name, engine in sorted(config.ue_engines.items()):
        alias_text = f" aliases={list(engine.aliases)}" if engine.aliases else ""
        print(f"- {name}: {engine.root}{alias_text}")


def handle_ue(args: argparse.Namespace) -> None:
    cwd = Path(args.cwd).resolve()
    state_dir = agent_dir(cwd)
    if args.ue_command == "doctor":
        print(render_doctor(discover_ue(cwd)))
        return

    if args.ue_command == "build":
        print(render_build_result(run_ue_build(cwd, state_dir, timeout_seconds=args.timeout)))
        return

    if args.ue_command == "run-python":
        script_path = Path(args.script).resolve()
        script = script_path.read_text(encoding="utf-8")
        result = run_ue_python(
            cwd=cwd,
            agent_dir=state_dir,
            script=script,
            mode=args.mode,
            kind="custom",
            execute=args.execute,
            timeout_seconds=args.timeout,
            source_script_path=script_path,
        )
        print(render_run_result(result))
        return

    if args.ue_command in {"list-assets", "validate-assets"}:
        result = run_ue_python(
            cwd=cwd,
            agent_dir=state_dir,
            script="",
            mode=args.mode,
            kind="list_assets" if args.ue_command == "list-assets" else "validate_assets",
            execute=args.execute,
            timeout_seconds=args.timeout,
        )
        print(render_run_result(result))
        return

    raise ValueError(f"Unknown UE command: {args.ue_command}")
