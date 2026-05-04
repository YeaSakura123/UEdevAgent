from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .loop import AgentOptions, run_agent, run_chat
from .shell import shell_name
from .tasks import TaskManager, TodoManager
from .team import MessageBus, TeamManager
from .ue import discover_ue, render_doctor, render_run_result, run_ue_python
from .worktrees import WorktreeManager


# 外部函数：构建 CLI 参数和子命令界面，负责 uedev 命令行界面、参数解析和子命令分发。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uedev",
        description="Agentic CLI for coding and automation workflows.",
    )
    parser.add_argument("--version", action="version", version=f"uedev {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create a local .env configuration file")
    init_parser.add_argument("--force", action="store_true", help="overwrite an existing .env file")

    subparsers.add_parser("doctor", help="check local configuration")

    run_parser = subparsers.add_parser("run", help="run one agent task")
    run_parser.add_argument("task", nargs="+", help="task for the agent")
    run_parser.add_argument("--max-iterations", "--max-steps", dest="max_steps", type=int, default=8, help=argparse.SUPPRESS)
    run_parser.add_argument("--timeout", type=int, default=120, help="shell command timeout in seconds")
    run_parser.add_argument("-y", "--yes", action="store_true", help="execute shell commands without asking")
    run_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory for shell commands")
    run_parser.add_argument("--verbose", action="store_true", help="show internal iteration and tool diagnostics")
    run_parser.add_argument("--context-threshold", type=int, default=60000, help="estimated token threshold for auto compact")

    chat_parser = subparsers.add_parser("chat", help="start an interactive agent chat session")
    chat_parser.add_argument("--max-iterations", "--max-steps", dest="max_steps", type=int, default=8, help=argparse.SUPPRESS)
    chat_parser.add_argument("--timeout", type=int, default=120, help="shell command timeout in seconds")
    chat_parser.add_argument("-y", "--yes", action="store_true", help="execute shell commands without asking")
    chat_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory for shell commands")
    chat_parser.add_argument("--verbose", action="store_true", help="show internal iteration and tool diagnostics")
    chat_parser.add_argument("--context-threshold", type=int, default=60000, help="estimated token threshold for auto compact")
    chat_parser.add_argument("--plain", action="store_true", help="use the script-friendly plain chat renderer")

    task_parser = subparsers.add_parser("tasks", help="show the persisted agent todo list")
    task_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .agent state")
    task_parser.add_argument("--graph", action="store_true", help="show persistent task graph instead of short todos")

    team_parser = subparsers.add_parser("team", help="show persistent teammate roster")
    team_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .team state")

    worktree_parser = subparsers.add_parser("worktrees", help="show managed task worktrees")
    worktree_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory that contains .worktrees state")

    ue_parser = subparsers.add_parser("ue", help="Unreal Engine helper commands")
    ue_subparsers = ue_parser.add_subparsers(dest="ue_command", required=True)

    ue_doctor = ue_subparsers.add_parser("doctor", help="find .uproject and Unreal Editor executables")
    ue_doctor.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")

    ue_run = ue_subparsers.add_parser("run-python", help="prepare or execute a UE Python script")
    ue_run.add_argument("script", help="path to a Python script to run inside UE")
    ue_run.add_argument("--cwd", default=str(Path.cwd()), help="UE project or workspace directory")
    ue_run.add_argument("--mode", choices=["commandlet", "full_editor"], default="commandlet", help="UE Python execution mode")
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


# 外部函数：作为程序入口加载配置并分派命令，负责 uedev 命令行界面、参数解析和子命令分发。
def main() -> None:
    try:
        load_dotenv(Path(".env"))
        parser = build_parser()
        args = parser.parse_args()

        if args.command == "init":
            init_config(force=args.force)
        elif args.command == "doctor":
            doctor()
        elif args.command == "run":
            run_agent(
                AgentOptions(
                    task=" ".join(args.task),
                    max_steps=args.max_steps,
                    auto_approve=args.yes,
                    cwd=Path(args.cwd).resolve(),
                    timeout_seconds=args.timeout,
                    verbose=args.verbose,
                    context_threshold=args.context_threshold,
                    plain=args.plain,
                )
            )
        elif args.command == "chat":
            run_chat(
                AgentOptions(
                    task="",
                    max_steps=args.max_steps,
                    auto_approve=args.yes,
                    cwd=Path(args.cwd).resolve(),
                    timeout_seconds=args.timeout,
                    verbose=args.verbose,
                    context_threshold=args.context_threshold,
                )
            )
        elif args.command == "tasks":
            cwd = Path(args.cwd).resolve()
            if args.graph:
                print(TaskManager(cwd / ".tasks").list_all())
            else:
                print(TodoManager(cwd / ".agent").render_current())
        elif args.command == "team":
            cwd = Path(args.cwd).resolve()
            task_manager = TaskManager(cwd / ".tasks")
            bus = MessageBus(cwd / ".team")
            print(TeamManager(cwd / ".team", task_manager, bus).list_all())
        elif args.command == "worktrees":
            cwd = Path(args.cwd).resolve()
            task_manager = TaskManager(cwd / ".tasks")
            print(WorktreeManager(cwd, cwd / ".worktrees", task_manager).list_all())
        elif args.command == "ue":
            handle_ue(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


# 内部函数：读取 .env 文件并写入进程环境变量，支撑 CLI 启动流程。
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# 外部函数：实现 init 命令的配置文件创建功能，负责 uedev 命令行界面、参数解析和子命令分发。
def init_config(force: bool = False) -> None:
    env_path = Path(".env")
    if env_path.exists() and not force:
        print(".env already exists. Use --force to overwrite it.")
        return

    example_path = Path(".env.example")
    if example_path.exists():
        content = example_path.read_text(encoding="utf-8")
    else:
        content = "\n".join(
            [
                "OPENAI_API_KEY=",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_MODEL=",
                "UE_PROJECT_PATH=",
                "UE_ENGINE_ROOT=",
                "UE_EDITOR_CMD_PATH=",
                "UE_EDITOR_PATH=",
                "",
            ]
        )

    env_path.write_text(content, encoding="utf-8")
    print("Created .env. Fill in OPENAI_API_KEY and OPENAI_MODEL before running the agent.")


# 外部函数：实现 doctor 命令的环境检查展示，负责 uedev 命令行界面、参数解析和子命令分发。
def doctor() -> None:
    print(f"Version: {__version__}")
    print(f"Working directory: {Path.cwd().resolve()}")
    print(f"Shell: {shell_name()}")
    print(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"OPENAI_MODEL: {os.environ.get('OPENAI_MODEL') or '(missing)'}")
    print(f"OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else '(missing)'}")
    print(f"UE_PROJECT_PATH: {os.environ.get('UE_PROJECT_PATH') or '(auto)'}")
    print(f"UE_ENGINE_ROOT: {os.environ.get('UE_ENGINE_ROOT') or '(missing)'}")


# 外部函数：处理 ue 子命令的界面分发，负责 uedev 命令行界面、参数解析和子命令分发。
def handle_ue(args: argparse.Namespace) -> None:
    cwd = Path(args.cwd).resolve()
    if args.ue_command == "doctor":
        print(render_doctor(discover_ue(cwd)))
        return

    if args.ue_command == "run-python":
        script_path = Path(args.script).resolve()
        script = script_path.read_text(encoding="utf-8")
        result = run_ue_python(
            cwd=cwd,
            agent_dir=cwd / ".agent",
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
            agent_dir=cwd / ".agent",
            script="",
            mode=args.mode,
            kind="list_assets" if args.ue_command == "list-assets" else "validate_assets",
            execute=args.execute,
            timeout_seconds=args.timeout,
        )
        print(render_run_result(result))
        return

    raise ValueError(f"Unknown UE command: {args.ue_command}")
