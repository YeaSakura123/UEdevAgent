from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .loop import AgentOptions, run_agent, run_chat
from .shell import shell_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myagent",
        description="Agentic CLI for coding and automation workflows.",
    )
    parser.add_argument("--version", action="version", version=f"myagent {__version__}")

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
    run_parser.add_argument("--verbose", action="store_true", help="show internal iteration and protocol diagnostics")

    chat_parser = subparsers.add_parser("chat", help="start an interactive agent chat session")
    chat_parser.add_argument("--max-iterations", "--max-steps", dest="max_steps", type=int, default=8, help=argparse.SUPPRESS)
    chat_parser.add_argument("--timeout", type=int, default=120, help="shell command timeout in seconds")
    chat_parser.add_argument("-y", "--yes", action="store_true", help="execute shell commands without asking")
    chat_parser.add_argument("--cwd", default=str(Path.cwd()), help="working directory for shell commands")
    chat_parser.add_argument("--verbose", action="store_true", help="show internal iteration and protocol diagnostics")

    return parser


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
                )
            )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


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
                "",
            ]
        )

    env_path.write_text(content, encoding="utf-8")
    print("Created .env. Fill in OPENAI_API_KEY and OPENAI_MODEL before running the agent.")


def doctor() -> None:
    print(f"Version: {__version__}")
    print(f"Working directory: {Path.cwd().resolve()}")
    print(f"Shell: {shell_name()}")
    print(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"OPENAI_MODEL: {os.environ.get('OPENAI_MODEL') or '(missing)'}")
    print(f"OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else '(missing)'}")
