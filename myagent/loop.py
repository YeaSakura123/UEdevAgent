from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .llm import ChatMessage, call_model
from .protocol import ActionParseError, FinalAction, ShellAction, parse_agent_action
from .shell import confirm_command, run_shell, shell_name


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool


SYSTEM_PROMPT = """You are a coding agent running in a command-line environment.

You can either request exactly one shell command or finish with a final answer.
Respond with exactly one JSON object only. Do not include markdown, prose, arrays, or extra keys.

Available actions:
{"type":"shell","command":"...","reason":"..."}
{"type":"final","answer":"..."}

Rules:
- Act, don't explain. If the user asks you to inspect, create, edit, delete, move, run, test, or verify something in the filesystem, request a shell action.
- Do not say you cannot access the filesystem. You can request shell commands and the CLI will ask the user for approval.
- Use short, inspectable commands.
- Commands run in the current working directory using the shell named in the user context.
- Prefer read-only inspection before edits.
- After creating or editing files, verify the result before returning final.
- Never run destructive commands such as deleting files, resetting git, or formatting disks.
- After shell output is returned, decide the next step.
- If the task is complete, return final."""


def run_agent(options: AgentOptions) -> None:
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Task: {options.task}\nWorking directory: {options.cwd}\nShell: {shell_name()}",
        ),
    ]

    final = run_turn(
        messages=messages,
        max_steps=options.max_steps,
        auto_approve=options.auto_approve,
        cwd=options.cwd,
        timeout_seconds=options.timeout_seconds,
        goal=options.task,
        verbose=options.verbose,
    )
    if final:
        print(f"\n{final}")


def run_chat(options: AgentOptions) -> None:
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]

    print("myagent chat. Type q, exit, or Ctrl+C to quit.")
    while True:
        query = input("\nmyagent >> ").strip()
        if query.lower() in {"", "q", "quit", "exit"}:
            return

        messages.append(ChatMessage(role="user", content=query))
        final = run_turn(
            messages=messages,
            max_steps=options.max_steps,
            auto_approve=options.auto_approve,
            cwd=options.cwd,
            timeout_seconds=options.timeout_seconds,
            goal=query,
            verbose=options.verbose,
        )
        if final:
            print(f"\n{final}")


def run_turn(
    messages: list[ChatMessage],
    max_steps: int,
    auto_approve: bool,
    cwd: Path,
    timeout_seconds: int,
    goal: str,
    verbose: bool,
) -> str | None:
    tool_used = False

    for step in range(1, max_steps + 1):
        print(f"\nThinking..." if not verbose else f"\nThinking... ({step}/{max_steps})")

        raw = call_model(messages)
        messages.append(ChatMessage(role="assistant", content=raw))
        try:
            action = parse_agent_action(raw)
        except ActionParseError as error:
            if verbose:
                print(f"\nProtocol correction: {error}")
                print(f"Model response: {truncate(error.raw, 1000)}")
            else:
                print("\nRetrying with a stricter action format...")
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous response did not match the required action protocol. "
                        "Return exactly one JSON object in one of these forms, with no markdown or prose:\n"
                        '{"type":"shell","command":"...","reason":"..."}\n'
                        '{"type":"final","answer":"..."}'
                    ),
                )
            )
            continue

        if isinstance(action, FinalAction):
            if requires_shell_action(goal) and not tool_used:
                if verbose:
                    print("\nRefusing premature final answer; task appears to require a shell action.")
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "The user's request requires filesystem or command-line work, "
                            "but you returned final without using the shell. Request a shell action now."
                        ),
                    )
                )
                continue
            return action.answer

        if isinstance(action, ShellAction):
            print(f"\n$ {action.command}")
            if verbose:
                print(f"Reason: {action.reason}")

            approved = auto_approve or confirm_command(action.command, action.reason)
            if not approved:
                messages.append(
                    ChatMessage(
                        role="user",
                        content="The user rejected that command. Choose a safer command or finish.",
                    )
                )
                continue

            result = run_shell(action.command, cwd, timeout_seconds)
            tool_used = True
            messages.append(
                ChatMessage(
                    role="user",
                    content="\n".join(
                        [
                            f"Shell result for: {result.command}",
                            f"exitCode: {result.exit_code}",
                            "stdout:",
                            truncate(result.stdout),
                            "stderr:",
                            truncate(result.stderr),
                        ]
                    ),
                )
            )

    print(f"\nStopped after {max_steps} iterations without a final answer.")
    return None


def truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value

    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


def requires_shell_action(goal: str) -> bool:
    normalized = goal.lower()
    keywords = [
        "创建",
        "新建",
        "写入",
        "修改",
        "编辑",
        "删除",
        "移动",
        "复制",
        "运行",
        "执行",
        "测试",
        "检查",
        "验证",
        "生成",
        "create",
        "write",
        "edit",
        "modify",
        "delete",
        "move",
        "copy",
        "run",
        "execute",
        "test",
        "check",
        "verify",
        "generate",
        "install",
    ]
    return any(keyword in normalized for keyword in keywords)
