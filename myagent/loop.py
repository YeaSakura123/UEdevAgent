from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .llm import ChatMessage, call_model
from .protocol import ActionParseError, FinalAction, ShellAction, ToolAction, parse_agent_action
from .shell import confirm_command, run_shell, shell_name
from .tasks import TodoManager
from .ue import discover_ue, render_doctor, render_run_result, run_ue_python


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool
    allow_ue_execute: bool = False


SYSTEM_PROMPT = """You are a coding agent running in a command-line environment.

You can either request exactly one tool action or finish with a final answer.
Respond with exactly one JSON object only. Do not include markdown, prose, arrays, or extra keys.

Available actions:
{"type":"tool","name":"shell","input":{"command":"...","reason":"..."}}
{"type":"tool","name":"todo_update","input":{"items":[{"id":"1","text":"...","status":"pending|in_progress|completed"}]}}
{"type":"tool","name":"todo_list","input":{}}
{"type":"tool","name":"ue_doctor","input":{}}
{"type":"tool","name":"ue_run_python","input":{"script":"...","mode":"commandlet|full_editor","kind":"custom|list_assets|validate_assets","reason":"...","execute":false}}
{"type":"shell","command":"...","reason":"..."}
{"type":"final","answer":"..."}

Rules:
- Act, don't explain. If the user asks you to inspect, create, edit, delete, move, run, test, or verify something in the filesystem, request a shell action.
- Do not say you cannot access the filesystem. You can request shell commands and the CLI will ask the user for approval.
- Use todo_update before and during multi-step tasks. Keep exactly one item in_progress.
- Use ue_doctor before UE operations. UE Python defaults to dry run; only set execute=true when the user explicitly asks to launch UE.
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
        allow_ue_execute=options.allow_ue_execute,
    )
    if final:
        print(f"\n{final}")


def run_chat(options: AgentOptions) -> None:
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]

    print("myagent chat. Type /help for commands; q, exit, or Ctrl+C to quit.")
    while True:
        query = input("\nmyagent >> ").strip()
        if query.lower() in {"", "q", "quit", "exit"}:
            return

        if handle_slash_command(query, options.cwd):
            continue

        messages.append(ChatMessage(role="user", content=query))
        final = run_turn(
            messages=messages,
            max_steps=options.max_steps,
            auto_approve=options.auto_approve,
            cwd=options.cwd,
            timeout_seconds=options.timeout_seconds,
            goal=query,
            verbose=options.verbose,
            allow_ue_execute=options.allow_ue_execute,
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
    allow_ue_execute: bool,
) -> str | None:
    tool_used = False
    todo_manager = TodoManager(cwd / ".agent")
    tools = build_tool_handlers(
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        auto_approve=auto_approve,
        allow_ue_execute=allow_ue_execute,
        todo_manager=todo_manager,
        verbose=verbose,
    )

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
                        '{"type":"tool","name":"shell","input":{"command":"...","reason":"..."}}\n'
                        '{"type":"final","answer":"..."}'
                    ),
                )
            )
            continue

        if isinstance(action, FinalAction):
            if requires_tool_action(goal) and not tool_used:
                if verbose:
                    print("\nRefusing premature final answer; task appears to require a tool action.")
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "The user's request requires filesystem, command-line, todo, or UE work, "
                            "but you returned final without using a tool. Request a tool action now."
                        ),
                    )
                )
                continue
            return action.answer

        if isinstance(action, ShellAction):
            # 兼容早期协议，避免旧模型提示还没更新时直接失效。
            action = ToolAction(
                type="tool",
                name="shell",
                input={"command": action.command, "reason": action.reason},
            )

        if isinstance(action, ToolAction):
            handler = tools.get(action.name)
            if not handler:
                messages.append(ChatMessage(role="user", content=f"Unknown tool: {action.name}"))
                continue

            try:
                output = handler(action.input)
            except Exception as error:
                output = f"Tool {action.name} failed: {error}"

            tool_used = True
            messages.append(
                ChatMessage(
                    role="user",
                    content="\n".join([f"Tool result for: {action.name}", truncate(output)]),
                )
            )

    print(f"\nStopped after {max_steps} iterations without a final answer.")
    return None


def handle_slash_command(query: str, cwd: Path) -> bool:
    if not query.startswith("/"):
        return False

    command = query.lower().strip()
    todo_manager = TodoManager(cwd / ".agent")
    if command == "/help":
        print("/help  /todos  /ue doctor  /clear")
        return True
    if command == "/todos":
        print(todo_manager.render_current())
        return True
    if command == "/ue doctor":
        print(render_doctor(discover_ue(cwd)))
        return True
    if command == "/clear":
        print("Conversation context will start fresh on your next prompt.")
        return True

    print(f"Unknown slash command: {query}")
    return True


def truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value

    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


def requires_tool_action(goal: str) -> bool:
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
        "ue",
        "unreal",
        "asset",
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


ToolHandler = Callable[[dict[str, object]], str]


def build_tool_handlers(
    *,
    cwd: Path,
    timeout_seconds: int,
    auto_approve: bool,
    allow_ue_execute: bool,
    todo_manager: TodoManager,
    verbose: bool,
) -> dict[str, ToolHandler]:
    def shell_tool(tool_input: dict[str, object]) -> str:
        command = str(tool_input.get("command", "")).strip()
        reason = str(tool_input.get("reason", "")).strip() or "agent requested shell execution"
        if not command:
            raise ValueError("shell command cannot be empty")

        print(f"\n$ {command}")
        if verbose:
            print(f"Reason: {reason}")

        approved = auto_approve or confirm_command(command, reason)
        if not approved:
            return "The user rejected that command. Choose a safer command or finish."

        result = run_shell(command, cwd, timeout_seconds)
        return "\n".join(
            [
                f"command: {result.command}",
                f"exitCode: {result.exit_code}",
                "stdout:",
                truncate(result.stdout),
                "stderr:",
                truncate(result.stderr),
            ]
        )

    def todo_update_tool(tool_input: dict[str, object]) -> str:
        items = tool_input.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError("todo_update requires input.items as a list of objects")
        return todo_manager.update(items)

    def ue_doctor_tool(_: dict[str, object]) -> str:
        return render_doctor(discover_ue(cwd))

    def ue_run_python_tool(tool_input: dict[str, object]) -> str:
        script = str(tool_input.get("script", ""))
        mode = str(tool_input.get("mode", "commandlet"))
        kind = str(tool_input.get("kind", "custom"))
        requested_execute = bool(tool_input.get("execute", False))
        execute = requested_execute and allow_ue_execute
        if requested_execute and not allow_ue_execute:
            print("\nUE execute requested, but this session is not allowed to launch UE; returning dry-run command.")

        result = run_ue_python(
            cwd=cwd,
            agent_dir=cwd / ".agent",
            script=script,
            mode=mode,
            kind=kind,
            execute=execute,
            timeout_seconds=timeout_seconds,
        )
        return render_run_result(result)

    return {
        "shell": shell_tool,
        "todo_update": todo_update_tool,
        "todo_list": lambda _: todo_manager.render_current(),
        "ue_doctor": ue_doctor_tool,
        "ue_run_python": ue_run_python_tool,
    }
