from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .background import BackgroundManager
from .context import compact_locally, estimate_tokens, micro_compact
from .llm import ChatMessage, call_model
from .protocol import ActionParseError, FinalAction, ShellAction, ToolAction, parse_agent_action
from .shell import confirm_command, run_shell, shell_name
from .skills import SkillLoader
from .tasks import TaskManager, TodoManager
from .team import MessageBus, TeamManager
from .ue import discover_ue, render_doctor, render_run_result, run_ue_python
from .workspace import edit_file, list_files, read_file, write_file
from .worktrees import WorktreeManager


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool
    allow_ue_execute: bool = False
    context_threshold: int = 60000


ToolHandler = Callable[[dict[str, object]], str]


def build_system_prompt(cwd: Path, skills: SkillLoader) -> str:
    return f"""You are a UE development agent running inside a command-line harness.

Architecture rule: the model supplies agency; the harness supplies tools, observation, permissions, context, tasks, team coordination, and worktree isolation.
Respond with exactly one JSON object only. Do not include markdown, prose, arrays, or extra keys.

Action forms:
{{"type":"tool","name":"tool_name","input":{{...}}}}
{{"type":"shell","command":"...","reason":"..."}}  // legacy alias for tool=shell
{{"type":"final","answer":"..."}}

Core tools by lesson:
- s01/s02 loop and tools: shell, read_file, write_file, edit_file, list_files
- s03 planning: todo_update, todo_list
- s04 context isolation: subagent
- s05 on-demand knowledge: load_skill
- s06 context management: compact
- s07 persistent task graph: task_create, task_get, task_update, task_list, claim_task
- s08 background execution: background_run, background_check
- s09 team mailbox: spawn_teammate, list_teammates, send_message, read_inbox, broadcast
- s10 protocols: shutdown_request, shutdown_response, plan_submit, plan_review
- s11 autonomy: idle, claim_task
- s12 worktree isolation: worktree_create, worktree_list, worktree_run, worktree_keep, worktree_remove
- UE tools: ue_doctor, ue_run_python

UE safety:
- Always call ue_doctor before UE editor operations.
- ue_run_python defaults to dry-run. Set execute=true only when the user explicitly asks to launch UE and the CLI allows it.
- Prefer commandlet mode for read-only automation. Use full_editor only when an API needs the editor.

Available skills:
{skills.descriptions()}

Working directory: {cwd}
Shell: {shell_name()}"""


def run_agent(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Task: {options.task}"),
    ]
    final = runtime.run_turn(messages, goal=options.task)
    if final:
        print(f"\n{final}")


def run_chat(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]

    print("myagent chat. Type /help for commands; q, exit, or Ctrl+C to quit.")
    while True:
        query = input("\nmyagent >> ").strip()
        if query.lower() in {"", "q", "quit", "exit"}:
            return

        if query.lower() == "/clear":
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
            ]
            print("Conversation context cleared.")
            continue

        if runtime.handle_slash_command(query):
            continue

        messages.append(ChatMessage(role="user", content=query))
        final = runtime.run_turn(messages, goal=query)
        if final:
            print(f"\n{final}")


class AgentRuntime:
    def __init__(self, options: AgentOptions):
        self.options = options
        self.agent_dir = options.cwd / ".agent"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.todo_manager = TodoManager(self.agent_dir)
        self.task_manager = TaskManager(options.cwd / ".tasks")
        self.skill_loader = SkillLoader(options.cwd / "skills")
        self.background = BackgroundManager(options.cwd)
        self.bus = MessageBus(options.cwd / ".team")
        self.team = TeamManager(options.cwd / ".team", self.task_manager, self.bus)
        self.worktrees = WorktreeManager(options.cwd, options.cwd / ".worktrees", self.task_manager)
        self.system_prompt = build_system_prompt(options.cwd, self.skill_loader)
        self.tools = self._build_tool_handlers()

    def run_turn(self, messages: list[ChatMessage], goal: str) -> str | None:
        work_tool_used = False
        rounds_without_todo = 0

        for step in range(1, self.options.max_steps + 1):
            self._inject_runtime_observations(messages)
            print(f"\nThinking..." if not self.options.verbose else f"\nThinking... ({step}/{self.options.max_steps})")

            raw = call_model(messages)
            messages.append(ChatMessage(role="assistant", content=raw))
            try:
                action = parse_agent_action(raw)
            except ActionParseError as error:
                self._append_protocol_correction(messages, error)
                continue

            if isinstance(action, FinalAction):
                if requires_tool_action(goal) and not work_tool_used:
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "The request requires harness work, but you returned final without using a work tool. "
                                "Request one tool action now."
                            ),
                        )
                    )
                    continue
                return action.answer

            if isinstance(action, ShellAction):
                action = ToolAction(type="tool", name="shell", input={"command": action.command, "reason": action.reason})

            if isinstance(action, ToolAction):
                output = self._execute_tool(action)
                if action.name == "compact":
                    messages[:] = compact_locally(messages, self.agent_dir / "transcripts", "manual compact tool")
                    messages.append(ChatMessage(role="user", content=f"Tool result for: compact\n{output}"))
                    continue

                if action.name == "todo_update":
                    rounds_without_todo = 0
                else:
                    rounds_without_todo += 1

                if action.name not in {"todo_update", "todo_list", "compact", "read_inbox", "list_teammates", "task_list"}:
                    work_tool_used = True

                tool_content = f"Tool result for: {action.name}\n{truncate(output)}"
                if self.todo_manager.has_open_items() and rounds_without_todo >= 3:
                    tool_content += "\n<reminder>Update your todos before continuing.</reminder>"
                    rounds_without_todo = 0
                messages.append(ChatMessage(role="user", content=tool_content))

        print(f"\nStopped after {self.options.max_steps} iterations without a final answer.")
        return None

    def handle_slash_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False

        command = query.lower().strip()
        if command == "/help":
            print("/help  /todos  /tasks  /team  /inbox  /ue doctor  /compact  /clear")
            return True
        if command == "/todos":
            print(self.todo_manager.render_current())
            return True
        if command == "/tasks":
            print(self.task_manager.list_all())
            return True
        if command == "/team":
            print(self.team.list_all())
            return True
        if command == "/inbox":
            print(json.dumps(self.bus.read_inbox("lead"), ensure_ascii=False, indent=2))
            return True
        if command == "/ue doctor":
            print(render_doctor(discover_ue(self.options.cwd)))
            return True
        if command == "/compact":
            print("Use the compact tool during an agent turn, or /clear to reset chat context.")
            return True

        print(f"Unknown slash command: {query}")
        return True

    def _inject_runtime_observations(self, messages: list[ChatMessage]) -> None:
        micro_compact(messages)
        if estimate_tokens(messages) > self.options.context_threshold:
            messages[:] = compact_locally(messages, self.agent_dir / "transcripts", "automatic threshold")

        notifications = self.background.drain()
        if notifications:
            rendered = "\n".join(f"[bg:{task.id}] {task.status}\n{truncate(task.result, 1000)}" for task in notifications)
            messages.append(ChatMessage(role="user", content=f"<background-results>\n{rendered}\n</background-results>"))

        inbox = self.bus.read_inbox("lead")
        if inbox:
            messages.append(ChatMessage(role="user", content=f"<inbox>{json.dumps(inbox, ensure_ascii=False, indent=2)}</inbox>"))

    def _append_protocol_correction(self, messages: list[ChatMessage], error: ActionParseError) -> None:
        if self.options.verbose:
            print(f"\nProtocol correction: {error}")
            print(f"Model response: {truncate(error.raw, 1000)}")
        else:
            print("\nRetrying with a stricter action format...")
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Your previous response did not match the required action protocol. "
                    "Return exactly one JSON object, for example:\n"
                    '{"type":"tool","name":"read_file","input":{"path":"README.md"}}\n'
                    '{"type":"final","answer":"..."}'
                ),
            )
        )

    def _execute_tool(self, action: ToolAction) -> str:
        handler = self.tools.get(action.name)
        if handler is None:
            return f"Unknown tool: {action.name}"
        try:
            return handler(action.input)
        except Exception as error:
            return f"Tool {action.name} failed: {error}"

    def _build_tool_handlers(self) -> dict[str, ToolHandler]:
        def shell_tool(tool_input: dict[str, object]) -> str:
            command = str(tool_input.get("command", "")).strip()
            reason = str(tool_input.get("reason", "")).strip() or "agent requested shell execution"
            if not command:
                raise ValueError("shell command cannot be empty")

            print(f"\n$ {command}")
            if self.options.verbose:
                print(f"Reason: {reason}")
            approved = self.options.auto_approve or confirm_command(command, reason)
            if not approved:
                return "The user rejected that command. Choose a safer command or finish."

            result = run_shell(command, self.options.cwd, self.options.timeout_seconds)
            return "\n".join(
                [
                    f"command: {result.command}",
                    f"exitCode: {result.exit_code}",
                    "stdout:",
                    result.stdout,
                    "stderr:",
                    result.stderr,
                ]
            )

        def background_run_tool(tool_input: dict[str, object]) -> str:
            command = str(tool_input.get("command", "")).strip()
            timeout = int(tool_input.get("timeout_seconds") or tool_input.get("timeout") or self.options.timeout_seconds)
            reason = str(tool_input.get("reason") or "agent requested background command")
            if not (self.options.auto_approve or confirm_command(command, reason)):
                return "The user rejected that background command."
            return self.background.run(command, timeout)

        def ue_run_python_tool(tool_input: dict[str, object]) -> str:
            requested_execute = bool(tool_input.get("execute", False))
            execute = requested_execute and self.options.allow_ue_execute
            if requested_execute and not self.options.allow_ue_execute:
                print("\nUE execute requested, but this session is not allowed to launch UE; returning dry-run command.")
            result = run_ue_python(
                cwd=self.options.cwd,
                agent_dir=self.agent_dir,
                script=str(tool_input.get("script", "")),
                mode=str(tool_input.get("mode", "commandlet")),
                kind=str(tool_input.get("kind", "custom")),
                execute=execute,
                timeout_seconds=self.options.timeout_seconds,
            )
            return render_run_result(result)

        def subagent_tool(tool_input: dict[str, object]) -> str:
            prompt = str(tool_input.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("subagent requires prompt")
            return self._run_subagent(prompt, str(tool_input.get("agent_type", "explore")))

        return {
            "shell": shell_tool,
            "read_file": lambda data: read_file(self.options.cwd, str(data.get("path", "")), _optional_int(data.get("limit"))),
            "write_file": lambda data: write_file(self.options.cwd, str(data.get("path", "")), str(data.get("content", ""))),
            "edit_file": lambda data: edit_file(
                self.options.cwd,
                str(data.get("path", "")),
                str(data.get("old_text", "")),
                str(data.get("new_text", "")),
            ),
            "list_files": lambda data: list_files(
                self.options.cwd,
                str(data.get("path", ".")),
                str(data.get("pattern", "*")),
                int(data.get("limit", 200)),
            ),
            "todo_update": lambda data: self.todo_manager.update(_require_list_of_dicts(data.get("items"), "items")),
            "todo_list": lambda data: self.todo_manager.render_current(),
            "subagent": subagent_tool,
            "load_skill": lambda data: self.skill_loader.load(str(data.get("name", ""))),
            "compact": lambda data: "Conversation compacted; full transcript saved.",
            "task_create": lambda data: self.task_manager.create(
                str(data.get("subject", "")),
                str(data.get("description", "")),
                _optional_int_list(data.get("blockedBy") or data.get("blocked_by")),
                _optional_str(data.get("owner")),
            ),
            "task_get": lambda data: self.task_manager.get(int(data.get("task_id", 0))),
            "task_update": lambda data: self.task_manager.update(
                int(data.get("task_id", 0)),
                _optional_str(data.get("status")),
                _optional_str(data.get("owner")),
                _optional_int_list(data.get("add_blocked_by")),
                _optional_int_list(data.get("remove_blocked_by")),
                _optional_str(data.get("worktree")),
            ),
            "task_list": lambda data: self.task_manager.list_all(),
            "claim_task": lambda data: self.task_manager.claim(int(data.get("task_id", 0)), str(data.get("owner", "lead"))),
            "background_run": background_run_tool,
            "background_check": lambda data: self.background.check(_optional_str(data.get("task_id"))),
            "spawn_teammate": lambda data: self.team.spawn(
                str(data.get("name", "")),
                str(data.get("role", "")),
                str(data.get("prompt", "")),
            ),
            "list_teammates": lambda data: self.team.list_all(),
            "send_message": lambda data: self.bus.send(
                "lead",
                str(data.get("to", "")),
                str(data.get("content", "")),
                str(data.get("msg_type", "message")),
            ),
            "read_inbox": lambda data: json.dumps(self.bus.read_inbox("lead"), ensure_ascii=False, indent=2),
            "broadcast": lambda data: self.team.broadcast("lead", str(data.get("content", ""))),
            "shutdown_request": lambda data: self.team.shutdown_request(str(data.get("teammate", ""))),
            "shutdown_response": lambda data: self.team.shutdown_response(
                str(data.get("request_id", "")),
                bool(data.get("approve", False)),
                str(data.get("reason", "")),
            ),
            "plan_submit": lambda data: self.team.plan_submit(str(data.get("teammate", "lead")), str(data.get("plan", ""))),
            "plan_review": lambda data: self.team.plan_review(
                str(data.get("request_id", "")),
                bool(data.get("approve", False)),
                str(data.get("feedback", "")),
            ),
            "idle": lambda data: self.team.idle(str(data.get("teammate", "lead"))),
            "worktree_create": lambda data: self.worktrees.create(
                str(data.get("name", "")),
                _optional_int(data.get("task_id")),
                str(data.get("base_ref", "HEAD")),
            ),
            "worktree_list": lambda data: self.worktrees.list_all(),
            "worktree_run": lambda data: self.worktrees.run(
                str(data.get("name", "")),
                str(data.get("command", "")),
                int(data.get("timeout_seconds", self.options.timeout_seconds)),
            ),
            "worktree_keep": lambda data: self.worktrees.keep(str(data.get("name", ""))),
            "worktree_remove": lambda data: self.worktrees.remove(
                str(data.get("name", "")),
                bool(data.get("force", False)),
                bool(data.get("complete_task", False)),
            ),
            "ue_doctor": lambda data: render_doctor(discover_ue(self.options.cwd)),
            "ue_run_python": ue_run_python_tool,
        }

    def _run_subagent(self, prompt: str, agent_type: str) -> str:
        child_messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a focused subagent with clean context. "
                    "Return JSON actions. Use only read_file, list_files, shell, or final unless asked to write."
                ),
            ),
            ChatMessage(role="user", content=prompt),
        ]
        allowed = {"read_file", "list_files", "shell"}
        if agent_type not in {"explore", "Explore"}:
            allowed.update({"write_file", "edit_file"})

        summaries: list[str] = []
        for _ in range(min(6, self.options.max_steps)):
            raw = call_model(child_messages)
            child_messages.append(ChatMessage(role="assistant", content=raw))
            action = parse_agent_action(raw)
            if isinstance(action, FinalAction):
                return action.answer
            if isinstance(action, ShellAction):
                action = ToolAction(type="tool", name="shell", input={"command": action.command, "reason": action.reason})
            if isinstance(action, ToolAction):
                if action.name not in allowed:
                    output = f"Subagent tool not allowed: {action.name}"
                else:
                    output = self._execute_tool(action)
                summaries.append(f"{action.name}: {truncate(output, 1000)}")
                child_messages.append(ChatMessage(role="user", content=f"Tool result for: {action.name}\n{truncate(output)}"))
        return "Subagent stopped after bounded steps.\n" + "\n".join(summaries[-5:])


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


def _require_list_of_dicts(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected a list of integers")
    return [int(item) for item in value]
