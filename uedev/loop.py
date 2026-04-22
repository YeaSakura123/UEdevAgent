from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from . import __version__
from .background import BackgroundManager
from .context import compact_locally, estimate_tokens, micro_compact
from .llm import ChatMessage, call_model
from .protocol import ActionParseError, FinalAction, ShellAction, ToolAction, parse_agent_action
from .shell import confirm_command, run_shell, shell_name
from .skills import SkillLoader
from .tasks import TaskManager, TodoManager
from .team import MessageBus, TeamManager
from .tool_specs import get_tool_specs
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


SLASH_COMMANDS = [
    ("/help", "Show available chat slash commands."),
    ("/todos", "Show the current lightweight todo list."),
    ("/tasks", "Show the persistent task graph."),
    ("/team", "Show the persistent teammate roster."),
    ("/inbox", "Show pending messages for the lead agent."),
    ("/ue doctor", "Inspect Unreal Engine project and editor configuration."),
    ("/compact", "Explain how to compact or reset chat context."),
    ("/clear", "Reset the current chat conversation context."),
]


# 外部函数：生成 /help 命令展示的帮助文本，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def render_slash_help() -> str:
    width = max(len(command) for command, _ in SLASH_COMMANDS)
    lines = ["Chat commands:"]
    lines.extend(f"  {command.ljust(width)}  {description}" for command, description in SLASH_COMMANDS)
    return "\n".join(lines)


class SlashCommandCompleter(Completer):
    # 外部函数：为 Prompt Toolkit 输入框提供 slash command 补全，负责 agent 主循环、chat 界面、工具分发和运行时观察。
    def get_completions(self, document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        for command, description in SLASH_COMMANDS:
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )


# 外部函数：生成 chat 启动界面的版本、模型和目录信息，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def render_chat_banner(options: AgentOptions) -> str:
    model = os.environ.get("OPENAI_MODEL") or "(missing)"
    return "\n".join(
        [
            f"uedev {__version__}",
            f"model: {model}",
            f"directory: {options.cwd}",
            'Type "/" for commands; exit or Ctrl+C to quit.',
        ]
    )


# 内部函数：创建 Prompt Toolkit 会话，配置 slash 补全、输入历史和提示样式。
def create_chat_session() -> PromptSession:
    return PromptSession(
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
        history=InMemoryHistory(),
        style=Style.from_dict(
            {
                "completion-menu.completion": "fg:#c0c0c0 bg:#202020",
                "completion-menu.completion.current": "fg:#ffffff bg:#005f87",
                "completion-menu.meta.completion": "fg:#808080 bg:#202020",
                "completion-menu.meta.completion.current": "fg:#ffffff bg:#005f87",
                "prompt": "fg:#5fafff bold",
            }
        ),
    )


# 内部函数：构建系统提示词，描述工具协议、安全规则和当前工作目录。
def build_system_prompt(cwd: Path, skills: SkillLoader) -> str:
    return f"""You are a UE development agent running inside a command-line harness.

Architecture rule: the model supplies agency; the harness supplies tools, observation, permissions, context, tasks, team coordination, and worktree isolation.
Use the provided native tools when workspace, shell, UE, task, team, or file observation is required.
When no tool is needed, answer normally in concise prose.

Conversation behavior:
- If the user is chatting, greeting, testing the interface, asking a conceptual question, or does not clearly ask you to inspect, modify, run, or check the workspace, answer directly with a final action.
- Only call tools when the user asks for concrete local work or information that requires observing the workspace, shell, UE project, task state, or files.
- Do not list files or inspect the workspace just because the user sends a short test message.

Use workspace-relative paths, not absolute paths, unless the user explicitly gives an absolute path.

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


# 外部函数：执行 run 命令的一次性 agent 任务，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def run_agent(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Task: {options.task}"),
    ]
    final = runtime.run_turn(messages, goal=options.task)
    if final:
        print(f"\n{final}")


# 外部函数：执行 chat 命令的交互式会话界面，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def run_chat(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    session = create_chat_session()
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]

    print(render_chat_banner(options))
    while True:
        query = session.prompt([("class:prompt", "\n> ")]).strip()
        if query.lower() in {"", "quit", "exit"}:
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
    # 内部函数：初始化当前类实例，准备 agent 主循环、chat 界面、工具分发和运行时观察 所需状态。
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

    # 内部函数：推进模型思考、工具执行和结果回填，是 agent 主循环的核心流程。
    def run_turn(self, messages: list[ChatMessage], goal: str) -> str | None:
        work_tool_used = False
        rounds_without_todo = 0
        tool_specs = get_tool_specs()

        for step in range(1, self.options.max_steps + 1):
            self._inject_runtime_observations(messages)
            print(f"\nThinking..." if not self.options.verbose else f"\nThinking... ({step}/{self.options.max_steps})")

            response = call_model(messages, tools=tool_specs)
            if response.tool_calls:
                messages.append(ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls))
                for tool_call in response.tool_calls:
                    action = ToolAction(type="tool", name=tool_call.name, input=tool_call.arguments)
                    output = self._execute_tool(action)
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

                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=tool_content,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )

                    if action.name == "compact":
                        messages[:] = compact_locally(messages, self.agent_dir / "transcripts", "manual compact tool")
                        break
                continue

            final_answer = response.content.strip()
            fallback_action = parse_legacy_action(final_answer)
            if fallback_action is None:
                if requires_tool_action(goal) and not work_tool_used:
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "The request requires harness work. Use one of the provided native tools before answering."
                            ),
                        )
                    )
                    continue
                return final_answer

            action = fallback_action
            messages.append(ChatMessage(role="assistant", content=final_answer))

            if isinstance(action, FinalAction):
                if requires_tool_action(goal) and not work_tool_used:
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "The request requires harness work. Use one of the provided native tools before answering."
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
                continue

        print(f"\nStopped after {self.options.max_steps} iterations without a final answer.")
        return None

    # 外部函数：处理 chat 内本地 slash command，负责 agent 主循环、chat 界面、工具分发和运行时观察。
    def handle_slash_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False

        command = query.lower().strip()
        if command == "/help":
            print(render_slash_help())
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

    # 内部函数：处理 _inject_runtime_observations 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
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

    # 内部函数：处理 _append_protocol_correction 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
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

    # 内部函数：处理 _execute_tool 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _execute_tool(self, action: ToolAction) -> str:
        handler = self.tools.get(action.name)
        if handler is None:
            return f"Unknown tool: {action.name}"
        try:
            return handler(action.input)
        except Exception as error:
            return f"Tool {action.name} failed: {error}"

    # 内部函数：处理 _build_tool_handlers 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _build_tool_handlers(self) -> dict[str, ToolHandler]:
        # 内部函数：处理 shell 工具调用，确认权限后执行命令并格式化结果。
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

        # 内部函数：处理 background_run 工具调用，确认权限后启动后台命令。
        def background_run_tool(tool_input: dict[str, object]) -> str:
            command = str(tool_input.get("command", "")).strip()
            timeout = int(tool_input.get("timeout_seconds") or tool_input.get("timeout") or self.options.timeout_seconds)
            reason = str(tool_input.get("reason") or "agent requested background command")
            if not (self.options.auto_approve or confirm_command(command, reason)):
                return "The user rejected that background command."
            return self.background.run(command, timeout)

        # 内部函数：处理 ue_run_python 工具调用，执行 dry-run 或受控启动 UE。
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

        # 内部函数：处理 edit_file 工具调用，兼容单次替换和 edits 列表两种输入格式。
        def edit_file_tool(tool_input: dict[str, object]) -> str:
            path = str(tool_input.get("path", ""))
            edits = tool_input.get("edits")
            if isinstance(edits, list):
                results = []
                for edit in edits:
                    if not isinstance(edit, dict):
                        raise ValueError("each edit must be an object")
                    old_text = str(edit.get("old_text", edit.get("oldText", "")))
                    new_text = str(edit.get("new_text", edit.get("newText", "")))
                    results.append(edit_file(self.options.cwd, path, old_text, new_text))
                return "\n".join(results)

            old_text = str(tool_input.get("old_text", tool_input.get("oldText", "")))
            new_text = str(tool_input.get("new_text", tool_input.get("newText", "")))
            return edit_file(self.options.cwd, path, old_text, new_text)

        # 内部函数：处理 subagent 工具调用，启动受限子 agent 完成子任务。
        def subagent_tool(tool_input: dict[str, object]) -> str:
            prompt = str(tool_input.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("subagent requires prompt")
            return self._run_subagent(prompt, str(tool_input.get("agent_type", "explore")))

        return {
            "shell": shell_tool,
            "read_file": lambda data: read_file(self.options.cwd, str(data.get("path", "")), _optional_int(data.get("limit"))),
            "write_file": lambda data: write_file(self.options.cwd, str(data.get("path", "")), str(data.get("content", ""))),
            "edit_file": edit_file_tool,
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

    # 内部函数：处理 _run_subagent 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
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

        subagent_tools = [
            tool for tool in get_tool_specs() if str(tool["function"]["name"]) in allowed
        ]
        summaries: list[str] = []
        for _ in range(min(6, self.options.max_steps)):
            response = call_model(child_messages, tools=subagent_tools)
            if response.tool_calls:
                child_messages.append(ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls))
                for tool_call in response.tool_calls:
                    action = ToolAction(type="tool", name=tool_call.name, input=tool_call.arguments)
                    if action.name not in allowed:
                        output = f"Subagent tool not allowed: {action.name}"
                    else:
                        output = self._execute_tool(action)
                    summaries.append(f"{action.name}: {truncate(output, 1000)}")
                    child_messages.append(
                        ChatMessage(
                            role="tool",
                            content=f"Tool result for: {action.name}\n{truncate(output)}",
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )
                continue

            fallback_action = parse_legacy_action(response.content)
            if fallback_action is None:
                return response.content
            child_messages.append(ChatMessage(role="assistant", content=response.content))
            if isinstance(fallback_action, FinalAction):
                return fallback_action.answer
            if isinstance(fallback_action, ShellAction):
                fallback_action = ToolAction(
                    type="tool",
                    name="shell",
                    input={"command": fallback_action.command, "reason": fallback_action.reason},
                )
            if isinstance(fallback_action, ToolAction):
                if fallback_action.name not in allowed:
                    output = f"Subagent tool not allowed: {fallback_action.name}"
                else:
                    output = self._execute_tool(fallback_action)
                summaries.append(f"{fallback_action.name}: {truncate(output, 1000)}")
                child_messages.append(ChatMessage(role="user", content=f"Tool result for: {fallback_action.name}\n{truncate(output)}"))
        return "Subagent stopped after bounded steps.\n" + "\n".join(summaries[-5:])


# 内部函数：截断过长工具输出，避免 observation 撑爆上下文。
def truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


# 内部函数：兼容旧版模型手写 JSON action，原生 tool calling 迁移后仅作为兜底路径。
def parse_legacy_action(content: str) -> FinalAction | ShellAction | ToolAction | None:
    stripped = content.strip()
    if not stripped.startswith("{") and "{" not in stripped:
        return None
    try:
        return parse_agent_action(stripped)
    except ActionParseError:
        return None


# 内部函数：判断用户目标是否必须使用工具，防止执行类任务直接 final。
def requires_tool_action(goal: str) -> bool:
    normalized = goal.lower().strip()
    low_intent_inputs = {"test", "ping", "hi", "hello", "hey", "你好", "在吗", "测试"}
    if normalized in low_intent_inputs:
        return False

    tool_phrases = [
        "run test",
        "run tests",
        "run the tests",
        "run unit tests",
        "pytest",
        "unittest",
        "test suite",
    ]
    if any(phrase in normalized for phrase in tool_phrases):
        return True

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
        "check",
        "verify",
        "generate",
        "install",
    ]
    return any(keyword in normalized for keyword in keywords)


# 内部函数：处理 _require_list_of_dicts 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
def _require_list_of_dicts(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


# 内部函数：处理 _optional_str 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


# 内部函数：处理 _optional_int 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


# 内部函数：处理 _optional_int_list 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
def _optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected a list of integers")
    return [int(item) for item in value]
