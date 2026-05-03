from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.styles import Style

from . import __version__
from .background import BackgroundManager
from .context import compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
from .events import (
    AgentEvent,
    final_event,
    stopped_event,
    thinking_event,
    tool_error_event,
    tool_result_event,
    tool_start_event,
)
from .llm import ChatMessage, call_model
from .prompts import PromptBundle, build_prompt_bundle, build_system_prompt as render_system_prompt
from .renderer import ConsoleRenderer
from .shell import confirm_command, run_shell, shell_name
from .skills import SkillLoader
from .tasks import TaskManager, TodoManager
from .team import MessageBus, TeamManager
from .tool_specs import get_tool_specs
from .ue import (
    UeRunResult,
    discover_ue,
    enqueue_editor_stop,
    execute_prepared_ue_python,
    prepare_ue_python,
    render_doctor,
    render_run_result,
)
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
    context_threshold: int = 60000


ToolHandler = Callable[[dict[str, object]], str]


@dataclass(frozen=True)
class ToolAction:
    name: str
    input: dict[str, Any]


SLASH_COMMANDS = [
    ("/help", "Show available chat slash commands."),
    ("/todos", "Show the current lightweight todo list."),
    ("/tasks", "Show the persistent task graph."),
    ("/team", "Show the persistent teammate roster."),
    ("/inbox", "Show pending messages for the lead agent."),
    ("/doctor", "Inspect Unreal Engine project and editor configuration."),
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

        for command, description in _match_slash_commands(text):
            yield Completion(
                command,
                start_position=-len(text),
                display=command,
                display_meta=description,
            )


def _match_slash_commands(text: str) -> list[tuple[str, str]]:
    query = _normalize_slash_query(text)
    if not query:
        return SLASH_COMMANDS.copy()

    prefix_matches: list[tuple[str, str]] = []
    word_matches: list[tuple[str, str]] = []
    fuzzy_matches: list[tuple[str, str]] = []

    for command, description in SLASH_COMMANDS:
        command_key = command.lower()
        command_compact = _normalize_slash_query(command)
        words = [part for part in command_key.replace("/", " ").split() if part]

        if command_key.startswith(text.lower()):
            prefix_matches.append((command, description))
        elif any(word.startswith(query) for word in words):
            word_matches.append((command, description))
        elif query in command_compact or _is_subsequence(query, command_compact):
            fuzzy_matches.append((command, description))

    ordered: list[tuple[str, str]] = []
    emitted: set[str] = set()
    for group in (prefix_matches, word_matches, fuzzy_matches):
        for command, description in group:
            if command not in emitted:
                ordered.append((command, description))
                emitted.add(command)
    return ordered


def _normalize_slash_query(text: str) -> str:
    return text.lower().lstrip("/").replace(" ", "")


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    position = 0
    for char in haystack:
        if char == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


# 外部函数：生成 chat 启动界面的版本、模型和目录信息，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def render_chat_banner(options: AgentOptions) -> str:
    model = os.environ.get("OPENAI_MODEL") or "(missing)"
    rows = [
        f">_ uedev (v{__version__})",
        "",
        f"model:     {model}",
        f"directory: {options.cwd}",
    ]
    width = max(len(row) for row in rows)
    framed = ["╭" + "─" * (width + 2) + "╮"]
    framed.extend(f"│ {row.ljust(width)} │" for row in rows)
    framed.append("╰" + "─" * (width + 2) + "╯")
    framed.append("")
    framed.append('  Tip: Type "/" for commands; exit or Ctrl+C to quit.')
    return "\n".join(framed)


# 内部函数：创建 Prompt Toolkit 会话，配置 slash 补全、输入历史和提示样式。
def create_chat_style() -> Style:
    return Style.from_dict(
        {
            "completion-menu.completion": "fg:#c0c0c0 bg:#202020",
            "completion-menu.completion.current": "fg:#ffffff bg:#005f87",
            "completion-menu.meta.completion": "fg:#808080 bg:#202020",
            "completion-menu.meta.completion.current": "fg:#ffffff bg:#005f87",
            "prompt": "fg:#5fafff bold",
        }
    )


def create_chat_prompt_options() -> dict[str, object]:
    return {
        "complete_while_typing": True,
        "complete_style": CompleteStyle.MULTI_COLUMN,
        "reserve_space_for_menu": 8,
        "cursor": SimpleCursorShapeConfig(CursorShape.BLINKING_BLOCK),
        "refresh_interval": 0.5,
    }


def create_chat_session(completer: Completer | None = None, input=None, output=None) -> PromptSession:
    kwargs = {
        "completer": completer or SlashCommandCompleter(),
        "complete_while_typing": True,
        "history": InMemoryHistory(),
        "style": create_chat_style(),
    }
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    return PromptSession(**kwargs)


# 内部函数：构建系统提示词，描述工具协议、安全规则和当前工作目录。
def build_system_prompt(cwd: Path, skills: SkillLoader) -> str:
    return render_system_prompt(cwd, shell_name(), skills.descriptions())


# 外部函数：执行 run 命令的一次性 agent 任务，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def run_agent(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Task: {options.task}"),
    ]
    renderer = ConsoleRenderer(verbose=options.verbose)
    for event in runtime.run_turn_events(messages, goal=options.task):
        renderer.render(event)


# 外部函数：执行 chat 命令的交互式会话界面，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def run_chat(options: AgentOptions) -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from .tui import ChatTuiApplication

            runtime = AgentRuntime(options)
            ChatTuiApplication(
                options=options,
                runtime=runtime,
                banner=render_chat_banner(options),
                completer=SlashCommandCompleter(),
            ).run()
            return
        except Exception as error:
            print(f"TUI unavailable, falling back to console chat: {error}", file=sys.stderr)

    run_console_chat(options)


# 外部函数：执行非 TUI 的交互式 chat fallback，负责普通控制台事件流输出。
def run_console_chat(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    session = create_chat_session()
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]
    renderer = ConsoleRenderer(verbose=options.verbose)

    print(render_chat_banner(options))
    while True:
        query = session.prompt([("class:prompt", "\n› ")], **create_chat_prompt_options()).strip()
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
        for event in runtime.run_turn_events(messages, goal=query):
            renderer.render(event)


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
        self.prompt_bundle: PromptBundle = build_prompt_bundle(
            options.cwd,
            shell_name(),
            self.skill_loader.descriptions(),
        )
        self.system_prompt = self.prompt_bundle.system_prompt
        self.tools = self._build_tool_handlers()

    # 内部函数：推进模型思考、工具执行和结果回填，是 agent 主循环的核心流程。
    def run_turn(self, messages: list[ChatMessage], goal: str) -> str | None:
        final_answer: str | None = None
        for event in self.run_turn_events(messages, goal):
            if event.type == "final":
                final_answer = event.message
            elif event.type == "stopped":
                final_answer = None
        return final_answer

    # 外部函数：推进模型思考、工具执行和结果回填，按事件流暴露 agent 主循环过程。
    def run_turn_events(self, messages: list[ChatMessage], goal: str, turn_id: str | None = None) -> Iterator[AgentEvent]:
        rounds_without_todo = 0
        tool_specs = get_tool_specs()
        current_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        started_at = time.perf_counter()

        for step in range(1, self.options.max_steps + 1):
            self._inject_runtime_observations(messages)
            yield thinking_event(step, self.options.max_steps, current_turn_id)

            response = call_model(messages, tools=tool_specs)
            if response.tool_calls:
                messages.append(ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls))
                for tool_call in response.tool_calls:
                    action = ToolAction(name=tool_call.name, input=tool_call.arguments)
                    yield tool_start_event(action.name, action.input, current_turn_id)
                    output, is_error = self._execute_tool_with_status(action)
                    if action.name == "todo_update":
                        rounds_without_todo = 0
                    else:
                        rounds_without_todo += 1

                    if is_error:
                        yield tool_error_event(action.name, output, current_turn_id)
                    else:
                        yield tool_result_event(action.name, output, current_turn_id)

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
            if self._defer_final_if_tool_needed(messages, goal, final_answer, record_assistant=True):
                continue
            yield final_event(final_answer, current_turn_id, _duration_ms(started_at))
            return

        yield stopped_event(
            f"Stopped after {self.options.max_steps} iterations without a final answer.",
            current_turn_id,
            _duration_ms(started_at),
        )
        return

    # 外部函数：处理 chat 内本地 slash command，负责 agent 主循环、chat 界面、工具分发和运行时观察。
    def handle_slash_command(self, query: str, emit: Callable[[str], None] = print) -> bool:
        if not query.startswith("/"):
            return False

        command = query.lower().strip()
        if command == "/help":
            emit(render_slash_help())
            return True
        if command == "/todos":
            emit(self.todo_manager.render_current())
            return True
        if command == "/tasks":
            emit(self.task_manager.list_all())
            return True
        if command == "/team":
            emit(self.team.list_all())
            return True
        if command == "/inbox":
            emit(json.dumps(self.bus.read_inbox("lead"), ensure_ascii=False, indent=2))
            return True
        if command == "/doctor":
            emit(render_doctor(discover_ue(self.options.cwd)))
            return True
        if command == "/ue doctor":
            emit(render_doctor(discover_ue(self.options.cwd)))
            return True
        if command == "/compact":
            emit("Use the compact tool during an agent turn, or /clear to reset chat context.")
            return True

        emit(f"Unknown slash command: {query}")
        return True

    # 内部函数：处理 _inject_runtime_observations 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _inject_runtime_observations(self, messages: list[ChatMessage]) -> None:
        micro_compact(messages)
        repair_tool_call_messages(messages)
        if estimate_tokens(messages) > self.options.context_threshold:
            messages[:] = compact_locally(messages, self.agent_dir / "transcripts", "automatic threshold")
            repair_tool_call_messages(messages)

        notifications = self.background.drain()
        if notifications:
            rendered = "\n".join(f"[bg:{task.id}] {task.status}\n{truncate(task.result, 1000)}" for task in notifications)
            messages.append(ChatMessage(role="user", content=f"<background-results>\n{rendered}\n</background-results>"))

        inbox = self.bus.read_inbox("lead")
        if inbox:
            messages.append(ChatMessage(role="user", content=f"<inbox>{json.dumps(inbox, ensure_ascii=False, indent=2)}</inbox>"))

    # 内部函数：处理 _execute_tool 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _execute_tool(self, action: ToolAction) -> str:
        output, _ = self._execute_tool_with_status(action)
        return output

    # 内部函数：执行工具并返回输出及错误标记，负责事件流中的 tool_result/tool_error 区分。
    def _execute_tool_with_status(self, action: ToolAction) -> tuple[str, bool]:
        handler = self.tools.get(action.name)
        if handler is None:
            return f"Unknown tool: {action.name}", True
        try:
            return handler(action.input), False
        except Exception as error:
            return f"Tool {action.name} failed: {error}", True

    # 内部函数：处理 _build_tool_handlers 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _defer_final_if_tool_needed(
        self,
        messages: list[ChatMessage],
        goal: str,
        answer: str,
        *,
        record_assistant: bool = False,
    ) -> bool:
        if defers_tool_confirmation(goal, answer):
            if record_assistant:
                messages.append(ChatMessage(role="assistant", content=answer))
            messages.append(ChatMessage(role="user", content=self.prompt_bundle.tool_confirmation_reminder))
            return True
        return False

    def _build_tool_handlers(self) -> dict[str, ToolHandler]:
        # 内部函数：处理 shell 工具调用，确认权限后执行命令并格式化结果。
        def shell_tool(tool_input: dict[str, object]) -> str:
            command = str(tool_input.get("command", "")).strip()
            reason = str(tool_input.get("reason", "")).strip() or "agent requested shell execution"
            if not command:
                raise ValueError("shell command cannot be empty")

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
            cwd = self._resolve_tool_cwd(tool_input.get("cwd"))
            script, source_script_path = self._resolve_ue_script_input(tool_input, cwd)
            mode = self._resolve_ue_mode(tool_input.get("mode"))
            prepared = prepare_ue_python(
                cwd=cwd,
                agent_dir=cwd / ".agent",
                script=script,
                mode=mode,
                kind="custom",
                source_script_path=source_script_path,
            )
            approved = confirm_command(
                prepared.command,
                "agent requested Unreal Engine Python execution",
            )
            if not approved:
                preview = UeRunResult(command=prepared.command, script_path=prepared.script_path, executed=False)
                return render_run_result(preview) + "\nUE execution rejected by user."
            result = execute_prepared_ue_python(
                prepared,
                cwd=cwd,
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
            "ue_doctor": lambda data: render_doctor(discover_ue(self._resolve_tool_cwd(data.get("cwd")))),
            "ue_run_python": ue_run_python_tool,
            "ue_stop_executor": lambda data: f"queued_stop_task: {enqueue_editor_stop(self._resolve_tool_cwd(data.get('cwd')) / '.agent')}",
        }

    # 内部函数：处理 _run_subagent 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _resolve_tool_cwd(self, raw_cwd: object) -> Path:
        raw = str(raw_cwd or "").strip()
        if not raw:
            return self.options.cwd
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.options.cwd / path
        return path.resolve()

    def _resolve_ue_mode(self, raw_mode: object) -> str:
        raw = str(raw_mode or "").strip().lower()
        if raw in {"", "commandlet", "cmd", "commandline", "command_line"}:
            return "commandlet"
        if raw in {"full_editor", "editor", "full", "gui"}:
            return "full_editor"
        raise ValueError("ue_run_python mode must be commandlet or full_editor")

    def _resolve_ue_script(self, tool_input: dict[str, object], cwd: Path) -> str:
        script, _ = self._resolve_ue_script_input(tool_input, cwd)
        return script

    def _resolve_ue_script_input(self, tool_input: dict[str, object], cwd: Path) -> tuple[str, Path | None]:
        script_path = str(tool_input.get("script_path") or "").strip()
        script = str(tool_input.get("script") or "")
        if not script_path:
            if not script.strip():
                raise ValueError("ue_run_python requires either script or script_path")
            return script, None

        path = Path(script_path).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"UE Python script file not found: {path}")
        if path.suffix.lower() != ".py":
            raise ValueError(f"UE Python script_path must point to a .py file: {path}")
        return path.read_text(encoding="utf-8"), path

    def _run_subagent(self, prompt: str, agent_type: str) -> str:
        child_messages = [
            ChatMessage(
                role="system",
                content=self.prompt_bundle.subagent_prompt,
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
                    action = ToolAction(name=tool_call.name, input=tool_call.arguments)
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

            return response.content
        return "Subagent stopped after bounded steps.\n" + "\n".join(summaries[-5:])


# 内部函数：截断过长工具输出，避免 observation 撑爆上下文。
def truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


# 内部函数：计算当前 turn 已耗时毫秒数，负责事件摘要中的耗时字段。
def _duration_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def defers_tool_confirmation(goal: str, answer: str) -> bool:
    goal_text = goal.lower()
    answer_text = answer.lower()
    if not any(token in goal_text for token in ["ue", "unreal", "editor", "脚本", "执行", "启动", "运行", ".py"]):
        return False
    confirmation_tokens = ["confirm", "confirmation", "确认", "是否", "y/n", "[y/n]"]
    action_tokens = ["run", "execute", "launch", "start", "continue", "执行", "启动", "运行", "继续"]
    return any(token in answer_text for token in confirmation_tokens) and any(
        token in answer_text for token in action_tokens
    )


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
