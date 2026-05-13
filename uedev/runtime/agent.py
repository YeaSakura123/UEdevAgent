from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts.prompt import CompleteStyle
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - exercised only in minimal test environments.
    PromptSession = None  # type: ignore[assignment]

    class Completer:  # type: ignore[no-redef]
        pass

    class Completion:  # type: ignore[no-redef]
        def __init__(self, text: str, **kwargs: object) -> None:
            self.text = text
            self.display = kwargs.get("display", text)
            self.display_meta = kwargs.get("display_meta", "")

    class CompleteStyle:  # type: ignore[no-redef]
        COLUMN = "COLUMN"

    class CursorShape:  # type: ignore[no-redef]
        BLINKING_BLOCK = "BLINKING_BLOCK"

    class SimpleCursorShapeConfig:  # type: ignore[no-redef]
        def __init__(self, cursor_shape: object) -> None:
            self.cursor_shape = cursor_shape

    class InMemoryHistory:  # type: ignore[no-redef]
        pass

    class Style:  # type: ignore[no-redef]
        @staticmethod
        def from_dict(style: dict[str, str]) -> dict[str, str]:
            return style

from .. import __version__
from ..background import BackgroundManager
from ..config import (
    ConfigError,
    active_model_name,
    agent_dir,
    format_model_profiles,
    load_project_config,
    load_system_config,
    reset_project_active_model,
    resolve_model_profile,
    save_project_active_model,
)
from ..context import compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
from ..events import (
    AgentEvent,
    final_event,
    stopped_event,
    thinking_event,
    tool_error_event,
    tool_result_event,
    tool_start_event,
)
from ..llm import ChatMessage, call_model
from ..mcp.registry import McpToolRegistry, is_mcp_tool_name
from ..permissions import (
    CollaborationMode,
    PermissionMode,
    VALID_PERMISSION_MODES,
    classify_tool_permission,
    format_permission_modes,
    is_proposed_plan,
    normalize_permission_mode,
    permission_mode_description,
    permission_mode_label,
)
from ..prompts import PromptBundle, build_prompt_bundle, build_system_prompt as render_system_prompt
from ..renderer import ConsoleRenderer
from ..shell import ApprovalProvider, confirm_command, run_shell, shell_name
from ..skills import SkillLoader
from ..tasks import TaskManager, TodoManager
from ..team import MessageBus, TeamManager
from ..tool_specs import get_tool_specs
from ..ue import (
    discover_ue,
    enqueue_editor_stop,
    execute_prepared_ue_python,
    p4_add,
    p4_checkout,
    p4_delete,
    p4_diff,
    p4_file_state,
    p4_opened,
    p4_reconcile,
    p4_status,
    prepare_ue_python,
    render_doctor,
    render_run_result,
)
from ..workspace import edit_file, list_files, read_file, write_file
from ..worktrees import WorktreeManager


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool
    context_threshold: int = 60000
    plain: bool = False


ToolHandler = Callable[[dict[str, object]], str]
RUNTIME_STATE_MARKER = "<runtime-state>"


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
    ("/model", "List or switch model profiles for this project."),
    ("/mcp", "Show configured MCP server status and tools."),
    ("/plan", "Enter, leave, or inspect Plan Mode."),
    ("/permissions", "Show or switch the current permission mode."),
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

        permission_matches = _match_permission_mode_commands(text)
        if permission_matches:
            for command, description in permission_matches:
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )
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


def _match_permission_mode_commands(text: str) -> list[tuple[str, str]]:
    lower = text.lower()
    if not (lower == "/permissions" or lower.startswith("/permissions ")):
        return []

    query = ""
    if lower.startswith("/permissions "):
        query = lower.split(" ", 1)[1].strip()

    matches: list[tuple[str, str]] = []
    for mode in VALID_PERMISSION_MODES:
        label = permission_mode_label(mode)
        command = f"/permissions {label}"
        if not query or label.startswith(query) or query in label:
            matches.append((command, permission_mode_description(mode)))
    return matches


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
    try:
        model = active_model_name(options.cwd)
    except ConfigError:
        model = "(missing config)"
    return "\n".join(
        [
            f">_ uedev (v{__version__})",
            f"model:     {model}",
            f"directory: {options.cwd}",
        ]
    )


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
        "complete_style": CompleteStyle.COLUMN,
        "reserve_space_for_menu": 8,
        "cursor": SimpleCursorShapeConfig(CursorShape.BLINKING_BLOCK),
        "refresh_interval": 0.5,
    }


def create_chat_session(
    completer: Completer | None = None,
    input=None,
    output=None,
    key_bindings=None,
    input_processors=None,
) -> PromptSession:
    if PromptSession is None:
        raise RuntimeError("prompt_toolkit is required for interactive chat sessions")
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
    if key_bindings is not None:
        kwargs["key_bindings"] = key_bindings
    if input_processors is not None:
        kwargs["input_processors"] = input_processors
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
    if options.plain or not (sys.stdin.isatty() and sys.stdout.isatty()):
        run_plain_chat(options)
        return

    from ..tui import ChatTuiApplication

    runtime = AgentRuntime(options)
    ChatTuiApplication(
        options=options,
        runtime=runtime,
        banner=render_chat_banner(options),
        completer=SlashCommandCompleter(),
    ).run()


# 外部函数：执行非 TUI 的交互式 chat fallback，负责普通控制台事件流输出。
def run_plain_chat(options: AgentOptions) -> None:
    runtime = AgentRuntime(options)
    session = create_chat_session()
    messages = [
        ChatMessage(role="system", content=runtime.system_prompt),
        ChatMessage(role="user", content=f"Working directory: {options.cwd}\nShell: {shell_name()}"),
    ]
    renderer = ConsoleRenderer(verbose=options.verbose)

    print(render_chat_banner(options))
    while True:
        query = session.prompt([("class:prompt", "\n> ")], **create_chat_prompt_options()).strip()
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
    def __init__(self, options: AgentOptions, approval_provider: ApprovalProvider | None = None):
        self.options = options
        self.approval_provider = approval_provider or confirm_command
        self.agent_dir = agent_dir(options.cwd)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        project_config = load_project_config(options.cwd)
        self.collaboration_mode: CollaborationMode = "default"
        self.permission_mode: PermissionMode = "full_access" if options.auto_approve else project_config.permission_mode
        self.todo_manager = TodoManager(self.agent_dir)
        self.task_manager = TaskManager(self.agent_dir / "tasks")
        self.skill_loader = SkillLoader(options.cwd / "skills")
        self.background = BackgroundManager(options.cwd)
        self.bus = MessageBus(self.agent_dir / "team")
        self.team = TeamManager(self.agent_dir / "team", self.task_manager, self.bus)
        self.worktrees = WorktreeManager(options.cwd, self.agent_dir / "worktrees", self.task_manager)
        self.mcp = McpToolRegistry.from_system_config()
        self.prompt_bundle: PromptBundle = build_prompt_bundle(
            options.cwd,
            shell_name(),
            self.skill_loader.descriptions(),
        )
        self.system_prompt = self.prompt_bundle.system_prompt
        self.tools = self._build_tool_handlers()
        self.tool_specs = get_tool_specs(self.mcp.tool_specs())

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
        current_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        started_at = time.perf_counter()
        tool_names_this_turn: list[str] = []

        for step in range(1, self.options.max_steps + 1):
            self._inject_runtime_observations(messages)
            yield thinking_event(step, self.options.max_steps, current_turn_id)

            response = call_model(messages, self.current_model_profile(), tools=self.tool_specs)
            if response.tool_calls:
                messages.append(ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls))
                for tool_call in response.tool_calls:
                    action = ToolAction(name=tool_call.name, input=tool_call.arguments)
                    tool_names_this_turn.append(action.name)
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
            if self.collaboration_mode == "plan" and not is_proposed_plan(final_answer):
                messages.append(ChatMessage(role="assistant", content=final_answer))
                messages.append(
                    ChatMessage(
                        role="system",
                        content=(
                            "Invalid final answer: Plan Mode final answers must be wrapped exactly in "
                            "<proposed_plan> and </proposed_plan>. Do not implement changes while Plan Mode is active."
                        ),
                    )
                )
                continue
            if tool_names_this_turn and is_acknowledgement_answer(final_answer):
                messages.append(ChatMessage(role="assistant", content=final_answer))
                messages.append(
                    ChatMessage(
                        role="system",
                        content=(
                            "Invalid final answer: do not acknowledge instructions or future behavior. "
                            "Answer the user's request using the latest tool result. "
                            "Do not call todo_update for acknowledgements."
                        ),
                    )
                )
                continue
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

        raw_command = query.strip()
        command = raw_command.lower()
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
        if command == "/model":
            try:
                emit(self.render_models())
            except ConfigError as error:
                emit(f"Config error: {error}")
            return True
        if command == "/mcp":
            emit(self.mcp.render_status())
            return True
        if command.startswith("/model "):
            try:
                emit(self.switch_model(raw_command.split(maxsplit=1)[1].strip()))
            except ConfigError as error:
                emit(f"Config error: {error}")
            return True
        if command == "/plan" or command.startswith("/plan "):
            emit(self.handle_plan_command(raw_command))
            return True
        if command == "/permissions" or command.startswith("/permissions "):
            try:
                emit(self.handle_permissions_command(raw_command))
            except ConfigError as error:
                emit(f"Config error: {error}")
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

    def handle_plan_command(self, raw_command: str) -> str:
        arg = raw_command.split(maxsplit=1)[1].strip().lower() if len(raw_command.split(maxsplit=1)) > 1 else ""
        if arg in {"", "on"}:
            self.collaboration_mode = "plan"
            return "Plan Mode enabled. Use Shift+Tab or /plan off to exit."
        if arg in {"off", "default"}:
            self.collaboration_mode = "default"
            return "Plan Mode disabled."
        if arg == "status":
            return f"Collaboration mode: {self.collaboration_mode}"
        return "Usage: /plan, /plan off, or /plan status"

    def handle_permissions_command(self, raw_command: str) -> str:
        parts = raw_command.split(maxsplit=1)
        if len(parts) == 1:
            return format_permission_modes(self.permission_mode) + "\nChanges made with /permissions affect this chat session only."

        raw_mode = parts[1].strip()
        mode = normalize_permission_mode(raw_mode)
        if mode is None:
            available = ", ".join(permission_mode_label(item) for item in VALID_PERMISSION_MODES)
            return f"Unknown permission mode: {raw_mode}\nAvailable modes: {available}"

        self.permission_mode = mode
        return f"Permission mode set to {permission_mode_label(mode)} for this chat session."

    # 内部函数：处理 _inject_runtime_observations 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _inject_runtime_observations(self, messages: list[ChatMessage]) -> None:
        micro_compact(messages)
        repair_tool_call_messages(messages)
        if estimate_tokens(messages) > self.options.context_threshold:
            messages[:] = compact_locally(messages, self.agent_dir / "transcripts", "automatic threshold")
            repair_tool_call_messages(messages)
        self._inject_runtime_state(messages)

        notifications = self.background.drain()
        if notifications:
            rendered = "\n".join(f"[bg:{task.id}] {task.status}\n{truncate(task.result, 1000)}" for task in notifications)
            messages.append(ChatMessage(role="user", content=f"<background-results>\n{rendered}\n</background-results>"))

        inbox = self.bus.read_inbox("lead")
        if inbox:
            messages.append(ChatMessage(role="user", content=f"<inbox>{json.dumps(inbox, ensure_ascii=False, indent=2)}</inbox>"))

    def _inject_runtime_state(self, messages: list[ChatMessage]) -> None:
        messages[:] = [
            message
            for message in messages
            if not (message.role == "system" and message.content.startswith(RUNTIME_STATE_MARKER))
        ]
        messages.append(
            ChatMessage(
                role="system",
                content=(
                    f"{RUNTIME_STATE_MARKER}\n"
                    f"collaboration_mode: {self.collaboration_mode}\n"
                    f"permission_mode: {permission_mode_label(self.permission_mode)}\n"
                    "Rules:\n"
                    "- /plan enables Plan Mode; /plan off disables it.\n"
                    "- /permissions shows or changes read-only, default, auto-review, and full-access modes.\n"
                    "- In Plan Mode, do not modify files, persistent state, worktrees, or UE editor state.\n"
                    "- In Plan Mode, final answers must be wrapped in <proposed_plan> and </proposed_plan>.\n"
                    "- Permission mode is enforced by the harness before each tool call; call tools directly when needed.\n"
                    "</runtime-state>"
                ),
            )
        )

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
            messages.append(ChatMessage(role="system", content=self.prompt_bundle.tool_confirmation_reminder))
            return True
        return False

    def current_model_profile(self):
        return resolve_model_profile(self.options.cwd)

    def render_models(self) -> str:
        return format_model_profiles(self.options.cwd)

    def switch_model(self, name: str) -> str:
        if not name:
            return self.render_models()
        if name.lower() == "reset":
            reset_project_active_model(self.options.cwd)
            profile = resolve_model_profile(self.options.cwd)
            return f"Active model reset to default profile {profile.name}: {profile.model or '(missing model)'}"

        config = load_system_config()
        if name not in config.models:
            available = ", ".join(sorted(config.models)) or "(none)"
            return f"Unknown model profile: {name}\nAvailable profiles: {available}"
        save_project_active_model(self.options.cwd, name)
        profile = config.models[name]
        return f"Active model set to {profile.name}: {profile.model or '(missing model)'}"

    def _confirm(self, command: str, reason: str) -> bool:
        return self.approval_provider(command, reason)

    def _guard_tool(self, name: str, handler: ToolHandler) -> ToolHandler:
        def guarded(tool_input: dict[str, object]) -> str:
            decision = classify_tool_permission(
                name,
                tool_input,
                collaboration_mode=self.collaboration_mode,
                permission_mode=self.permission_mode,
            )
            if decision.action == "deny":
                return f"Tool denied by policy: {decision.reason}"
            if decision.action == "ask":
                label = _permission_prompt_label(name, tool_input)
                reason = str(tool_input.get("reason") or decision.reason)
                if not self._confirm(label, reason):
                    return "The user rejected that action."
            return handler(tool_input)

        return guarded

    def _build_tool_handlers(self) -> dict[str, ToolHandler]:
        # 内部函数：处理 shell 工具调用，确认权限后执行命令并格式化结果。
        def shell_tool(tool_input: dict[str, object]) -> str:
            command = str(tool_input.get("command", "")).strip()
            if not command:
                raise ValueError("shell command cannot be empty")

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
            return self.background.run(command, timeout)

        # 内部函数：处理 ue_run_python 工具调用，执行 dry-run 或受控启动 UE。
        def ue_run_python_tool(tool_input: dict[str, object]) -> str:
            cwd = self._resolve_tool_cwd(tool_input.get("cwd"))
            script, source_script_path = self._resolve_ue_script_input(tool_input, cwd)
            mode = self._resolve_ue_mode(tool_input.get("mode"))
            prepared = prepare_ue_python(
                cwd=cwd,
                agent_dir=agent_dir(cwd),
                script=script,
                mode=mode,
                kind="custom",
                source_script_path=source_script_path,
            )
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

        handlers: dict[str, ToolHandler] = {
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
            "ue_stop_executor": lambda data: f"queued_stop_task: {enqueue_editor_stop(agent_dir(self._resolve_tool_cwd(data.get('cwd'))))}",
            "p4_status": lambda data: p4_status(self._resolve_tool_cwd(data.get("cwd"))),
            "p4_file_state": lambda data: p4_file_state(
                self._resolve_tool_cwd(data.get("cwd")),
                _string_list(data.get("paths")),
            ),
            "p4_opened": lambda data: p4_opened(
                self._resolve_tool_cwd(data.get("cwd")),
                _optional_str(data.get("changelist")),
            ),
            "p4_checkout": lambda data: p4_checkout(
                self._resolve_tool_cwd(data.get("cwd")),
                _string_list(data.get("paths")),
                _optional_str(data.get("changelist")),
            ),
            "p4_add": lambda data: p4_add(
                self._resolve_tool_cwd(data.get("cwd")),
                _string_list(data.get("paths")),
                _optional_str(data.get("changelist")),
            ),
            "p4_delete": lambda data: p4_delete(
                self._resolve_tool_cwd(data.get("cwd")),
                _string_list(data.get("paths")),
                _optional_str(data.get("changelist")),
            ),
            "p4_reconcile": lambda data: p4_reconcile(
                self._resolve_tool_cwd(data.get("cwd")),
                _optional_string_list(data.get("paths")),
                _optional_str(data.get("changelist")),
            ),
            "p4_diff": lambda data: p4_diff(
                self._resolve_tool_cwd(data.get("cwd")),
                _optional_string_list(data.get("paths")),
            ),
        }
        handlers.update(self.mcp.handlers())
        return {name: self._guard_tool(name, handler) for name, handler in handlers.items()}

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
        if raw in {"commandlet", "cmd", "commandline", "command_line"}:
            return "commandlet"
        if raw in {"full_editor", "editor", "full", "gui", ""}:
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
            if _looks_like_inline_runpy_loader(script):
                raise ValueError("Use script_path instead of passing an inline runpy.run_path loader script.")
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
            response = call_model(child_messages, self.current_model_profile(), tools=subagent_tools)
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
def _permission_prompt_label(name: str, tool_input: dict[str, object]) -> str:
    if name in {"shell", "background_run", "worktree_run"}:
        return str(tool_input.get("command") or name)
    if name in {"write_file", "edit_file", "read_file", "list_files"}:
        path = str(tool_input.get("path") or "").strip()
        return f"{name} {path}".strip()
    if name == "worktree_remove":
        worktree = str(tool_input.get("name") or "").strip()
        return f"worktree_remove {worktree}".strip()
    if name.startswith("p4_"):
        paths = tool_input.get("paths")
        if isinstance(paths, list):
            rendered = " ".join(str(path) for path in paths[:3])
            if len(paths) > 3:
                rendered += f" ... ({len(paths)} paths)"
            return f"{name} {rendered}".strip()
        path = str(paths or tool_input.get("cwd") or "").strip()
        return f"{name} {path}".strip()
    if is_mcp_tool_name(name):
        return name
    return name


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
    if not any(
        token in goal_text
        for token in ["ue", "unreal", "editor", "script", "execute", "launch", "run", "脚本", "执行", "启动", "运行", ".py"]
    ):
        return False
    confirmation_tokens = ["confirm", "confirmation", "确认", "是否", "y/n", "[y/n]"]
    action_tokens = ["run", "execute", "launch", "start", "continue", "执行", "启动", "运行", "继续"]
    return any(token in answer_text for token in confirmation_tokens) and any(
        token in answer_text for token in action_tokens
    )


def is_acknowledgement_answer(answer: str) -> bool:
    normalized = " ".join(answer.strip().lower().split())
    if not normalized:
        return False
    result_tokens = [
        "project=",
        "ue doctor",
        "engine",
        "perforce",
        "项目",
        "引擎",
        "版本",
        "存在",
        "不存在",
        "结果",
        "error",
        "failed",
    ]
    if any(token in normalized for token in result_tokens):
        return False
    acknowledgement_tokens = [
        "understood",
        "got it",
        "i'll",
        "i will",
        "i’ll",
        "will follow",
        "will directly invoke",
        "future",
        "harness",
        "收到",
        "明白",
        "了解",
        "已按你的要求",
        "会遵循",
        "遵循该行为",
        "以后会",
        "下次会",
    ]
    return any(token in normalized for token in acknowledgement_tokens)


def _looks_like_inline_runpy_loader(script: str) -> bool:
    lines = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) > 2:
        return False
    return any(line == "import runpy" or line.startswith("import runpy ") for line in lines) and any(
        "runpy.run_path(" in line for line in lines
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


def _string_list(value: object) -> list[str]:
    result = _optional_string_list(value)
    if not result:
        raise ValueError("expected a non-empty list of strings")
    return result


def _optional_string_list(value: object) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value]
