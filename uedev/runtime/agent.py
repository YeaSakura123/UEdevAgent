from __future__ import annotations

import json
import subprocess
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
from ..tools.background import BackgroundManager
from ..state.config import (
    ConfigError,
    active_model_name,
    agent_dir,
    format_model_profiles,
    load_project_config,
    load_system_config,
    reset_project_active_model,
    resolve_model_profile,
    resolve_subagent_model_profile,
    save_project_active_model,
)
from .context import (
    build_compacted_history,
    build_compaction_request,
    estimate_tokens,
    latest_real_user_message,
    micro_compact,
    repair_tool_call_messages,
    save_transcript,
)
from ..ui.events import (
    AgentEvent,
    compact_event,
    final_event,
    stopped_event,
    thinking_event,
    tool_error_event,
    tool_result_event,
    tool_start_event,
)
from .history import (
    HistoryError,
    HistoryRecorder,
    create_standalone_session_transcript_path,
    ensure_system_prompt,
    list_history_entries,
    load_history_file,
)
from ..llm.client import ChatMessage, call_model
from ..mcp.registry import McpToolRegistry, is_mcp_tool_name
from ..policy.permissions import (
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
from .prompts import PromptBundle, build_prompt_bundle, build_system_prompt as render_system_prompt
from .subagents import SubagentManager, parse_subagent_spec
from ..ui.renderer import ConsoleRenderer
from ..tools.shell import ApprovalProvider, confirm_command, run_shell, shell_name
from .skills import SkillLoader
from ..state.tasks import TaskManager, TodoManager
from ..state.team import MessageBus, TeamManager
from ..tools.specs import get_tool_specs
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
from ..tools.workspace import edit_file, list_files, read_file, write_file
from ..tools.worktrees import WorktreeManager


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool
    context_threshold: int | None = None
    plain: bool = False


ToolHandler = Callable[[dict[str, object]], str]
RUNTIME_STATE_MARKER = "<runtime-state>"


@dataclass(frozen=True)
class ToolAction:
    name: str
    input: dict[str, Any]


SLASH_COMMANDS = [
    ("/help", "Show available chat slash commands."),
    ("/context", "Show current conversation context usage."),
    ("/diff", "Show Git and Perforce workspace changes."),
    ("/todos", "Show the current lightweight todo list."),
    ("/tasks", "Show the persistent task graph."),
    ("/team", "Show the persistent teammate roster."),
    ("/inbox", "Show pending messages for the lead agent."),
    ("/history", "Load a previous conversation from this project."),
    ("/subagents", "Choose a subagent conversation to view."),
    ("/worktree", "Create a UE Git linked worktree from the current project."),
    ("/model", "List or switch model profiles for this project."),
    ("/mcp", "Show configured MCP server status and tools."),
    ("/plan", "Enter, leave, or inspect Plan Mode."),
    ("/permissions", "Show or switch the current permission mode."),
    ("/doctor", "Inspect Unreal Engine project and editor configuration."),
    ("/ue doctor", "Inspect Unreal Engine project and editor configuration."),
    ("/compact", "Compact the current conversation context."),
    ("/clear", "Reset the current chat conversation context."),
]


# 外部函数：生成 /help 命令展示的帮助文本，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def render_slash_help() -> str:
    width = max(len(command) for command, _ in SLASH_COMMANDS)
    lines = ["Chat commands:"]
    lines.extend(f"  {command.ljust(width)}  {description}" for command, description in SLASH_COMMANDS)
    return "\n".join(lines)


def render_context_usage(messages: list[ChatMessage], model_profile: Any, compact_threshold: int) -> str:
    estimated = estimate_tokens(messages)
    context_window = int(getattr(model_profile, "context_window", 0) or 0)
    model_name = str(getattr(model_profile, "model", "") or "(missing model)")
    profile_name = str(getattr(model_profile, "name", "") or "(unknown)")

    return "\n".join(
        [
            "Context:",
            f"model: {model_name} (profile: {profile_name})",
            f"estimated tokens: {_format_number(estimated)}",
            f"context window: {_format_number(context_window)}",
            f"context usage: {_format_percent(estimated, context_window)}",
            "",
            f"auto compact threshold: {_format_number(compact_threshold)}",
            f"threshold usage: {_format_percent(estimated, compact_threshold)}",
            f"remaining to threshold: {_format_number(max(0, compact_threshold - estimated))}",
            f"remaining to window: {_format_number(max(0, context_window - estimated))}",
        ]
    )


def _format_number(value: int) -> str:
    return f"{value:,}"


def _format_percent(value: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(value / total) * 100:.1f}%"


@dataclass(frozen=True)
class _CommandCapture:
    args: list[str]
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    status: str = "completed"

    @property
    def command(self) -> str:
        return " ".join(self.args)


def render_workspace_diff(cwd: Path, timeout_seconds: int, max_output_chars: int) -> str:
    return "\n\n".join(
        [
            _render_git_diff(cwd, timeout_seconds, max_output_chars),
            _render_perforce_diff(cwd, max_output_chars),
        ]
    )


def _render_git_diff(cwd: Path, timeout_seconds: int, max_output_chars: int) -> str:
    probe = _run_readonly_command(["git", "rev-parse", "--is-inside-work-tree"], cwd, timeout_seconds)
    if probe.status == "missing":
        return f"Git:\n{probe.stderr.strip()}"
    if probe.exit_code != 0:
        return "Git:\nGit: not a repository"

    status = _run_readonly_command(["git", "status", "--short", "--branch"], cwd, timeout_seconds)
    unstaged = _run_readonly_command(["git", "diff", "--no-ext-diff"], cwd, timeout_seconds)
    staged = _run_readonly_command(["git", "diff", "--cached", "--no-ext-diff"], cwd, timeout_seconds)

    lines = [
        "Git:",
        *_render_git_status(status, max_output_chars),
        "",
        _render_patch_section("unstaged diff", unstaged, max_output_chars),
        _render_patch_section("staged diff", staged, max_output_chars),
    ]
    return "\n".join(lines)


def _render_perforce_diff(cwd: Path, max_output_chars: int) -> str:
    try:
        status_raw = p4_status(cwd)
    except Exception as error:
        status_raw = f"p4_status failed: {error}"
    try:
        opened_raw = p4_opened(cwd)
    except Exception as error:
        opened_raw = f"p4_opened failed: {error}"

    lines = ["Perforce:"]
    status_data = _parse_json_object(status_raw)
    if status_data is None:
        lines.extend(["p4_status: raw output", _truncate_diff_output(status_raw.rstrip() or "(no status output)", max_output_chars)])
    else:
        lines.extend(_render_p4_status_summary(status_data))

    lines.append("")
    opened_data = _parse_json_object(opened_raw)
    if opened_data is None:
        lines.extend(["opened: raw output", _truncate_diff_output(opened_raw.rstrip() or "(no opened output)", max_output_chars)])
    else:
        lines.extend(_render_p4_opened_summary(opened_data, max_output_chars))
    return "\n".join(lines)


def _run_readonly_command(args: list[str], cwd: Path, timeout_seconds: int) -> _CommandCapture:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return _CommandCapture(args, None, stderr=f"{args[0]} executable not found on PATH.", status="missing")
    except subprocess.TimeoutExpired:
        return _CommandCapture(args, 124, stderr=f"{' '.join(args)} timed out after {timeout_seconds}s.", status="timeout")
    except OSError as error:
        return _CommandCapture(args, None, stderr=f"Cannot run {' '.join(args)}: {error}", status="error")
    return _CommandCapture(args, result.returncode, result.stdout, result.stderr)


def _format_command_output(result: _CommandCapture, empty_message: str, max_output_chars: int) -> str:
    parts: list[str] = []
    if result.exit_code not in (0, None):
        parts.append(f"exitCode: {result.exit_code}")
    if result.stdout.strip():
        parts.append(_truncate_diff_output(result.stdout.rstrip(), max_output_chars))
    elif result.exit_code == 0:
        parts.append(empty_message)
    if result.stderr.strip():
        parts.append("stderr:")
        parts.append(_truncate_diff_output(result.stderr.rstrip(), max_output_chars))
    if not parts:
        return empty_message
    return "\n".join(parts)


def _render_git_status(result: _CommandCapture, max_output_chars: int) -> list[str]:
    if result.exit_code != 0:
        return ["status: failed", _format_command_output(result, "(no status output)", max_output_chars)]

    branch = "(unknown)"
    file_lines: list[str] = []
    staged = 0
    unstaged = 0
    untracked = 0

    for line in result.stdout.splitlines():
        if line.startswith("## "):
            branch = _format_git_branch(line[3:].strip())
            continue
        if not line.strip():
            continue
        file_lines.append(line)
        if line.startswith("??"):
            untracked += 1
            continue
        if len(line) >= 2:
            if line[0] not in {" ", "?", "!"}:
                staged += 1
            if line[1] not in {" ", "?", "!"}:
                unstaged += 1

    lines = [
        f"branch: {branch}",
        f"status: staged {staged}, unstaged {unstaged}, untracked {untracked}",
    ]
    if untracked:
        lines.append("note: untracked files are not included in git diff output.")
    if file_lines:
        lines.append("")
        lines.append("status files:")
        lines.append(_truncate_diff_output("\n".join(f"  {line}" for line in file_lines), max_output_chars))
    return lines


def _format_git_branch(raw: str) -> str:
    prefix = "No commits yet on "
    if raw.startswith(prefix):
        return f"{raw[len(prefix):]} (no commits)"
    return raw or "(unknown)"


def _render_patch_section(label: str, result: _CommandCapture, max_output_chars: int) -> str:
    if result.exit_code != 0:
        return f"{label}: failed\n{_format_command_output(result, '(no output)', max_output_chars)}"
    output = result.stdout.rstrip()
    if not output:
        return f"{label}: none"
    return f"{label}:\n{_truncate_diff_output(output, max_output_chars)}"


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _render_p4_status_summary(data: dict[str, Any]) -> list[str]:
    available = bool(data.get("available"))
    in_workspace = bool(data.get("in_workspace"))
    project_tracked = bool(data.get("project_tracked"))
    lines: list[str] = []

    if not available:
        lines.append("status: unavailable")
    elif not in_workspace:
        lines.append("status: available (not a workspace)")

    if available and in_workspace:
        lines.append(f"workspace: {_display_value(data.get('client_name'))}")
        lines.append(f"user: {_display_value(data.get('user_name'))}")
        lines.append(f"server: {_display_value(data.get('server_address'))}")
        lines.append(f"root: {_display_value(data.get('client_root'))}")
        project = _display_value(data.get("project_depot_path"))
        lines.append(f"project: {'tracked' if project_tracked else 'untracked'}" + (f" {project}" if project != "(unknown)" else ""))

    opened_count = data.get("opened_count")
    if opened_count is not None:
        lines.append(f"opened count: {opened_count}")

    notes = _string_items(data.get("notes"))
    if notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in notes)
    return lines or ["status: unknown"]


def _render_p4_opened_summary(data: dict[str, Any], max_output_chars: int) -> list[str]:
    if not bool(data.get("ok")):
        lines = ["opened: failed"]
        command = data.get("command")
        if command:
            lines.append(f"command: {command}")
        if data.get("exit_code") is not None:
            lines.append(f"exitCode: {data.get('exit_code')}")
        stderr = str(data.get("stderr") or "").strip()
        stdout = str(data.get("stdout") or "").strip()
        if stderr:
            lines.append("stderr:")
            lines.append(_truncate_diff_output(stderr, max_output_chars))
        elif stdout:
            lines.append("stdout:")
            lines.append(_truncate_diff_output(stdout, max_output_chars))
        return lines

    opened = _dedupe_preserve_order(_string_items(data.get("opened")))
    if not opened:
        return ["opened: none"]

    opened_lines = "\n".join(_format_p4_opened_line(line) for line in opened)
    return [
        f"opened: {data.get('opened_count', len(opened))}",
        _truncate_diff_output(opened_lines, max_output_chars),
    ]


def _format_p4_opened_line(line: str) -> str:
    stripped = line.strip()
    if " - " not in stripped:
        return f"  {stripped}"
    path_part, detail = stripped.split(" - ", 1)
    path = path_part.rsplit("#", 1)[0] if "#" in path_part else path_part
    detail_parts = detail.split()
    action = detail_parts[0] if detail_parts else "?"
    file_type = "-"
    type_start = detail.find("(")
    type_end = detail.find(")", type_start + 1)
    if type_start >= 0 and type_end > type_start:
        file_type = detail[type_start + 1 : type_end].strip() or "-"
    return f"  {action:<5} {file_type:<9} {path}"


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _display_value(value: object) -> str:
    text = str(value or "").strip()
    return text or "(unknown)"


def _truncate_diff_output(value: str, max_output_chars: int) -> str:
    limit = max(1, max_output_chars)
    if len(value) <= limit:
        return value
    return (
        f"{value[:limit]}\n"
        f"... truncated at {limit} chars; increase display.diff_output_max_chars in ~/.uedev/config.json ..."
    )


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
    ]
    renderer = ConsoleRenderer(verbose=options.verbose)
    for event in runtime.run_turn_events(messages, goal=options.task):
        renderer.render(event)


# 外部函数：执行 chat 命令的交互式会话界面，负责 agent 主循环、chat 界面、工具分发和运行时观察。
def run_chat(options: AgentOptions) -> None:
    if options.plain or not (sys.stdin.isatty() and sys.stdout.isatty()):
        run_plain_chat(options)
        return

    from ..ui.tui import ChatTuiApplication

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
    history = HistoryRecorder(runtime.agent_dir, messages)
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
            history.reset(messages)
            print("Conversation context cleared.")
            continue

        if query.lower() == "/history":
            loaded = _handle_plain_history(runtime, session)
            if loaded is not None:
                messages = loaded
                history.reset(messages)
            continue

        if runtime.handle_slash_command(query, messages=messages, history=history):
            continue

        for event in runtime.run_turn_events(messages, goal=query, history=history):
            renderer.render(event)


def _handle_plain_history(runtime: "AgentRuntime", session: PromptSession) -> list[ChatMessage] | None:
    entries = list_history_entries(runtime.agent_dir)
    if not entries:
        print("No history found for this project.")
        return None

    for index, entry in enumerate(entries, start=1):
        print(f"{index}. {entry.label}")

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("History selection requires an interactive terminal.")
        return None

    raw = session.prompt("Load history number: ").strip()
    if not raw:
        return None
    try:
        selected = entries[int(raw) - 1]
    except (ValueError, IndexError):
        print(f"Invalid history selection: {raw}")
        return None

    try:
        messages = ensure_system_prompt(load_history_file(selected.path), runtime.system_prompt)
    except HistoryError as error:
        print(f"Failed to load history: {error}")
        return None

    print(f"Loaded history: {selected.path}")
    return messages


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
        self.subagents = SubagentManager(
            self.agent_dir,
            options.max_steps,
            self._execute_subagent_tool,
            self.current_subagent_model_profile,
        )
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
    def run_turn_events(
        self,
        messages: list[ChatMessage],
        goal: str,
        turn_id: str | None = None,
        history: HistoryRecorder | None = None,
    ) -> Iterator[AgentEvent]:
        rounds_without_todo = 0
        current_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        started_at = time.perf_counter()
        standalone_subagents_dir: Path | None = None
        tool_names_this_turn: list[str] = []
        goal_message = ChatMessage(role="user", content=goal)
        goal_already_appended = bool(messages and messages[-1].role == "user" and messages[-1].content == goal)
        if goal_already_appended:
            messages.pop()
        context_threshold = self._context_threshold()
        if estimate_tokens([*messages, goal_message]) > context_threshold:
            try:
                transcript = self._compact_messages(
                    messages,
                    "automatic threshold before user turn",
                    transcript_path=history.ensure_transcript_path() if history is not None else None,
                )
            except Exception as error:
                yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                return
            yield compact_event(
                f"Conversation compacted before this turn. Full transcript saved at: {transcript}",
                current_turn_id,
                str(transcript),
            )
        messages.append(goal_message)
        if history is not None:
            history.append(goal_message)

        for step in range(1, self.options.max_steps + 1):
            self._inject_runtime_observations(messages)
            if estimate_tokens(messages) > context_threshold:
                try:
                    transcript = self._compact_messages(
                        messages,
                        "automatic threshold during turn",
                        preserve_last_user=True,
                        transcript_path=history.ensure_transcript_path() if history is not None else None,
                    )
                except Exception as error:
                    yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                    return
                yield compact_event(
                    f"Conversation compacted during this turn. Full transcript saved at: {transcript}",
                    current_turn_id,
                    str(transcript),
                )
                self._inject_runtime_observations(messages)
            yield thinking_event(step, self.options.max_steps, current_turn_id)

            response = call_model(messages, self.current_model_profile(), tools=self.tool_specs)
            if response.tool_calls:
                assistant_message = ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)
                messages.append(assistant_message)
                if history is not None:
                    history.append(assistant_message)
                subagent_specs = []
                subagent_errors: dict[str, str] = {}
                for tool_call in response.tool_calls:
                    if tool_call.name != "subagent":
                        continue
                    try:
                        spec = parse_subagent_spec(tool_call.arguments)
                        self.subagents.validate_spec(spec)
                        subagent_specs.append((tool_call.id, spec))
                    except Exception as error:
                        subagent_errors[tool_call.id] = str(error)

                subagent_outputs: dict[str, tuple[str, bool]] = {}
                if subagent_specs or subagent_errors:
                    for tool_call in response.tool_calls:
                        if tool_call.name == "subagent":
                            action = ToolAction(name=tool_call.name, input=tool_call.arguments)
                            tool_names_this_turn.append(action.name)
                            yield tool_start_event(action.name, action.input, current_turn_id)
                if subagent_specs:
                    try:
                        if history is not None:
                            subagents_dir = history.ensure_session() / "subagents"
                        else:
                            if standalone_subagents_dir is None:
                                standalone_history = HistoryRecorder(self.agent_dir, list(messages))
                                standalone_subagents_dir = standalone_history.ensure_session() / "subagents"
                            subagents_dir = standalone_subagents_dir
                        results = self.subagents.run_batch([spec for _, spec in subagent_specs], list(messages), subagents_dir)
                        for (tool_call_id, _), result in zip(subagent_specs, results):
                            subagent_outputs[tool_call_id] = (result.output, result.record.status == "failed")
                    except Exception as error:
                        for tool_call_id, _ in subagent_specs:
                            subagent_outputs[tool_call_id] = (f"Subagent batch failed: {error}", True)
                for tool_call_id, error in subagent_errors.items():
                    subagent_outputs[tool_call_id] = (error, True)

                for tool_call in response.tool_calls:
                    action = ToolAction(name=tool_call.name, input=tool_call.arguments)
                    if action.name == "subagent":
                        output, is_error = subagent_outputs.get(tool_call.id, ("Subagent did not return a result.", True))
                    else:
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

                    tool_message = ChatMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                    messages.append(tool_message)
                    if history is not None:
                        history.append(tool_message)

                    if action.name == "compact":
                        try:
                            transcript = self._compact_messages(
                                messages,
                                "manual compact tool",
                                preserve_last_user=True,
                                transcript_path=history.ensure_transcript_path() if history is not None else None,
                            )
                        except Exception as error:
                            yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                            return
                        yield compact_event(
                            f"Conversation compacted by compact tool. Full transcript saved at: {transcript}",
                            current_turn_id,
                            str(transcript),
                        )
                        break
                continue

            final_answer = response.content.strip()
            if self.collaboration_mode == "plan" and not is_proposed_plan(final_answer):
                assistant_message = ChatMessage(role="assistant", content=final_answer)
                retry_message = ChatMessage(
                    role="system",
                    content=(
                        "Invalid final answer: Plan Mode final answers must be wrapped exactly in "
                        "<proposed_plan> and </proposed_plan>. Do not implement changes while Plan Mode is active."
                    ),
                )
                messages.append(assistant_message)
                messages.append(retry_message)
                if history is not None:
                    history.append(assistant_message)
                continue
            if tool_names_this_turn and is_acknowledgement_answer(final_answer):
                assistant_message = ChatMessage(role="assistant", content=final_answer)
                retry_message = ChatMessage(
                    role="system",
                    content=(
                        "Invalid final answer: do not acknowledge instructions or future behavior. "
                        "Answer the user's request using the latest tool result. "
                        "Do not call todo_update for acknowledgements."
                    ),
                )
                messages.append(assistant_message)
                messages.append(retry_message)
                if history is not None:
                    history.append(assistant_message)
                continue
            if self._defer_final_if_tool_needed(messages, goal, final_answer, record_assistant=True):
                if history is not None:
                    history.append(ChatMessage(role="assistant", content=final_answer))
                continue
            assistant_message = ChatMessage(role="assistant", content=final_answer)
            messages.append(assistant_message)
            if history is not None:
                history.append(assistant_message)
            yield final_event(final_answer, current_turn_id, _duration_ms(started_at))
            return

        yield stopped_event(
            f"Stopped after {self.options.max_steps} iterations without a final answer.",
            current_turn_id,
            _duration_ms(started_at),
        )
        return

    # 外部函数：处理 chat 内本地 slash command，负责 agent 主循环、chat 界面、工具分发和运行时观察。
    def handle_slash_command(
        self,
        query: str,
        emit: Callable[[str], None] = print,
        messages: list[ChatMessage] | None = None,
        history: HistoryRecorder | None = None,
    ) -> bool:
        if not query.startswith("/"):
            return False

        raw_command = query.strip()
        command = raw_command.lower()
        if command == "/help":
            emit(render_slash_help())
            return True
        if command == "/context":
            if messages is None:
                emit("Use /context inside chat to inspect the current conversation context.")
                return True
            try:
                emit(render_context_usage(messages, self.current_model_profile(), self._context_threshold()))
            except ConfigError as error:
                emit(f"Config error: {error}")
            return True
        if raw_command.split(maxsplit=1)[0].lower() == "/context":
            emit("Usage: /context")
            return True
        if command == "/diff":
            try:
                emit(render_workspace_diff(self.options.cwd, self.options.timeout_seconds, self._diff_output_max_chars()))
            except ConfigError as error:
                emit(f"Config error: {error}")
            return True
        if raw_command.split(maxsplit=1)[0].lower() == "/diff":
            emit("Usage: /diff")
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
        if command == "/history":
            emit("Use /history inside chat to choose and load a previous conversation.")
            return True
        if command == "/subagents":
            subagents_dir = history.session_dir / "subagents" if history is not None and history.session_dir is not None else None
            emit(self.subagents.render_list(subagents_dir))
            return True
        if raw_command.split(maxsplit=1)[0].lower() == "/subagents":
            emit("Usage: /subagents")
            return True
        if command == "/worktree":
            emit("Use /worktree in interactive chat to create a UE Git linked worktree.")
            return True
        if raw_command.split(maxsplit=1)[0].lower() == "/worktree":
            emit("Usage: /worktree")
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
            if messages is None:
                emit("Use /compact inside chat to compact the current conversation context.")
                return True
            try:
                transcript = self._compact_messages(
                    messages,
                    "manual slash command",
                    transcript_path=history.ensure_transcript_path() if history is not None else None,
                )
            except Exception as error:
                emit(f"Conversation compact failed: {error}")
                return True
            emit(f"Conversation compacted. Full transcript saved at: {transcript}")
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

    def _compact_messages(
        self,
        messages: list[ChatMessage],
        reason: str,
        preserve_last_user: bool = False,
        transcript_path: Path | None = None,
    ) -> Path:
        original_messages = list(messages)
        transcript = save_transcript(
            original_messages,
            transcript_path or create_standalone_session_transcript_path(self.agent_dir, original_messages),
        )
        working_messages = list(original_messages)
        micro_compact(working_messages)
        repair_tool_call_messages(working_messages)

        request = build_compaction_request(working_messages, reason)
        response = call_model(request, self.current_model_profile())
        summary = response.content.strip()
        if not summary:
            raise RuntimeError("Compaction model returned an empty summary.")

        summary_payload = f"Reason: {reason}\nFull transcript saved at: {transcript}\n\n{summary}"
        compacted = build_compacted_history(working_messages, summary_payload)
        preserved_user = latest_real_user_message(original_messages) if preserve_last_user else None
        if preserved_user is not None:
            compacted = [
                message
                for message in compacted
                if not (message.role == "user" and message.content == preserved_user.content)
            ]
            compacted.append(preserved_user)

        messages[:] = compacted
        repair_tool_call_messages(messages)
        return transcript

    # 内部函数：处理 _inject_runtime_observations 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _inject_runtime_observations(self, messages: list[ChatMessage]) -> None:
        micro_compact(messages)
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

    def _execute_subagent_tool(self, name: str, arguments: dict[str, Any]) -> str:
        output, is_error = self._execute_tool_with_status(ToolAction(name=name, input=arguments))
        if is_error:
            return f"Tool {name} failed: {output}"
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

    def current_subagent_model_profile(self):
        return resolve_subagent_model_profile(self.options.cwd, self.current_model_profile())

    def render_models(self) -> str:
        return format_model_profiles(self.options.cwd)

    def _context_threshold(self) -> int:
        if self.options.context_threshold is not None:
            return self.options.context_threshold
        return max(1, int(self.current_model_profile().context_window * 0.9))

    def _diff_output_max_chars(self) -> int:
        return load_system_config().diff_output_max_chars

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
            spec = parse_subagent_spec(tool_input)
            standalone_history = HistoryRecorder(self.agent_dir, [ChatMessage(role="system", content=self.system_prompt)])
            return self.subagents.run_batch([spec], [], standalone_history.ensure_session() / "subagents")[0].output

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

    # Resolve an optional tool working directory relative to the agent cwd.
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
