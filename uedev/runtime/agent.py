from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..tools.background import BackgroundManager
from ..state.config import (
    ConfigError,
    DEFAULT_WORKSPACE_EXCLUDED_DIRS,
    RuntimeBudgetConfig,
    SystemConfig,
    VALID_REASONING_EFFORTS,
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
from .completion import (
    SLASH_COMMANDS,
    SlashCommandCompleter,
    create_chat_prompt_options,
    create_chat_session,
    create_chat_style,
    render_slash_help,
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
    assistant_delta_event,
    budget_event,
    compact_event,
    final_event,
    incomplete_event,
    plan_event,
    stopped_event,
    thinking_event,
    tool_error_event,
    tool_result_event,
    tool_start_event,
    usage_event,
)
from .history import (
    HistoryError,
    HistoryRecorder,
    SessionModelState,
    create_standalone_session_transcript_path,
    ensure_system_prompt,
    list_history_entries,
    load_session_metadata,
    load_history_file,
    load_transcript_token_usage,
    summarize_token_usage,
    update_session_active_plan,
)
from .options import AgentOptions
from ..llm.client import (
    ChatMessage,
    ModelResponse,
    ModelStreamEvent,
    TokenUsage,
    call_model as _client_call_model,
    call_model_stream as _client_call_model_stream,
)
from .tool_guard import ToolHandler, _permission_prompt_label
from .turn_loop import (
    ToolAction,
    TurnBudgetState,
    _duration_ms,
    defers_tool_confirmation,
    is_acknowledgement_answer,
    truncate,
)
from ..mcp.registry import McpToolRegistry
from ..policy.permissions import (
    CollaborationMode,
    PermissionMode,
    VALID_PERMISSION_MODES,
    classify_tool_permission,
    format_permission_modes,
    is_proposed_plan,
    normalize_permission_mode,
    permission_mode_label,
)
from .prompts import PromptBundle, build_prompt_bundle, build_system_prompt as render_system_prompt
from .subagents import SubagentManager, parse_subagent_spec
from ..ui.renderer import ConsoleRenderer
from ..tools.shell import ApprovalProvider, confirm_command, run_shell, shell_name
from .skills import SkillLoader
from ..state.plans import (
    PlanManager,
    PlanRecord,
    extract_proposed_plan_content,
    plan_record_from_dict,
)
from ..state.tasks import TaskManager, TodoManager
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
    render_build_result,
    render_doctor,
    render_run_result,
    run_ue_build,
)
from ..tools.workspace import edit_file, grep, list_files, read_file, write_file
from ..tools.worktrees import WorktreeManager


RUNTIME_STATE_MARKER = "<runtime-state>"
call_model = _client_call_model


def call_model_stream(
    messages: list[ChatMessage],
    profile: Any,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[ModelStreamEvent]:
    if call_model is not _client_call_model:
        yield ModelStreamEvent(type="final", response=call_model(messages, profile, tools=tools))
        return
    yield from _client_call_model_stream(messages, profile, tools=tools)


FORCED_FINALIZATION_PROMPT = """You have reached the tool and step budget for this turn.

Do not call tools. Provide the final answer now. Summarize what changed, what failed, and what remains. If work is incomplete, say so clearly and mention the next concrete step.

If Plan Mode is active, the final answer must be wrapped exactly in <proposed_plan> and </proposed_plan>."""


def render_context_usage(
    messages: list[ChatMessage],
    model_profile: Any,
    compact_threshold: int,
    budget_config: RuntimeBudgetConfig | None = None,
) -> str:
    estimated = estimate_tokens(messages)
    context_window = int(getattr(model_profile, "context_window", 0) or 0)
    model_name = str(getattr(model_profile, "model", "") or "(missing model)")
    profile_name = str(getattr(model_profile, "name", "") or "(unknown)")

    lines = [
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
    if budget_config is not None:
        lines.extend(
            [
                "",
                "Budgets:",
                f"model requests: {_format_number(budget_config.model_request_hard_limit)}",
                f"tool calls soft limit: {_format_number(budget_config.tool_call_soft_limit)}",
                f"wall clock: {_format_number(budget_config.wall_clock_seconds)}s",
                f"consecutive tool failures: {_format_number(budget_config.consecutive_tool_failures)}",
                f"permission denials: {_format_number(budget_config.permission_denials)}",
                f"no-progress rounds: {_format_number(budget_config.no_progress_rounds)}",
                f"context compact ratio: {budget_config.context_compact_ratio:.2f}",
                f"output soft ratio: {budget_config.output_token_soft_ratio:.2f}",
            ]
        )
    return "\n".join(lines)


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
        if query.lower() in {"", "quit", "exit", "/exit"}:
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

    restored = runtime.restore_history_model_state(selected.model_state)
    print(f"Loaded history: {selected.path}")
    if restored:
        print(restored)
    return messages


class AgentRuntime:
    # 内部函数：初始化当前类实例，准备 agent 主循环、chat 界面、工具分发和运行时观察 所需状态。
    def __init__(self, options: AgentOptions, approval_provider: ApprovalProvider | None = None):
        self.options = options
        self.budget_config = options.runtime_budget or RuntimeBudgetConfig(model_request_hard_limit=options.max_steps)
        self.approval_provider = approval_provider or confirm_command
        self.agent_dir = agent_dir(options.cwd)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        project_config = load_project_config(options.cwd)
        self.collaboration_mode: CollaborationMode = "default"
        self.permission_mode: PermissionMode = "full_access" if options.auto_approve else project_config.permission_mode
        self.effort_overrides: dict[str, str] = {}
        self.todo_manager = TodoManager(self.agent_dir)
        self.task_manager = TaskManager(self.agent_dir / "tasks")
        self.plan_manager = PlanManager()
        self.active_plan: PlanRecord | None = None
        self.skill_loader = SkillLoader(options.cwd / "skills")
        self.background = BackgroundManager(options.cwd)
        self.worktrees = WorktreeManager(options.cwd, self.agent_dir / "worktrees", self.task_manager)
        self.workspace_excluded_dirs = self._load_workspace_excluded_dirs()
        self.subagents = SubagentManager(
            self.agent_dir,
            self.budget_config.model_request_hard_limit,
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
        emit_budget_events: bool = False,
    ) -> Iterator[AgentEvent]:
        rounds_without_todo = 0
        current_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        budget = TurnBudgetState(self.budget_config, time.perf_counter())
        started_at = budget.started_at
        standalone_subagents_dir: Path | None = None
        tool_names_this_turn: list[str] = []
        tool_events_this_turn: list[AgentEvent] = []
        edited_paths_this_turn: list[str] = []
        goal_message = ChatMessage(role="user", content=goal)
        goal_already_appended = bool(messages and messages[-1].role == "user" and messages[-1].content == goal)
        if goal_already_appended:
            messages.pop()
        if history is not None:
            self.record_history_model_state(history)
        context_threshold = self._context_threshold()
        if estimate_tokens([*messages, goal_message]) > context_threshold:
            try:
                transcript, compact_usage = self._compact_messages(
                    messages,
                    "automatic threshold before user turn",
                    transcript_path=history.ensure_transcript_path() if history is not None else None,
                    history=history,
                    turn_id=current_turn_id,
                )
            except Exception as error:
                yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                return
            yield compact_event(
                f"Conversation compacted before this turn. Full transcript saved at: {transcript}",
                current_turn_id,
                str(transcript),
            )
            if compact_usage is not None:
                yield compact_usage
        messages.append(goal_message)
        if history is not None:
            history.append(goal_message)

        for step in range(1, budget.config.model_request_hard_limit + 1):
            if budget.wall_clock_exceeded():
                yield incomplete_event(
                    self._render_incomplete_summary(
                        f"Stopped after reaching the wall clock budget of {budget.config.wall_clock_seconds}s.",
                        tool_events_this_turn,
                        edited_paths_this_turn,
                    ),
                    current_turn_id,
                    budget.elapsed_ms(),
                )
                return
            self._inject_runtime_observations(messages)
            if estimate_tokens(messages) > context_threshold:
                try:
                    transcript, compact_usage = self._compact_messages(
                        messages,
                        "automatic threshold during turn",
                        preserve_last_user=True,
                        transcript_path=history.ensure_transcript_path() if history is not None else None,
                        history=history,
                        turn_id=current_turn_id,
                    )
                except Exception as error:
                    yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                    return
                yield compact_event(
                    f"Conversation compacted during this turn. Full transcript saved at: {transcript}",
                    current_turn_id,
                    str(transcript),
                )
                if compact_usage is not None:
                    yield compact_usage
                self._inject_runtime_observations(messages)
            budget.next_model_request()
            yield thinking_event(step, budget.config.model_request_hard_limit, current_turn_id)
            if emit_budget_events:
                yield budget_event(budget.status("requesting model"), current_turn_id, budget.elapsed_ms(), budget.status("requesting model"))

            try:
                response: ModelResponse | None = None
                request_profile = self.current_model_profile()
                for model_event in call_model_stream(messages, request_profile, tools=self.tool_specs):
                    if model_event.type == "delta":
                        if model_event.delta:
                            budget.total_output_tokens += _estimate_text_tokens(model_event.delta)
                            yield assistant_delta_event(model_event.delta, current_turn_id)
                        continue
                    response = model_event.response
                if response is None:
                    raise RuntimeError("Model stream ended without a final response.")
                token_event = self._record_token_usage(
                    response.usage,
                    request_profile,
                    current_turn_id,
                    "main",
                    history,
                )
                if token_event is not None:
                    yield token_event
            except Exception as error:
                yield stopped_event(str(error), current_turn_id, _duration_ms(started_at))
                return
            budget.total_output_tokens += _estimate_text_tokens(response.content)
            if response.tool_calls:
                assistant_message = ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
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
                        results = self.subagents.run_batch(
                            [spec for _, spec in subagent_specs],
                            list(messages),
                            subagents_dir,
                            parent_turn_id=current_turn_id,
                        )
                        for (tool_call_id, _), result in zip(subagent_specs, results):
                            subagent_outputs[tool_call_id] = (result.output, result.record.status == "failed")
                    except Exception as error:
                        for tool_call_id, _ in subagent_specs:
                            subagent_outputs[tool_call_id] = (f"Subagent batch failed: {error}", True)
                for tool_call_id, error in subagent_errors.items():
                    subagent_outputs[tool_call_id] = (error, True)

                for tool_call in response.tool_calls:
                    action = ToolAction(name=tool_call.name, input=tool_call.arguments)
                    tool_call_count = budget.record_tool_call(action.name)
                    tool_limit = budget.tool_limit_for(action.name)
                    limit_exceeded = tool_limit is not None and tool_call_count > tool_limit
                    if emit_budget_events:
                        yield budget_event(budget.status(f"running {action.name}"), current_turn_id, budget.elapsed_ms(), budget.status(f"running {action.name}"))
                    if action.name == "subagent":
                        if limit_exceeded:
                            output, is_error = _tool_limit_message(action.name, tool_call_count, tool_limit or 0), True
                        else:
                            output, is_error = subagent_outputs.get(tool_call.id, ("Subagent did not return a result.", True))
                    else:
                        tool_names_this_turn.append(action.name)
                        yield tool_start_event(action.name, action.input, current_turn_id)
                        if limit_exceeded:
                            output, is_error = _tool_limit_message(action.name, tool_call_count, tool_limit or 0), True
                        elif action.name in {"edit_file", "write_file"}:
                            path = str(action.input.get("path") or "").strip()
                            if path and path not in edited_paths_this_turn:
                                edited_paths_this_turn.append(path)
                            output, is_error = self._execute_tool_with_status(action)
                        else:
                            output, is_error = self._execute_tool_with_status(action)
                    permission_denied = _is_permission_denial_output(output)
                    if permission_denied:
                        is_error = True
                    if action.name == "todo_update":
                        rounds_without_todo = 0
                    else:
                        rounds_without_todo += 1
                    progress = not is_error and bool((output or "").strip())
                    if action.name in {"edit_file", "write_file"} and not is_error:
                        progress = True
                    budget.record_tool_result(is_error=is_error, permission_denied=permission_denied, progress=progress)

                    if is_error:
                        event = tool_error_event(action.name, output, current_turn_id)
                    else:
                        event = tool_result_event(action.name, output, current_turn_id)
                    tool_events_this_turn.append(event)
                    yield event

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
                    self._append_budget_reminders(messages, budget)

                    if action.name == "compact":
                        try:
                            transcript, compact_usage = self._compact_messages(
                                messages,
                                "manual compact tool",
                                preserve_last_user=True,
                                transcript_path=history.ensure_transcript_path() if history is not None else None,
                                history=history,
                                turn_id=current_turn_id,
                            )
                        except Exception as error:
                            yield stopped_event(f"Conversation compact failed: {error}", current_turn_id, _duration_ms(started_at))
                            return
                        yield compact_event(
                            f"Conversation compacted by compact tool. Full transcript saved at: {transcript}",
                            current_turn_id,
                            str(transcript),
                        )
                        if compact_usage is not None:
                            yield compact_usage
                        break
                continue

            final_answer = response.content.strip()
            if self.collaboration_mode == "plan" and not is_proposed_plan(final_answer):
                assistant_message = ChatMessage(role="assistant", content=final_answer, reasoning_content=response.reasoning_content)
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
                assistant_message = ChatMessage(role="assistant", content=final_answer, reasoning_content=response.reasoning_content)
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
            if self._defer_final_if_tool_needed(
                messages,
                goal,
                final_answer,
                record_assistant=True,
                reasoning_content=response.reasoning_content,
            ):
                if history is not None:
                    history.append(ChatMessage(role="assistant", content=final_answer, reasoning_content=response.reasoning_content))
                continue
            assistant_message = ChatMessage(role="assistant", content=final_answer, reasoning_content=response.reasoning_content)
            messages.append(assistant_message)
            if history is not None:
                history.append(assistant_message)
            if self.collaboration_mode == "plan" and is_proposed_plan(final_answer):
                proposed_plan_event = self._persist_proposed_plan(final_answer, current_turn_id, history)
                if history is not None:
                    history.record_event(proposed_plan_event)
                yield proposed_plan_event
            yield final_event(final_answer, current_turn_id, _duration_ms(started_at))
            return

        yield self._force_finalize_turn(
            messages,
            current_turn_id,
            started_at,
            history,
            tool_events_this_turn,
            edited_paths_this_turn,
        )
        return

    def _force_finalize_turn(
        self,
        messages: list[ChatMessage],
        turn_id: str,
        started_at: float,
        history: HistoryRecorder | None,
        tool_events: list[AgentEvent],
        edited_paths: list[str],
    ) -> AgentEvent:
        request = [
            *messages,
            ChatMessage(role="system", content=FORCED_FINALIZATION_PROMPT),
        ]
        profile = self.current_model_profile()
        try:
            response = call_model(request, profile)
        except Exception as error:
            return incomplete_event(
                self._render_incomplete_summary(
                    f"Stopped after {self.budget_config.model_request_hard_limit} model requests without a final answer. "
                    f"Forced finalization failed: {error}",
                    tool_events,
                    edited_paths,
                ),
                turn_id,
                _duration_ms(started_at),
            )

        final_answer = response.content.strip()
        if response.tool_calls:
            return incomplete_event(
                self._render_incomplete_summary(
                    f"Stopped after {self.budget_config.model_request_hard_limit} model requests without a final answer. "
                    "Forced finalization returned tool calls instead of a final answer.",
                    tool_events,
                    edited_paths,
                ),
                turn_id,
                _duration_ms(started_at),
            )
        if not final_answer:
            return incomplete_event(
                self._render_incomplete_summary(
                    f"Stopped after {self.budget_config.model_request_hard_limit} model requests without a final answer. "
                    "Forced finalization returned an empty answer.",
                    tool_events,
                    edited_paths,
                ),
                turn_id,
                _duration_ms(started_at),
            )
        if self.collaboration_mode == "plan" and not is_proposed_plan(final_answer):
            return incomplete_event(
                self._render_incomplete_summary(
                    f"Stopped after {self.budget_config.model_request_hard_limit} model requests without a valid Plan Mode final answer.",
                    tool_events,
                    edited_paths,
                ),
                turn_id,
                _duration_ms(started_at),
            )

        assistant_message = ChatMessage(
            role="assistant",
            content=final_answer,
            reasoning_content=response.reasoning_content,
        )
        messages.append(assistant_message)
        if history is not None:
            history.append(assistant_message)
        token_event = self._record_token_usage(response.usage, profile, turn_id, "forced_finalization", history)
        return final_event(
            final_answer,
            turn_id,
            _duration_ms(started_at),
            usage=token_event.usage if token_event is not None else None,
        )

    def _render_incomplete_summary(
        self,
        reason: str,
        tool_events: list[AgentEvent],
        edited_paths: list[str],
    ) -> str:
        lines = [
            "Incomplete turn.",
            "",
            f"Reason: {reason}",
            f"Model request budget: {self.budget_config.model_request_hard_limit}",
            f"Tools completed: {len(tool_events)}",
        ]
        if edited_paths:
            lines.append(f"Edited files: {', '.join(edited_paths[:8])}")
            if len(edited_paths) > 8:
                lines.append(f"... ({len(edited_paths) - 8} more edited files)")
        if tool_events:
            last = tool_events[-1]
            label = "Last tool error" if last.type == "tool_error" else "Last tool result"
            detail = last.message or last.output
            lines.extend(["", f"{label}: {last.name}", truncate(detail, 1200)])
        lines.extend(["", "Continue with a new turn to resume from the current workspace state."])
        return "\n".join(lines)

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
                emit(render_context_usage(messages, self.current_model_profile(), self._context_threshold(), self.budget_config))
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
        if command == "/history":
            emit("Use /history inside chat to choose and load a previous conversation.")
            return True
        if command == "/usage":
            emit(self.render_token_usage(history))
            return True
        if raw_command.split(maxsplit=1)[0].lower() == "/usage":
            emit("Usage: /usage")
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
        if command == "/effort" or command.startswith("/effort "):
            result = self.handle_effort_command(raw_command)
            emit(result)
            if history is not None and result.startswith(("Reasoning effort set", "Reasoning effort reset")):
                self.record_history_model_state(history)
            return True
        if command == "/mcp":
            emit(self.mcp.render_status())
            return True
        if command.startswith("/model "):
            try:
                result = self.switch_model(raw_command.split(maxsplit=1)[1].strip())
                emit(result)
                if history is not None and result.startswith("Active model"):
                    self.record_history_model_state(history)
            except ConfigError as error:
                emit(f"Config error: {error}")
            return True
        if command == "/plan" or command.startswith("/plan "):
            emit(self.handle_plan_command(raw_command, history=history))
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
                transcript, compact_usage = self._compact_messages(
                    messages,
                    "manual slash command",
                    transcript_path=history.ensure_transcript_path() if history is not None else None,
                    history=history,
                    turn_id=f"compact-{uuid.uuid4().hex[:8]}",
                )
            except Exception as error:
                emit(f"Conversation compact failed: {error}")
                return True
            emit(f"Conversation compacted. Full transcript saved at: {transcript}")
            if history is not None and compact_usage is not None:
                history.record_event(compact_usage)
            return True

        emit(f"Unknown slash command: {query}")
        return True

    def handle_plan_command(self, raw_command: str, history: HistoryRecorder | None = None) -> str:
        arg = raw_command.split(maxsplit=1)[1].strip().lower() if len(raw_command.split(maxsplit=1)) > 1 else ""
        if arg in {"", "on"}:
            self.collaboration_mode = "plan"
            return "Plan Mode enabled. Use Shift+Tab or /plan off to exit."
        if arg in {"off", "default"}:
            self.collaboration_mode = "default"
            return "Plan Mode disabled."
        if arg == "status":
            return self._render_plan_status(history)
        if arg == "approve":
            return self._review_active_plan("approved", history)
        if arg == "reject":
            return self._review_active_plan("rejected", history)
        return "Usage: /plan, /plan off, /plan status, /plan approve, or /plan reject"

    def _persist_proposed_plan(
        self,
        final_answer: str,
        turn_id: str,
        history: HistoryRecorder | None,
    ) -> AgentEvent:
        content = extract_proposed_plan_content(final_answer)
        session_id = "standalone"
        session_dir: Path | None = None
        if history is not None:
            session_dir = history.ensure_session()
            session_id = session_dir.name
        record = self.plan_manager.save_proposed_plan(session_id, turn_id, content)
        self.active_plan = record
        if session_dir is not None:
            update_session_active_plan(session_dir, record.to_dict())
        return plan_event(content, record.path, record.title, record.status, turn_id)

    def _render_plan_status(self, history: HistoryRecorder | None = None) -> str:
        record = self._load_active_plan(history)
        lines = [
            f"Collaboration mode: {self.collaboration_mode}",
            f"Plan directory: {self.plan_manager.plans_dir}",
        ]
        if record is None:
            lines.append("Active plan: none")
            return "\n".join(lines)
        lines.extend(
            [
                f"Active plan: {record.title}",
                f"Status: {record.status}",
                f"Path: {record.path}",
            ]
        )
        return "\n".join(lines)

    def _review_active_plan(self, status: str, history: HistoryRecorder | None) -> str:
        if status not in {"approved", "rejected"}:
            raise ValueError(f"unsupported plan status: {status}")
        record = self._load_active_plan(history)
        if record is None:
            return "No active plan."
        content, readable_record = self.plan_manager.read_content(record)
        if readable_record.status == "missing":
            self.active_plan = readable_record
            if history is not None and history.session_dir is not None:
                update_session_active_plan(history.session_dir, readable_record.to_dict())
                history.record_event(
                    plan_event(content, readable_record.path, readable_record.title, readable_record.status, readable_record.turn_id)
                )
            return f"Active plan file is missing: {readable_record.path}"

        updated = self.plan_manager.with_status(record, status)  # type: ignore[arg-type]
        self.active_plan = updated
        if history is not None and history.session_dir is not None:
            update_session_active_plan(history.session_dir, updated.to_dict())
            history.record_event(plan_event(content, updated.path, updated.title, updated.status, updated.turn_id))
        if status == "approved":
            self.collaboration_mode = "default"
            return f"Plan approved: {updated.path}"
        self.collaboration_mode = "plan"
        return f"Plan rejected: {updated.path}"

    def _load_active_plan(self, history: HistoryRecorder | None = None) -> PlanRecord | None:
        if history is not None and history.session_dir is not None:
            try:
                metadata = load_session_metadata(history.session_dir)
            except HistoryError:
                metadata = {}
            record = plan_record_from_dict(metadata.get("active_plan"))
            if record is not None:
                self.active_plan = record
                return record
        return self.active_plan

    def plan_display_records_for_session(self, session_dir: Path | None) -> list[dict[str, object]]:
        if session_dir is None:
            return []
        try:
            metadata = load_session_metadata(session_dir)
        except HistoryError:
            return []
        record = plan_record_from_dict(metadata.get("active_plan"))
        if record is None:
            return []
        content, visible_record = self.plan_manager.read_content(record)
        event = plan_event(
            content,
            visible_record.path,
            visible_record.title,
            visible_record.status,
            visible_record.turn_id,
        )
        return [{"type": "event", "event": event.__dict__}]

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
        history: HistoryRecorder | None = None,
        turn_id: str = "",
    ) -> tuple[Path, AgentEvent | None]:
        original_messages = list(messages)
        transcript = save_transcript(
            original_messages,
            transcript_path or create_standalone_session_transcript_path(self.agent_dir, original_messages),
        )
        working_messages = list(original_messages)
        micro_compact(working_messages)
        profile = self.current_model_profile()
        repair_tool_call_messages(working_messages, require_reasoning_content=profile.requires_reasoning_content)

        request = build_compaction_request(working_messages, reason)
        response = call_model(request, profile)
        token_event = self._record_token_usage(response.usage, profile, turn_id, "compact", history)
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
        repair_tool_call_messages(messages, require_reasoning_content=profile.requires_reasoning_content)
        return transcript, token_event

    # 内部函数：处理 _inject_runtime_observations 辅助逻辑，支撑 agent 主循环、chat 界面、工具分发和运行时观察。
    def _inject_runtime_observations(self, messages: list[ChatMessage]) -> None:
        micro_compact(messages)
        repair_tool_call_messages(
            messages,
            require_reasoning_content=self.current_model_profile().requires_reasoning_content,
        )
        self._inject_runtime_state(messages)

        notifications = self.background.drain()
        if notifications:
            rendered = "\n".join(f"[bg:{task.id}] {task.status}\n{truncate(task.result, 1000)}" for task in notifications)
            messages.append(ChatMessage(role="user", content=f"<background-results>\n{rendered}\n</background-results>"))

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
        reasoning_content: str | None = None,
    ) -> bool:
        if defers_tool_confirmation(goal, answer):
            if record_assistant:
                messages.append(ChatMessage(role="assistant", content=answer, reasoning_content=reasoning_content))
            messages.append(ChatMessage(role="system", content=self.prompt_bundle.tool_confirmation_reminder))
            return True
        return False

    def current_model_profile(self):
        profile = resolve_model_profile(self.options.cwd)
        override = self.effort_overrides.get(profile.name)
        return replace(profile, effort=override) if override is not None else profile

    def current_subagent_model_profile(self):
        return resolve_subagent_model_profile(self.options.cwd, self.current_model_profile())

    def render_models(self) -> str:
        return format_model_profiles(self.options.cwd)

    def effort_levels(self) -> tuple[str, ...]:
        profile = resolve_model_profile(self.options.cwd)
        if profile.response or profile.effort or "deepseek" in profile.model.casefold():
            return VALID_REASONING_EFFORTS
        return ()

    def current_effort(self) -> str:
        return self.current_model_profile().effort or "auto"

    def default_effort(self) -> str:
        return resolve_model_profile(self.options.cwd).effort or "auto"

    def record_history_model_state(self, history: HistoryRecorder) -> None:
        profile = self.current_model_profile()
        history.update_model_state(profile.name, profile.model, profile.effort)

    def _record_token_usage(
        self,
        usage: TokenUsage | None,
        profile: Any,
        turn_id: str,
        purpose: str,
        history: HistoryRecorder | None,
    ) -> AgentEvent | None:
        if usage is None:
            return None
        payload: dict[str, object] = {
            "request_id": f"req_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "turn_id": turn_id,
            "purpose": purpose,
            "profile_name": profile.name,
            "model": profile.model,
            "api_mode": "responses" if profile.response else "chat_completions",
            "effort": profile.effort,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "source": usage.source,
        }
        if history is not None:
            history.record_token_usage(payload)
        return usage_event(payload, turn_id)

    def render_token_usage(self, history: HistoryRecorder | None) -> str:
        path = history.transcript_path if history is not None else None
        if path is None or not path.exists():
            return "No token usage recorded for the current session."
        records = load_transcript_token_usage(path)
        if not records:
            return "No token usage recorded for the current session."
        total = summarize_token_usage(records)
        latest_turn_id = next((str(record.get("turn_id") or "") for record in reversed(records) if record.get("turn_id")), "")
        latest = [record for record in records if str(record.get("turn_id") or "") == latest_turn_id]
        latest_total = summarize_token_usage(latest)
        lines = ["Session token usage", self._format_token_usage_summary(total)]
        if latest:
            lines.extend([f"Latest turn: {latest_turn_id}", self._format_token_usage_summary(latest_total), "Requests:"])
            for index, record in enumerate(latest, start=1):
                source = f" {record.get('source')}" if record.get("source") == "estimated" else ""
                lines.append(
                    f"{index}. {record.get('purpose') or 'model'}: "
                    f"{int(record.get('total_tokens') or 0):,} total "
                    f"({int(record.get('input_tokens') or 0):,} in / "
                    f"{int(record.get('output_tokens') or 0):,} out){source}"
                )
        return "\n".join(lines)

    @staticmethod
    def _format_token_usage_summary(summary: dict[str, int]) -> str:
        source = f" · {summary['estimated_requests']} estimated" if summary["estimated_requests"] else ""
        return (
            f"{summary['requests']} requests · {summary['total_tokens']:,} total · "
            f"{summary['input_tokens']:,} in · {summary['output_tokens']:,} out · "
            f"{summary['cached_input_tokens']:,} cached · {summary['reasoning_tokens']:,} reasoning{source}"
        )

    def restore_history_model_state(self, state: SessionModelState | None) -> str | None:
        if state is None:
            return None
        config = load_system_config()
        matches = [name for name, profile in config.models.items() if state.model and profile.model == state.model]
        profile_name: str | None = matches[0] if len(matches) == 1 else None
        if profile_name is None and state.profile_name:
            folded = state.profile_name.casefold()
            display_matches = [name for name in config.models if name.casefold() == folded]
            if len(display_matches) == 1:
                profile_name = display_matches[0]
        if profile_name is None:
            saved = state.model or state.profile_name
            return f"History model {saved!r} is not available; kept {self.current_model_profile().name}."

        profile = config.models[profile_name]
        self._save_active_profile(profile_name, config)
        if state.effort is None:
            self.effort_overrides.pop(profile_name, None)
        elif state.effort in VALID_REASONING_EFFORTS:
            self.effort_overrides[profile_name] = state.effort
        else:
            self.effort_overrides.pop(profile_name, None)
            return (
                f"Restored model {profile.name}, but saved effort {state.effort!r} is invalid; "
                f"using {profile.effort or 'auto'}."
            )
        return f"Restored model {profile.name} with effort {self.current_effort()}."

    def handle_effort_command(self, raw_command: str) -> str:
        levels = self.effort_levels()
        profile = resolve_model_profile(self.options.cwd)
        if not levels:
            return f"Model {profile.name} does not support configurable reasoning effort."
        parts = raw_command.split(maxsplit=1)
        if len(parts) == 1:
            return (
                f"Reasoning effort for {profile.name}: {self.current_effort()}\n"
                f"Available levels: {', '.join(levels)}"
            )
        effort = parts[1].strip().lower()
        if effort == "reset":
            self.effort_overrides.pop(profile.name, None)
            return f"Reasoning effort reset to {profile.effort or 'auto'} for {profile.name}."
        if effort not in levels:
            return f"Unknown effort: {effort}\nAvailable levels: {', '.join(levels)}"
        self.effort_overrides[profile.name] = effort
        return f"Reasoning effort set to {effort} for {profile.name}."

    def _append_budget_reminders(self, messages: list[ChatMessage], budget: TurnBudgetState) -> None:
        reminders: list[str] = []
        if budget.tool_calls >= budget.config.tool_call_soft_limit and not budget.tool_soft_limit_reminded:
            budget.tool_soft_limit_reminded = True
            reminders.append(
                "Tool call soft budget reached. Stop exploring unless absolutely necessary; summarize results or produce the final answer."
            )
        output_limit = self._output_token_soft_limit()
        if output_limit is not None and budget.total_output_tokens >= output_limit and not budget.output_soft_limit_reminded:
            budget.output_soft_limit_reminded = True
            reminders.append(
                "Output token soft budget is nearly used. Keep the next response concise and finish the turn."
            )
        if budget.consecutive_tool_failures >= budget.config.consecutive_tool_failures:
            reminders.append(
                "Several tools have failed consecutively. Change strategy or stop and explain the blocker instead of repeating the same attempt."
            )
            budget.consecutive_tool_failures = 0
        if budget.permission_denials >= budget.config.permission_denials:
            reminders.append(
                "Permission was denied repeatedly. Do not retry the same restricted action; ask for a different approach or provide a final explanation."
            )
            budget.permission_denials = 0
        if budget.no_progress_rounds >= budget.config.no_progress_rounds and not budget.no_progress_reminded:
            budget.no_progress_reminded = True
            reminders.append(
                "No meaningful progress has been observed for several tool rounds. Converge on a final answer or choose a clearly different strategy."
            )
        for reminder in reminders:
            messages.append(ChatMessage(role="system", content=f"<budget-reminder>{reminder}</budget-reminder>"))

    def _output_token_soft_limit(self) -> int | None:
        profile = self.current_model_profile()
        max_output = None
        responses = getattr(profile, "responses", {}) or {}
        if isinstance(responses, dict):
            max_output = responses.get("max_output_tokens")
        if isinstance(max_output, int) and max_output > 0:
            return max(1, int(max_output * self.budget_config.output_token_soft_ratio))
        context_window = int(getattr(profile, "context_window", 0) or 0)
        if context_window <= 0:
            return None
        return max(1, int(context_window * 0.1 * self.budget_config.output_token_soft_ratio))

    def _context_threshold(self) -> int:
        if self.options.context_threshold is not None:
            return self.options.context_threshold
        return max(1, int(self.current_model_profile().context_window * self.budget_config.context_compact_ratio))

    def _diff_output_max_chars(self) -> int:
        return load_system_config().diff_output_max_chars

    def _load_workspace_excluded_dirs(self) -> tuple[str, ...]:
        try:
            return load_system_config().workspace_excluded_dirs
        except ConfigError as error:
            if "Config file not found" in str(error):
                return DEFAULT_WORKSPACE_EXCLUDED_DIRS
            raise

    def switch_model(self, name: str) -> str:
        if not name:
            return self.render_models()
        if name.lower() == "reset":
            reset_project_active_model(self.options.cwd)
            profile = resolve_model_profile(self.options.cwd)
            return f"Active model reset to default profile {profile.name}"

        config = load_system_config()
        if name not in config.models:
            available = ", ".join(sorted(config.models)) or "(none)"
            return f"Unknown model profile: {name}\nAvailable profiles: {available}"
        profile = config.models[name]
        self._save_active_profile(name, config)
        return f"Active model set to {profile.name}"

    def _save_active_profile(self, name: str, config: SystemConfig) -> None:
        profile = config.models[name]
        # Persist the API model identifier so changing the outer profile key
        # (the CLI display name) does not invalidate the project selection.
        model_is_unique = bool(profile.model) and sum(
            candidate.model == profile.model for candidate in config.models.values()
        ) == 1
        save_project_active_model(self.options.cwd, profile.model if model_is_unique else name)

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
                    results.append(edit_file(self.options.cwd, path, old_text, new_text, excluded_dirs=self.workspace_excluded_dirs))
                return "\n".join(results)

            old_text = str(tool_input.get("old_text", tool_input.get("oldText", "")))
            new_text = str(tool_input.get("new_text", tool_input.get("newText", "")))
            return edit_file(self.options.cwd, path, old_text, new_text, excluded_dirs=self.workspace_excluded_dirs)

        def grep_tool(tool_input: dict[str, object]) -> str:
            limit = _optional_int(tool_input.get("limit"))
            return grep(
                self.options.cwd,
                str(tool_input.get("pattern", "")),
                str(tool_input.get("path", ".")),
                _optional_str(tool_input.get("glob")),
                limit if limit is not None else 100,
                case_sensitive=_optional_bool(tool_input.get("case_sensitive"), default=True),
                output_mode=str(tool_input.get("output_mode") or "content"),
                include_asset_paths=_optional_bool(tool_input.get("include_asset_paths"), default=True),
                excluded_dirs=self.workspace_excluded_dirs,
            )

        # 内部函数：处理 subagent 工具调用，启动受限子 agent 完成子任务。
        def subagent_tool(tool_input: dict[str, object]) -> str:
            spec = parse_subagent_spec(tool_input)
            standalone_history = HistoryRecorder(self.agent_dir, [ChatMessage(role="system", content=self.system_prompt)])
            return self.subagents.run_batch([spec], [], standalone_history.ensure_session() / "subagents")[0].output

        handlers: dict[str, ToolHandler] = {
            "shell": shell_tool,
            "read_file": lambda data: read_file(
                self.options.cwd,
                str(data.get("path", "")),
                _optional_int(data.get("limit")),
                excluded_dirs=self.workspace_excluded_dirs,
            ),
            "write_file": lambda data: write_file(
                self.options.cwd,
                str(data.get("path", "")),
                str(data.get("content", "")),
                excluded_dirs=self.workspace_excluded_dirs,
            ),
            "edit_file": edit_file_tool,
            "list_files": lambda data: list_files(
                self.options.cwd,
                str(data.get("path", ".")),
                str(data.get("pattern", "*")),
                int(data.get("limit", 200)),
                excluded_dirs=self.workspace_excluded_dirs,
            ),
            "grep": grep_tool,
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
            "ue_build": lambda data: render_build_result(
                run_ue_build(
                    self._resolve_tool_cwd(data.get("cwd")),
                    agent_dir(self._resolve_tool_cwd(data.get("cwd"))),
                    timeout_seconds=_optional_int(data.get("timeout_seconds")) or 1800,
                )
            ),
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

def _estimate_text_tokens(value: str) -> int:
    if not value:
        return 0
    return max(1, len(value) // 4)


def _tool_limit_message(name: str, observed: int, limit: int) -> str:
    return (
        f"Tool call limit reached for {name}: attempted call {observed}, "
        f"limit is {limit} for this turn. Choose another strategy or finalize."
    )


def _is_permission_denial_output(output: str) -> bool:
    normalized = output.strip().lower()
    return "the user rejected that action" in normalized or "tool denied by policy" in normalized


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


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


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
