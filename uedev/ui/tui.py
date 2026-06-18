from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, Label, TextArea
from rich.console import Console
from rich.markdown import Markdown

from ..state.config import ConfigError, active_model_name, load_system_config
from ..policy.permissions import VALID_PERMISSION_MODES, permission_mode_label
from ..runtime.history import (
    HistoryEntry,
    HistoryError,
    HistoryRecorder,
    HistorySnapshot,
    ensure_system_prompt,
    list_history_entries,
    load_display_history,
    load_history_file,
)
from uedev.ui.events import stopped_event
from ..llm.client import ChatMessage
from uedev.ui.renderer import TuiRenderer, TurnViewState, _compact_lines, _event_from_dict
from ..policy.permissions import is_proposed_plan
from ..tools.shell import shell_name

if TYPE_CHECKING:
    from ..runtime.agent import AgentOptions, AgentRuntime
    from ..runtime.subagents import SubagentRecord


@dataclass
class TranscriptBlock:
    role: str
    body: str
    title: str = ""
    turn_id: str = ""
    transient: bool = False
    rendered_source: str = field(default="", init=False, repr=False)
    rendered_body: str = field(default="", init=False, repr=False)


@dataclass
class ScreenTurnSnapshot:
    block_count: int
    turn_count: int
    running: bool
    status_message: str
    sticky_scroll: bool
    modal: "ModalState | None"
    stream_block: TranscriptBlock | None


@dataclass
class ApprovalModal:
    command: str
    reason: str
    done: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    cancelled: bool = False


@dataclass
class SelectionModal:
    title: str
    labels: list[str]
    values: list[object]
    on_select: Callable[[object], None]
    selected_index: int = 0


ModalState = ApprovalModal | SelectionModal


@dataclass
class UiStateAction:
    name: str
    apply: Callable[["ChatScreenState"], None]


@dataclass
class TurnSnapshot:
    goal: str
    messages_len: int
    screen: ScreenTurnSnapshot | None
    history: HistorySnapshot


class TurnCancelled(RuntimeError):
    pass


class ChatScreenState:
    def __init__(self, banner: str, verbose: bool = False, max_output_chars: int = 240):
        self.banner = banner
        self.verbose = verbose
        self.max_output_chars = max_output_chars
        self.blocks: list[TranscriptBlock] = []
        self.turns: list[TurnViewState] = []
        self.running = False
        self.status_message = "ready"
        self.modal: ModalState | None = None
        self.input_history: list[str] = []
        self.history_index: int | None = None
        self.sticky_scroll = True
        self._stream_block: TranscriptBlock | None = None

    def print_banner(self) -> None:
        self.blocks.append(TranscriptBlock(role="system", title="uedev", body=self.banner))

    def clear(self) -> None:
        self.blocks.clear()
        self.turns.clear()
        self.running = False
        self.status_message = "ready"
        self.sticky_scroll = True
        self._stream_block = None

    def print_system(self, message: str) -> None:
        self.blocks.append(TranscriptBlock(role="system", body=message))

    def print_approval(self, command: str, reason: str) -> None:
        self.blocks.append(TranscriptBlock(role="approval required", body=f"Reason: {reason}\nCommand: {command}"))

    def start_turn(self, turn_id: str, user_message: str, *, echo_user: bool = True) -> None:
        self.turns.append(TurnViewState(turn_id=turn_id, user_message=user_message))
        self.running = True
        self.status_message = "thinking"
        self.sticky_scroll = True
        self._stream_block = None
        if echo_user:
            self.blocks.append(TranscriptBlock(role="user", body=user_message, turn_id=turn_id))

    def begin_turn_snapshot(self) -> ScreenTurnSnapshot:
        return ScreenTurnSnapshot(
            block_count=len(self.blocks),
            turn_count=len(self.turns),
            running=self.running,
            status_message=self.status_message,
            sticky_scroll=self.sticky_scroll,
            modal=self.modal,
            stream_block=self._stream_block,
        )

    def rollback_turn_snapshot(self, snapshot: ScreenTurnSnapshot) -> None:
        del self.blocks[snapshot.block_count :]
        del self.turns[snapshot.turn_count :]
        self.running = snapshot.running
        self.status_message = snapshot.status_message if snapshot.running else "ready"
        self.sticky_scroll = snapshot.sticky_scroll
        self.modal = snapshot.modal
        self._stream_block = snapshot.stream_block if snapshot.stream_block in self.blocks else None

    def render(self, event) -> None:
        turn = self._find_turn(event.turn_id)
        if turn is None:
            turn = TurnViewState(turn_id=event.turn_id or f"turn-{len(self.turns) + 1}", user_message="")
            self.turns.append(turn)

        if event.type == "assistant_delta":
            if not event.message:
                return
            if self._stream_block is None:
                self._stream_block = TranscriptBlock(role="assistant", body="", turn_id=event.turn_id, transient=True)
                self.blocks.append(self._stream_block)
            self._stream_block.body += event.message
            self.status_message = "streaming assistant"
            return

        if event.type == "thinking":
            self.status_message = _thinking_status(event)
            if self.verbose or not any(existing.type == "thinking" for existing in turn.events):
                self.blocks.append(TranscriptBlock(role="thinking", body="Thinking...", turn_id=event.turn_id))
        elif event.type == "compact":
            self.status_message = "compacted context"
            self.blocks.append(TranscriptBlock(role="system", body=event.message, turn_id=event.turn_id))
        elif event.type == "plan":
            self.status_message = event.summary or "plan"
            self.blocks.append(TranscriptBlock(role="plan", body=_plan_text(event), turn_id=event.turn_id))
        elif event.type == "tool_start":
            self.status_message = f"running {event.name}"
            self.blocks.append(TranscriptBlock(role=f"tool: {event.name}", body=_tool_start_text(event, self.verbose, self.max_output_chars), turn_id=event.turn_id))
        elif event.type == "tool_result":
            self.status_message = f"completed {event.name}"
            self.blocks.append(TranscriptBlock(role=f"tool: {event.name}", body=_tool_result_text(event, self.verbose, self.max_output_chars), turn_id=event.turn_id))
        elif event.type == "tool_error":
            self.status_message = f"failed {event.name}"
            self.blocks.append(TranscriptBlock(role=f"tool: {event.name}", body=_tool_error_text(event), turn_id=event.turn_id))
        elif event.type in {"final", "stopped"}:
            plan_already_rendered = any(existing.type == "plan" for existing in turn.events)
            if self._stream_block is not None:
                try:
                    self.blocks.remove(self._stream_block)
                except ValueError:
                    pass
                self._stream_block = None
            turn.add_event(event)
            self.running = False
            summary = turn.summary()
            self.blocks.append(TranscriptBlock(role="summary", body=summary, turn_id=event.turn_id))
            self.status_message = "ready" if event.type == "final" else event.status or "stopped"
            if event.type == "final" and plan_already_rendered and is_proposed_plan(event.message):
                return
            role = "assistant" if event.type == "final" else event.status or "stopped"
            self.blocks.append(TranscriptBlock(role=role, body=event.message, turn_id=event.turn_id))
            return

        turn.add_event(event)

    def render_history(self, messages: list[ChatMessage], source: str) -> None:
        from ..runtime.context import SUMMARY_PREFIX, is_runtime_state_message
        from ..state.plans import extract_proposed_plan_content

        self.clear()
        self.sticky_scroll = True
        self.print_system(f"Loaded history: {source}")
        for message in messages:
            if message.role == "system":
                if is_runtime_state_message(message):
                    continue
                continue
            if message.role == "user":
                content = message.content.strip()
                if not content or content.startswith("Working directory:"):
                    continue
                if content.startswith(SUMMARY_PREFIX):
                    self.blocks.append(TranscriptBlock(role="system", body=content))
                    continue
                self.blocks.append(TranscriptBlock(role="user", body=message.content))
                continue
            if message.role == "assistant":
                if message.content.strip():
                    if is_proposed_plan(message.content):
                        self.blocks.append(TranscriptBlock(role="plan", body=extract_proposed_plan_content(message.content)))
                    else:
                        self.blocks.append(TranscriptBlock(role="assistant", body=message.content))
                elif message.tool_calls:
                    names = ", ".join(tool_call.name for tool_call in message.tool_calls)
                    self.blocks.append(TranscriptBlock(role="assistant", body=f"Tool calls: {names}"))
                continue
            if message.role == "tool":
                self.blocks.append(TranscriptBlock(role=f"tool: {message.name or 'tool'}", body=message.content.strip()))

    def render_display_history(self, records: list[dict[str, object]], source: str) -> None:
        self.clear()
        self.sticky_scroll = True
        self.print_system(f"Loaded history: {source}")
        for record in records:
            record_type = record.get("type")
            if record_type == "turn_start":
                self.start_turn(str(record.get("turn_id") or ""), str(record.get("message") or ""), echo_user=True)
                continue
            if record_type == "event":
                raw_event = record.get("event")
                if isinstance(raw_event, dict):
                    self.render(_event_from_dict(raw_event))

    def render_text(self) -> str:
        return "\n\n".join(_render_block_text(block) for block in self.blocks if block.body or block.title)

    def _find_turn(self, turn_id: str) -> TurnViewState | None:
        if not self.turns:
            return None
        if not turn_id:
            return self.turns[-1]
        for turn in reversed(self.turns):
            if turn.turn_id == turn_id:
                return turn
        return None


class FullscreenRenderer:
    def __init__(self, state: ChatScreenState, submit: Callable[[UiStateAction], None]):
        self.state = state
        self.submit = submit

    def _submit(self, name: str, apply: Callable[[ChatScreenState], None]) -> None:
        self.submit(UiStateAction(name, apply))

    def print_banner(self) -> None:
        self._submit("print_banner", lambda state: state.print_banner())

    def print_system(self, message: str) -> None:
        self._submit("print_system", lambda state: state.print_system(message))

    def print_approval(self, command: str, reason: str) -> None:
        self._submit("print_approval", lambda state: state.print_approval(command, reason))

    def start_turn(self, turn_id: str, user_message: str, *, echo_user: bool = False) -> None:
        self._submit("start_turn", lambda state: state.start_turn(turn_id, user_message, echo_user=True))

    def render(self, event) -> None:
        self._submit("render_event", lambda state: state.render(event))

    def clear(self) -> None:
        self._submit("clear", lambda state: state.clear())

    def render_history(self, messages: list[ChatMessage], source: str) -> None:
        self._submit("render_history", lambda state: state.render_history(messages, source))

    def render_display_history(self, records: list[dict[str, object]], source: str) -> None:
        self._submit("render_display_history", lambda state: state.render_display_history(records, source))

    def render_text(self) -> str:
        return self.state.render_text()


def _render_block_text(block: TranscriptBlock) -> str:
    label = block.title or block.role
    if block.role in {"assistant", "plan"} and not block.transient:
        if block.rendered_source != block.body:
            block.rendered_body = _render_markdown_text(block.body)
            block.rendered_source = block.body
        body = block.rendered_body
    else:
        body = block.body
    if not label:
        return body
    if not body:
        return label
    return f"{label}\n{body}"


def _render_markdown_text(message: str) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=100, soft_wrap=True)
    try:
        console.print(Markdown(message))
    except Exception:
        return message
    rendered = stream.getvalue().rstrip()
    return rendered or message


class TranscriptLexer(Lexer):
    def lex_document(self, document: Document):
        current_style = "class:transcript.body"
        line_styles: list[tuple[str, bool]] = []
        for line in document.lines:
            stripped = line.strip()
            label_style = _label_style(stripped)
            if stripped and label_style != "class:transcript.body":
                current_style = label_style
                line_styles.append((label_style, True))
            elif not stripped:
                line_styles.append(("", False))
            else:
                line_styles.append((current_style, False))

        def get_line(line_number: int):
            style, is_label = line_styles[line_number] if line_number < len(line_styles) else ("", False)
            if is_label:
                return [(style + " bold", document.lines[line_number])]
            return [(style, document.lines[line_number])]

        return get_line


def _label_style(label: str) -> str:
    lowered = label.lower()
    if lowered in {"uedev", "system", "approval required"}:
        return "class:transcript.system"
    if lowered == "user":
        return "class:transcript.user"
    if lowered == "assistant":
        return "class:transcript.assistant"
    if lowered == "plan":
        return "class:transcript.plan"
    if lowered in {"summary", "thinking"}:
        return "class:transcript.dim"
    if lowered.startswith("tool:"):
        return "class:transcript.tool"
    if lowered in {"failed", "error", "stopped", "incomplete"}:
        return "class:transcript.error"
    return "class:transcript.body"


def create_fullscreen_style() -> Style:
    return Style.from_dict(
        {
            "text-area": "bg:#101214 fg:#d7dae0",
            "text-area.prompt": "fg:#5fafff bold",
            "prompt": "fg:#5fafff bold",
            "transcript.body": "fg:#d7dae0",
            "transcript.system": "fg:#75a7ff",
            "transcript.user": "fg:#52d6b4",
            "transcript.assistant": "fg:#d7dae0",
            "transcript.tool": "fg:#e6b450",
            "transcript.plan": "fg:#c792ea",
            "transcript.error": "fg:#ff6b6b",
            "transcript.dim": "fg:#7f8794",
            "completion-panel": "bg:#171a1f fg:#cfd6e4",
            "completion.current": "bg:#264f78 fg:#ffffff bold",
            "completion.item": "bg:#171a1f fg:#d7dae0",
            "completion.meta": "bg:#171a1f fg:#8f98a8",
            "status": "bg:#1f2329 fg:#cfd6e4",
            "status.model": "bg:#1f2329 fg:#82aaff bold",
            "status.cwd": "bg:#1f2329 fg:#c3e88d",
            "status.mode": "bg:#1f2329 fg:#ffcb6b",
            "progress": "bg:#2b2418 fg:#ffcb6b",
            "modal": "bg:#1f2329 fg:#d7dae0",
        }
    )


def _thinking_status(event) -> str:
    if event.step and event.total:
        return f"thinking {event.step}/{event.total}"
    return "thinking"


def _tool_start_text(event, verbose: bool, max_output_chars: int) -> str:
    if verbose and event.input:
        try:
            rendered = json.dumps(event.input, ensure_ascii=False, indent=2)
        except TypeError:
            rendered = str(event.input)
        return f"Running {event.name}\n{rendered[:max_output_chars]}"
    return f"Running {event.name}"


def _tool_result_text(event, verbose: bool, max_output_chars: int) -> str:
    if verbose:
        output = event.output.rstrip()
    else:
        output = _compact_lines(event.output, max_output_chars)
    if output:
        return f"OK {event.name}\n{output}"
    return f"OK {event.name} completed"


def _tool_error_text(event) -> str:
    detail = (event.message or event.output).rstrip()
    if detail:
        return f"Failed {event.name}\n{detail}"
    return f"Failed {event.name}"


def _plan_text(event) -> str:
    lines = [
        event.summary or "Proposed plan",
        f"status: {event.status or 'pending'}",
    ]
    if event.output:
        lines.append(f"path: {event.output}")
    if event.message:
        lines.extend(["", event.message])
    return "\n".join(lines)


class ChatTuiApplication:
    def __init__(
        self,
        options: "AgentOptions",
        runtime: "AgentRuntime",
        banner: str,
        completer: Completer,
        input: Input | None = None,
        output: Output | None = None,
    ):
        self.options = options
        self.runtime = runtime
        self.banner = banner
        self.completer = completer
        self.input = input
        self.output = output
        self.renderer = TuiRenderer(banner=banner, verbose=options.verbose)
        self.runtime.approval_provider = self.confirm_command
        self.messages = self._initial_messages()
        self.history = HistoryRecorder(self.runtime.agent_dir, self.messages)
        self.current_subagent: "SubagentRecord | None" = None
        self.screen: ChatScreenState | None = None
        self._fullscreen_app: Application | None = None
        self._transcript_area: TextArea | None = None
        self._input_area: TextArea | None = None
        self._ui_events: queue.SimpleQueue[UiStateAction] = queue.SimpleQueue()
        self._ui_lock = threading.RLock()
        self._invalidate_lock = threading.Lock()
        self._last_invalidate_at = 0.0
        self._invalidate_timer: threading.Timer | None = None
        self._invalidate_pending = False
        self._invalidate_count = 0
        self._suppress_completion_once = False
        self._cancel_requested = False
        self._cancel_input_restored = False
        self._active_turn_snapshot: TurnSnapshot | None = None
        self._pending_worktree_name = False

    def run(self) -> None:
        if self._should_use_legacy_prompt_loop():
            self.run_prompt_loop()
            return
        self.run_fullscreen()

    def _should_use_legacy_prompt_loop(self) -> bool:
        return self.input is not None or self.output is not None or not (sys.stdin.isatty() and sys.stdout.isatty())

    def run_prompt_loop(self) -> None:
        from ..runtime.agent import create_chat_prompt_options, create_chat_session

        session = create_chat_session(
            completer=self.completer,
            input=self.input,
            output=self.output,
            key_bindings=self.create_key_bindings(),
        )
        self.renderer.print_banner()

        while True:
            try:
                prompt_options = create_chat_prompt_options()
                prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
                query = session.prompt([("class:prompt", "\n> ")], **prompt_options).strip()
            except (EOFError, KeyboardInterrupt):
                return

            if query.lower() in {"", "quit", "exit"}:
                return

            if query.lower() == "/clear":
                self.messages = self._initial_messages()
                self.history.reset(self.messages)
                self.renderer.clear()
                self.renderer.print_system("Conversation context cleared.")
                continue

            if query.lower() == "/history":
                selected = self.prompt_history_selection(session)
                if selected is not None:
                    self.load_history(selected)
                continue

            if query.lower() == "/subagents":
                selected = self.prompt_subagent_selection(session)
                if selected == "main":
                    self.load_main_conversation()
                elif selected is not None:
                    self.load_subagent(selected)
                continue

            if query.lower() == "/worktree":
                name = self.prompt_worktree_name(session)
                if name is not None:
                    self.create_ue_linked_worktree(name)
                continue

            if query.lower() == "/model":
                selected = self.prompt_model_selection(session)
                if selected is not None:
                    try:
                        self.renderer.print_system(self.runtime.switch_model(selected))
                    except ConfigError as error:
                        self.renderer.print_system(f"Config error: {error}")
                continue

            if query.lower().startswith("/model "):
                self.renderer.print_system("Use /model and choose a profile with the arrow keys.")
                continue

            if query.lower() == "/permissions":
                selected = self.prompt_permission_mode(session)
                if selected is None:
                    continue
                query = selected

            if self.current_subagent is not None and not query.startswith("/"):
                self.renderer.print_system(
                    f"Subagent {self.current_subagent.id} is {self.current_subagent.status} and closed. "
                    "Use /subagents to switch back to the main conversation."
                )
                continue

            if self.runtime.handle_slash_command(query, emit=self.renderer.print_system, messages=self.messages, history=self.history):
                continue

            self._run_turn(query)

    def run_fullscreen(self) -> None:
        self.screen = ChatScreenState(self.banner, verbose=self.options.verbose)
        self.renderer = FullscreenRenderer(self.screen, self._enqueue_ui_action)  # type: ignore[assignment]
        self.runtime.approval_provider = self.confirm_command
        self.renderer.print_banner()

        self._transcript_area = TextArea(
            text=self.screen.render_text(),
            read_only=True,
            lexer=TranscriptLexer(),
            scrollbar=True,
            wrap_lines=True,
            focusable=False,
        )
        self._input_area = TextArea(
            height=1,
            prompt=[("class:prompt", "> ")],
            multiline=False,
            completer=self.completer,
            complete_while_typing=True,
            wrap_lines=False,
        )
        self._input_area.buffer.on_text_changed += self._on_fullscreen_input_changed

        progress_line = ConditionalContainer(
            content=Label(self._progress_fragments, style="class:progress"),
            filter=Condition(lambda: bool(self.screen and (self.screen.running or self.screen.status_message != "ready"))),
        )
        completion_panel = ConditionalContainer(
            content=Window(
                content=FormattedTextControl(self._completion_fragments),
                height=Dimension(min=1, max=8),
                style="class:completion-panel",
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self._completion_visible()),
        )
        status_line = Label(self._fullscreen_status_fragments)
        root = FloatContainer(
            content=HSplit(
                [
                    self._transcript_area,
                    progress_line,
                    completion_panel,
                    self._input_area,
                    status_line,
                ]
            ),
            floats=[
                Float(
                    content=ConditionalContainer(
                        content=Frame(Label(self._modal_fragments), title="uedev"),
                        filter=Condition(lambda: bool(self.screen and self.screen.modal is not None)),
                    ),
                    top=2,
                    left=4,
                    right=4,
                )
            ],
        )

        self._fullscreen_app = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=self.create_fullscreen_key_bindings(),
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.25,
            style=create_fullscreen_style(),
            input=self.input,
            output=self.output,
            before_render=lambda app: self._sync_fullscreen_controls(),
        )
        self._fullscreen_app.run()

    def _request_refresh(self) -> None:
        self._schedule_invalidate()

    def _enqueue_ui_action(self, action: UiStateAction) -> None:
        self._ui_events.put(action)
        self._request_refresh()

    def _schedule_invalidate(self) -> None:
        delay = 0.033
        app = self._fullscreen_app
        if app is None:
            return
        now = time.monotonic()
        with self._invalidate_lock:
            elapsed = now - self._last_invalidate_at
            if elapsed >= delay:
                self._last_invalidate_at = now
                self._invalidate_count += 1
                app.invalidate()
                return
            if self._invalidate_pending:
                return
            self._invalidate_pending = True
            wait = max(0.0, delay - elapsed)
            timer = threading.Timer(wait, self._flush_throttled_invalidate)
            timer.daemon = True
            self._invalidate_timer = timer
            timer.start()

    def _flush_throttled_invalidate(self) -> None:
        app = self._fullscreen_app
        with self._invalidate_lock:
            self._invalidate_pending = False
            self._last_invalidate_at = time.monotonic()
            self._invalidate_count += 1
        if app is not None:
            app.invalidate()

    def _drain_ui_events(self) -> None:
        if self.screen is None:
            return
        while True:
            try:
                action = self._ui_events.get_nowait()
            except queue.Empty:
                return
            with self._ui_lock:
                action.apply(self.screen)

    def _sync_fullscreen_controls(self) -> None:
        self._drain_ui_events()
        if self.screen is None or self._transcript_area is None:
            return
        with self._ui_lock:
            rendered = self.screen.render_text()
            previous_row = self._transcript_cursor_row()
            if self._transcript_area.text != rendered:
                self._transcript_area.text = rendered
                self._set_transcript_cursor_for_render(previous_row)
            elif self.screen.sticky_scroll:
                self._transcript_area.buffer.cursor_position = len(rendered)

    def _transcript_cursor_row(self) -> int:
        if self._transcript_area is None:
            return 0
        document = Document(self._transcript_area.text, cursor_position=self._transcript_area.buffer.cursor_position)
        return document.cursor_position_row

    def _set_transcript_cursor_for_render(self, previous_row: int) -> None:
        if self.screen is None or self._transcript_area is None:
            return
        text = self._transcript_area.text
        if self.screen.sticky_scroll:
            self._transcript_area.buffer.cursor_position = len(text)
            return
        document = Document(text, cursor_position=0)
        row = max(0, min(document.line_count - 1, previous_row))
        self._transcript_area.buffer.cursor_position = document.translate_row_col_to_index(row, 0)

    def _progress_fragments(self):
        if self.screen is None:
            return ""
        return f" {self.screen.status_message} "

    def _fullscreen_status_fragments(self):
        fragments = self.status_fragments()
        permission = self.runtime.permission_mode.replace("_", "-")
        extra = f"   {permission}"
        return [(style or "class:status", text) for style, text in [*fragments, ("class:status.mode", extra)]]

    def _modal_fragments(self):
        if self.screen is None or self.screen.modal is None:
            return ""
        modal = self.screen.modal
        if isinstance(modal, ApprovalModal):
            return (
                "Approval required\n\n"
                f"Reason: {modal.reason}\n"
                f"Command: {modal.command}\n\n"
                "Press y to approve, n or Esc to reject."
            )
        visible = []
        for index, label in enumerate(modal.labels):
            marker = "> " if index == modal.selected_index else "  "
            visible.append(f"{marker}{label}")
        return f"{modal.title}\n\n" + "\n".join(visible) + "\n\nEnter selects. Esc closes."

    def _on_fullscreen_input_changed(self, _buffer) -> None:
        if self._input_area is None:
            return
        if self._suppress_completion_once:
            self._suppress_completion_once = False
            self._request_refresh()
            return
        buffer = self._input_area.buffer
        text = buffer.document.text_before_cursor
        if text.startswith("/") and buffer.complete_state is None:
            try:
                buffer.start_completion(select_first=False)
            except Exception:
                pass
        elif not text.startswith("/") and buffer.complete_state is not None:
            buffer.cancel_completion()
        self._request_refresh()

    def _completion_visible(self) -> bool:
        if self.screen is not None and self.screen.modal is not None:
            return False
        if self._input_area is None:
            return False
        buffer = self._input_area.buffer
        state = buffer.complete_state
        return bool(buffer.document.text_before_cursor.startswith("/") and state is not None and state.completions)

    def _completion_fragments(self):
        if self._input_area is None:
            return []
        state = self._input_area.buffer.complete_state
        if state is None or not state.completions:
            return []
        selected_index = state.complete_index if state.complete_index is not None else 0
        fragments: list[tuple[str, str]] = []
        start = max(0, min(selected_index - 4, max(0, len(state.completions) - 8)))
        visible = state.completions[start : start + 8]
        for offset, completion in enumerate(visible):
            index = start + offset
            current = index == selected_index
            item_style = "class:completion.current" if current else "class:completion.item"
            meta_style = "class:completion.current" if current else "class:completion.meta"
            marker = "> " if current else "  "
            meta = completion.display_meta_text
            fragments.append((item_style, marker + completion.display_text))
            if meta:
                fragments.append((meta_style, "  " + meta))
            fragments.append(("", "\n"))
        if fragments:
            fragments.pop()
        return fragments

    def _accept_completion(self) -> bool:
        if self._input_area is None:
            return False
        state = self._input_area.buffer.complete_state
        if state is None or not state.completions:
            return False
        completion = state.current_completion or state.completions[0]
        self._suppress_completion_once = True
        self._input_area.buffer.apply_completion(completion)
        self._request_refresh()
        return True

    def _move_completion(self, delta: int) -> bool:
        if self._input_area is None:
            return False
        state = self._input_area.buffer.complete_state
        if state is None or not state.completions:
            return False
        if delta < 0:
            self._input_area.buffer.complete_previous(disable_wrap_around=True)
        else:
            self._input_area.buffer.complete_next(disable_wrap_around=True)
        self._request_refresh()
        return True

    def _cancel_completion(self) -> bool:
        if self._input_area is None or self._input_area.buffer.complete_state is None:
            return False
        self._input_area.buffer.cancel_completion()
        self._request_refresh()
        return True

    def create_fullscreen_key_bindings(self) -> KeyBindings:
        bindings = self.create_key_bindings()

        @bindings.add("enter")
        def _submit(event) -> None:
            if self._handle_modal_submit():
                self._request_refresh()
                return
            if self._accept_completion():
                return
            self._submit_fullscreen_input()

        @bindings.add("tab")
        def _tab(event) -> None:
            if self._accept_completion():
                return
            if self._input_area is not None:
                self._input_area.buffer.start_completion(select_first=True)
                self._request_refresh()

        @bindings.add("c-c")
        def _cancel_or_clear(event) -> None:
            if self.screen is not None and self.screen.running:
                self._request_turn_cancel()
                return
            if self._input_area is not None:
                self._input_area.buffer.reset()
                self._request_refresh()

        @bindings.add("c-q")
        @bindings.add("c-d")
        def _exit(event) -> None:
            if self.screen is not None and self.screen.running:
                self._request_turn_cancel()
                return
            event.app.exit()

        @bindings.add("escape")
        def _escape(event) -> None:
            if self.screen is not None and self.screen.running:
                self._request_turn_cancel()
                return
            if self.screen is not None and self.screen.modal is not None:
                modal = self.screen.modal
                self.screen.modal = None
                if isinstance(modal, ApprovalModal):
                    modal.approved = False
                    modal.done.set()
                self._request_refresh()
                return
            if self._cancel_completion():
                return

        @bindings.add("y", filter=Condition(lambda: self._approval_modal_active()))
        def _approve(event) -> None:
            self._finish_approval(True)
            self._request_refresh()

        @bindings.add("n", filter=Condition(lambda: self._approval_modal_active()))
        def _reject(event) -> None:
            self._finish_approval(False)
            self._request_refresh()

        @bindings.add("up")
        def _up(event) -> None:
            if self._move_selection(-1):
                self._request_refresh()
                return
            if self._move_completion(-1):
                return

        @bindings.add("down")
        def _down(event) -> None:
            if self._move_selection(1):
                self._request_refresh()
                return
            if self._move_completion(1):
                return

        @bindings.add(Keys.ControlUp)
        def _history_up(event) -> None:
            self._browse_input_history(-1)

        @bindings.add(Keys.ControlDown)
        def _history_down(event) -> None:
            self._browse_input_history(1)

        @bindings.add(Keys.ScrollUp)
        def _scroll_up(event) -> None:
            self._scroll_transcript(-3)

        @bindings.add(Keys.ScrollDown)
        def _scroll_down(event) -> None:
            self._scroll_transcript(3)

        @bindings.add("pageup")
        def _page_up(event) -> None:
            self._scroll_transcript(-10)

        @bindings.add("pagedown")
        def _page_down(event) -> None:
            self._scroll_transcript(10)

        @bindings.add("home")
        def _home(event) -> None:
            if self._transcript_area is not None:
                self._transcript_area.buffer.cursor_position = 0
            if self.screen is not None:
                self.screen.sticky_scroll = False
            self._request_refresh()

        @bindings.add("end")
        def _end(event) -> None:
            if self._transcript_area is not None:
                self._transcript_area.buffer.cursor_position = len(self._transcript_area.text)
            if self.screen is not None:
                self.screen.sticky_scroll = True
            self._request_refresh()

        return bindings

    def _request_turn_cancel(self) -> None:
        self._cancel_requested = True
        with self._ui_lock:
            if self.screen is not None:
                if isinstance(self.screen.modal, ApprovalModal):
                    modal = self.screen.modal
                    modal.cancelled = True
                    modal.approved = False
                    self.screen.modal = None
                    modal.done.set()
                self.screen.status_message = "canceling..."
        self._restore_cancelled_input()
        self._request_refresh()

    def _restore_cancelled_input(self) -> None:
        snapshot = self._active_turn_snapshot
        if snapshot is None or self._input_area is None:
            return
        self._input_area.buffer.document = Document(snapshot.goal, cursor_position=len(snapshot.goal))
        self._cancel_input_restored = True

    def _approval_modal_active(self) -> bool:
        return bool(self.screen is not None and isinstance(self.screen.modal, ApprovalModal))

    def _finish_approval(self, approved: bool) -> None:
        if self.screen is None or not isinstance(self.screen.modal, ApprovalModal):
            return
        modal = self.screen.modal
        modal.approved = approved
        self.screen.modal = None
        modal.done.set()

    def _handle_modal_submit(self) -> bool:
        if self.screen is None or self.screen.modal is None:
            return False
        modal = self.screen.modal
        if isinstance(modal, ApprovalModal):
            return True
        if not modal.values:
            self.screen.modal = None
            return True
        value = modal.values[modal.selected_index]
        self.screen.modal = None
        modal.on_select(value)
        return True

    def _move_selection(self, delta: int) -> bool:
        if self.screen is None or not isinstance(self.screen.modal, SelectionModal):
            return False
        modal = self.screen.modal
        if not modal.labels:
            return True
        modal.selected_index = max(0, min(len(modal.labels) - 1, modal.selected_index + delta))
        return True

    def _browse_input_history(self, delta: int) -> None:
        if self.screen is None or self._input_area is None:
            return
        if self._input_area.text.strip():
            return
        if not self.screen.input_history:
            return
        if self.screen.history_index is None:
            self.screen.history_index = len(self.screen.input_history)
        self.screen.history_index = max(0, min(len(self.screen.input_history) - 1, self.screen.history_index + delta))
        self._input_area.buffer.document = Document(self.screen.input_history[self.screen.history_index], cursor_position=len(self.screen.input_history[self.screen.history_index]))
        self._request_refresh()

    def _scroll_transcript(self, delta_rows: int) -> None:
        self._drain_ui_events()
        if self._transcript_area is None:
            return
        text = self._transcript_area.text
        document = Document(text, cursor_position=self._transcript_area.buffer.cursor_position)
        row = max(0, min(document.line_count - 1, document.cursor_position_row + delta_rows))
        self._transcript_area.buffer.cursor_position = document.translate_row_col_to_index(row, 0)
        if self.screen is not None:
            self.screen.sticky_scroll = row >= document.line_count - 1
        self._request_refresh()

    def _submit_fullscreen_input(self) -> None:
        if self.screen is None or self._input_area is None:
            return
        query = self._input_area.text.strip()
        if not query:
            return
        if self.screen.running:
            self.screen.status_message = "turn already running"
            self._request_refresh()
            return
        self._cancel_completion()
        self._input_area.buffer.reset()
        self.screen.input_history.append(query)
        self.screen.history_index = None
        self.screen.sticky_scroll = True
        self._handle_fullscreen_query(query)

    def _handle_fullscreen_query(self, query: str) -> None:
        lowered = query.lower()
        if self._pending_worktree_name:
            self._pending_worktree_name = False
            self.create_ue_linked_worktree(query)
            return
        if lowered in {"quit", "exit"}:
            if self._fullscreen_app is not None:
                self._fullscreen_app.exit()
            return
        if lowered == "/clear":
            self.messages = self._initial_messages()
            self.history.reset(self.messages)
            self.renderer.clear()
            self.renderer.print_system("Conversation context cleared.")
            return
        if lowered == "/history":
            self._open_history_modal()
            return
        if lowered == "/subagents":
            self._open_subagent_modal()
            return
        if lowered == "/worktree":
            self._pending_worktree_name = True
            self.renderer.print_system("Enter worktree name, then press Enter.")
            return
        if lowered == "/model":
            self._open_model_modal()
            return
        if lowered.startswith("/model "):
            self.renderer.print_system("Use /model and choose a profile with the arrow keys.")
            return
        if lowered == "/permissions":
            self._open_permission_modal()
            return
        if self.current_subagent is not None and not query.startswith("/"):
            self.renderer.print_system(
                f"Subagent {self.current_subagent.id} is {self.current_subagent.status} and closed. "
                "Use /subagents to switch back to the main conversation."
            )
            return
        if self.runtime.handle_slash_command(query, emit=self.renderer.print_system, messages=self.messages, history=self.history):
            return
        self._run_turn_async(query)

    def _open_history_modal(self) -> None:
        entries = list_history_entries(self.runtime.agent_dir)
        if not entries:
            self.renderer.print_system("No history found for this project.")
            return
        labels = [f"{index}. {entry.label}" for index, entry in enumerate(entries, start=1)]

        def select(value: object) -> None:
            if isinstance(value, HistoryEntry):
                self.load_history(value)

        self._set_selection_modal("History", labels, list(entries), select)

    def _open_subagent_modal(self) -> None:
        subagents_dir = self.history.session_dir / "subagents" if self.history.session_dir is not None else None
        records = self.runtime.subagents.list_records(subagents_dir)
        if not records:
            self.renderer.print_system(self.runtime.subagents.render_list(subagents_dir))
            return
        labels = ["Main conversation", *[f"{index}. {record.label}" for index, record in enumerate(records, start=1)]]
        values: list[object] = ["main", *records]

        def select(value: object) -> None:
            if value == "main":
                self.load_main_conversation()
            else:
                self.load_subagent(value)  # type: ignore[arg-type]

        self._set_selection_modal("Subagents", labels, values, select)

    def _open_model_modal(self) -> None:
        try:
            config = load_system_config()
            active = active_model_name(self.options.cwd, config)
        except ConfigError as error:
            self.renderer.print_system(f"Config error: {error}")
            return

        labels: list[str] = []
        values: list[object] = []
        for name, profile in sorted(config.models.items()):
            markers: list[str] = []
            if name == active:
                markers.append("active")
            if name == config.default_model:
                markers.append("default")
            if profile.gpt_model:
                markers.append("responses")
            if profile.requires_reasoning_content:
                markers.append("reasoning")
            suffix = f" ({', '.join(markers)})" if markers else ""
            labels.append(f"{name} - {profile.model or '(missing model)'}{suffix}")
            values.append(name)
        labels.append(f"Reset to default ({config.default_model})")
        values.append("reset")

        def select(value: object) -> None:
            try:
                self.renderer.print_system(self.runtime.switch_model(str(value)))
            except ConfigError as error:
                self.renderer.print_system(f"Config error: {error}")

        self._set_selection_modal("Model", labels, values, select)

    def _open_permission_modal(self) -> None:
        labels = [f"/permissions {permission_mode_label(mode)}" for mode in VALID_PERMISSION_MODES]
        values: list[object] = labels

        def select(value: object) -> None:
            try:
                self.renderer.print_system(self.runtime.handle_permissions_command(str(value)))
            except ConfigError as error:
                self.renderer.print_system(f"Config error: {error}")

        self._set_selection_modal("Permissions", labels, values, select)

    def _set_selection_modal(
        self,
        title: str,
        labels: list[str],
        values: list[object],
        on_select: Callable[[object], None],
    ) -> None:
        if self.screen is None:
            return
        self.screen.modal = SelectionModal(title=title, labels=labels, values=values, on_select=on_select)
        self._request_refresh()

    def _run_turn_async(self, goal: str) -> None:
        snapshot = self._create_turn_snapshot(goal)
        self._active_turn_snapshot = snapshot
        self._cancel_requested = False
        self._cancel_input_restored = False
        thread = threading.Thread(target=self._run_turn_worker, args=(goal, snapshot), daemon=True)
        thread.start()

    def _run_turn_worker(self, goal: str, snapshot: TurnSnapshot) -> None:
        try:
            self._run_turn(goal, rollback_snapshot=snapshot)
        finally:
            if self._active_turn_snapshot is snapshot:
                self._active_turn_snapshot = None

    def _create_turn_snapshot(self, goal: str) -> TurnSnapshot:
        screen_snapshot = self.screen.begin_turn_snapshot() if self.screen is not None else None
        return TurnSnapshot(
            goal=goal,
            messages_len=len(self.messages),
            screen=screen_snapshot,
            history=self.history.snapshot(),
        )

    def create_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _exit_plan_mode(event) -> None:
            if self.exit_plan_mode():
                event.app.invalidate()

        return bindings

    def exit_plan_mode(self) -> bool:
        if self.runtime.collaboration_mode != "plan":
            return False
        self.runtime.collaboration_mode = "default"
        return True

    def status_fragments(self):
        model = self._status_model_name()
        directory = str(self.options.cwd)
        right = ""
        if self.current_subagent is not None:
            right = f"Viewing {self.current_subagent.id} "
        elif self.runtime.collaboration_mode == "plan":
            right = "Plan mode "
        left_length = len(model) + 3 + len(directory)
        right_length = len(right)
        width = self._terminal_width()
        fragments = [
            ("class:status.model", model),
            ("class:status", "   "),
            ("class:status.cwd", directory),
        ]
        if right:
            fragments.append(("class:status", " " * max(1, width - left_length - right_length)))
            fragments.append(("class:status.mode", right))
        return fragments

    def status_bottom_toolbar(self):
        return self.status_fragments()

    def prompt_permission_mode(self, session: PromptSession) -> str | None:
        from ..runtime.agent import create_chat_prompt_options

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\n> ")],
                default="/permissions ",
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return selected or None

    def prompt_history_selection(self, session: PromptSession) -> HistoryEntry | None:
        from ..runtime.agent import create_chat_prompt_options

        entries = list_history_entries(self.runtime.agent_dir)
        if not entries:
            self.renderer.print_system("No history found for this project.")
            return None

        labels = [f"{index}. {entry.label}" for index, entry in enumerate(entries, start=1)]
        by_label = {label: entry for label, entry in zip(labels, entries)}

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\nHistory> ")],
                completer=WordCompleter(labels, ignore_case=True, sentence=True),
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not selected:
            return None
        entry = by_label.get(selected)
        if entry is None:
            self.renderer.print_system(f"Unknown history selection: {selected}")
        return entry

    def prompt_subagent_selection(self, session: PromptSession):
        from ..runtime.agent import create_chat_prompt_options

        subagents_dir = self.history.session_dir / "subagents" if self.history.session_dir is not None else None
        records = self.runtime.subagents.list_records(subagents_dir)
        if not records:
            self.renderer.print_system(self.runtime.subagents.render_list(subagents_dir))
            return None
        labels = ["Main conversation", *[f"{index}. {record.label}" for index, record in enumerate(records, start=1)]]
        by_label = {"Main conversation": "main"}
        by_label.update({label: record for label, record in zip(labels[1:], records)})

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\nSubagents> ")],
                completer=WordCompleter(labels, ignore_case=True, sentence=True),
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not selected:
            return None
        choice = by_label.get(selected)
        if choice is None:
            self.renderer.print_system(f"Unknown subagent selection: {selected}")
        return choice

    def prompt_worktree_name(self, session: PromptSession) -> str | None:
        from ..runtime.agent import create_chat_prompt_options

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt([("class:prompt", "\nWorktree name> ")], **prompt_options).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return selected or None

    def prompt_model_selection(self, session: PromptSession) -> str | None:
        from ..runtime.agent import create_chat_prompt_options

        try:
            config = load_system_config()
            active = active_model_name(self.options.cwd, config)
        except ConfigError as error:
            self.renderer.print_system(f"Config error: {error}")
            return None

        labels: list[str] = []
        by_label: dict[str, str] = {}
        for name, profile in sorted(config.models.items()):
            markers: list[str] = []
            if name == active:
                markers.append("active")
            if name == config.default_model:
                markers.append("default")
            if profile.gpt_model:
                markers.append("responses")
            if profile.requires_reasoning_content:
                markers.append("reasoning")
            suffix = f" ({', '.join(markers)})" if markers else ""
            label = f"{name} - {profile.model or '(missing model)'}{suffix}"
            labels.append(label)
            by_label[label] = name

        reset_label = f"Reset to default ({config.default_model})"
        labels.append(reset_label)
        by_label[reset_label] = "reset"

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\nModel> ")],
                completer=WordCompleter(labels, ignore_case=True, sentence=True),
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not selected:
            return None
        choice = by_label.get(selected)
        if choice is None:
            self.renderer.print_system(f"Unknown model selection: {selected}")
        return choice

    def create_ue_linked_worktree(self, name: str) -> None:
        try:
            default_root = load_system_config().worktree_default_root
            session_dir = self.history.ensure_session()
            result = self.runtime.worktrees.create_ue_git_linked(name, default_root=default_root, session_dir=session_dir)
        except Exception as error:
            self.renderer.print_system(f"Failed to create UE linked worktree: {error}")
            return
        self.renderer.print_system(result)

    def load_history(self, entry: HistoryEntry) -> None:
        try:
            messages = ensure_system_prompt(load_history_file(entry.path), self.runtime.system_prompt)
        except HistoryError as error:
            self.renderer.print_system(f"Failed to load history: {error}")
            return
        display_records = self._load_display_records(entry.display_path)
        if not display_records:
            display_records = self.runtime.plan_display_records_for_session(entry.session_dir)
        self.messages = messages
        self.history.resume(entry, self.messages)
        self.current_subagent = None
        if display_records:
            self.renderer.render_display_history(display_records, str(entry.path))
        else:
            self.renderer.render_history(self.messages, str(entry.path))

    def load_subagent(self, record: "SubagentRecord") -> None:
        try:
            messages = self.runtime.subagents.load_messages(record)
        except HistoryError as error:
            self.renderer.print_system(f"Failed to load subagent history: {error}")
            return
        display_path = getattr(record, "display_history_path", "") or ""
        display_records = self._load_display_records(Path(display_path) if display_path else None)
        self.current_subagent = record
        source = f"subagent {record.id}: {record.history_path}"
        if display_records:
            self.renderer.render_display_history(display_records, source)
        else:
            self.renderer.render_history(messages, source)

    def load_main_conversation(self) -> None:
        self.current_subagent = None
        messages = self.messages
        source = "main conversation"
        display_records = self.history.initial_display_records
        if self.history.path is not None:
            try:
                messages = ensure_system_prompt(load_history_file(self.history.path), self.runtime.system_prompt)
                source = str(self.history.path)
                display_records = self._load_display_records(self.history.display_path)
                if not display_records:
                    display_records = self.runtime.plan_display_records_for_session(self.history.session_dir)
            except HistoryError as error:
                self.renderer.print_system(f"Failed to load main conversation history: {error}")
        if display_records:
            self.renderer.render_display_history(display_records, source)
        else:
            self.renderer.render_history(messages, source)

    def _load_display_records(self, path: Path | None) -> list[dict[str, object]]:
        if path is None or not path.exists():
            return []
        try:
            return load_display_history(path)
        except HistoryError as error:
            self.renderer.print_system(f"Failed to load display history: {error}")
            return []

    def _status_model_name(self) -> str:
        try:
            profile = self.runtime.current_model_profile()
        except ConfigError:
            return "(missing config)"
        return profile.model or profile.name or "(missing model)"

    def plan_mode_bottom_toolbar(self):
        return self.status_fragments()

    def _terminal_width(self) -> int:
        if self.output is not None:
            try:
                return int(self.output.get_size().columns)
            except Exception:
                pass
        return shutil.get_terminal_size((80, 20)).columns

    def confirm_command(self, command: str, reason: str) -> bool:
        if self.screen is not None and self._fullscreen_app is not None:
            modal = ApprovalModal(command=command, reason=reason)
            def show_modal(state: ChatScreenState) -> None:
                state.modal = modal
                state.status_message = "approval required"

            self._enqueue_ui_action(UiStateAction("approval_modal", show_modal))
            modal.done.wait()
            approved = modal.approved
            if modal.cancelled:
                return False
            self.renderer.print_system("Approved." if approved else "Rejected.")
            return approved
        self.renderer.print_approval(command, reason)
        session_kwargs = {}
        if self.input is not None:
            session_kwargs["input"] = self.input
        if self.output is not None:
            session_kwargs["output"] = self.output
        session = PromptSession(**session_kwargs)
        answer = session.prompt("Approve? [y/N] ").strip().lower()
        approved = answer == "y"
        self.renderer.print_system("Approved." if approved else "Rejected.")
        return approved

    def _initial_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.runtime.system_prompt),
            ChatMessage(role="user", content=f"Working directory: {self.options.cwd}\nShell: {shell_name()}"),
        ]

    def _run_turn(self, goal: str, rollback_snapshot: TurnSnapshot | None = None) -> None:
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        self.renderer.start_turn(turn_id, goal)
        self.history.record_turn_start(turn_id, goal)
        try:
            for event in self.runtime.run_turn_events(self.messages, goal=goal, turn_id=turn_id, history=self.history):
                if self._cancel_requested:
                    raise TurnCancelled()
                self.renderer.render(event)
                if event.type not in {"plan", "assistant_delta"}:
                    self.history.record_event(event)
        except TurnCancelled:
            if rollback_snapshot is not None:
                self._rollback_cancelled_turn(rollback_snapshot)
        except Exception as error:
            if self._cancel_requested and rollback_snapshot is not None:
                self._rollback_cancelled_turn(rollback_snapshot)
                return
            event = stopped_event(f"Error: {error}", turn_id=turn_id)
            self.renderer.render(event)
            self.history.record_event(event)
        finally:
            self._cancel_requested = False
            self._cancel_input_restored = False

    def _rollback_cancelled_turn(self, snapshot: TurnSnapshot) -> None:
        del self.messages[snapshot.messages_len :]
        self.history.restore(snapshot.history)

        def rollback_screen(state: ChatScreenState) -> None:
            if snapshot.screen is not None:
                state.rollback_turn_snapshot(snapshot.screen)
            state.running = False
            state.status_message = "ready"
            state.modal = None
            state.sticky_scroll = True

        self._enqueue_ui_action(UiStateAction("rollback_cancelled_turn", rollback_screen))
        if not self._cancel_input_restored:
            def restore_input(_state: ChatScreenState) -> None:
                if self._input_area is not None:
                    self._input_area.buffer.document = Document(snapshot.goal, cursor_position=len(snapshot.goal))
                    self._cancel_input_restored = True

            self._enqueue_ui_action(UiStateAction("restore_cancelled_input", restore_input))
