from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

try:
    from rich import box
    from rich.console import Console, Group, RenderableType
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - only used in minimal test environments.
    RenderableType = object

    class _Box:
        ASCII = None

    box = _Box()

    class Text:
        def __init__(self, text: str = "", style: str | None = None) -> None:
            self.parts = [text] if text else []

        def append(self, text: str, style: str | None = None) -> None:
            self.parts.append(text)

        def __str__(self) -> str:
            return "".join(self.parts)

    class Markdown(Text):
        pass

    class Syntax(Text):
        def __init__(self, code: str, lexer: str, word_wrap: bool = False) -> None:
            super().__init__(code)

    class Panel:
        def __init__(self, renderable: object, **kwargs: object) -> None:
            self.renderable = renderable

        def __str__(self) -> str:
            return str(self.renderable)

    class Rule(Text):
        pass

    class Group:
        def __init__(self, *items: object) -> None:
            self.items = items

        def __str__(self) -> str:
            return "\n".join(str(item) for item in self.items)

    class Console:
        def __init__(self, file: TextIO | None = None, **kwargs: object) -> None:
            self.file = file or sys.stdout

        def print(self, renderable: object) -> None:
            print(str(renderable), file=self.file)

from uedev.ui.events import AgentEvent
from ..policy.permissions import is_proposed_plan
from ..runtime.context import SUMMARY_PREFIX, is_runtime_state_message
from ..state.plans import extract_proposed_plan_content
from ..llm.client import ChatMessage


@dataclass
class TurnViewState:
    turn_id: str
    user_message: str
    events: list[AgentEvent] = field(default_factory=list)
    collapsed: bool = False
    final_answer: str = ""

    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)
        if event.type in {"final", "stopped"}:
            self.final_answer = event.message
            self.collapsed = True

    def summary(self) -> str:
        steps = max((event.step for event in self.events), default=0)
        tool_events = [event for event in self.events if event.type in {"tool_result", "tool_error"}]
        edited_files = sum(1 for event in tool_events if event.name in {"edit_file", "write_file"})
        errors = sum(1 for event in self.events if event.is_error)
        incomplete = any(event.status == "incomplete" for event in self.events)
        duration_ms = max((event.duration_ms or 0 for event in self.events), default=0)

        if duration_ms:
            parts = [f"Worked for {_format_duration(duration_ms)}"]
        else:
            parts = [f"Worked for {steps} step{'s' if steps != 1 else ''}"]
        parts.append(f"{len(tool_events)} tool{'s' if len(tool_events) != 1 else ''} used")
        if edited_files:
            parts.append(f"{edited_files} file{'s' if edited_files != 1 else ''} edited")
        if errors:
            parts.append(f"{errors} error{'s' if errors != 1 else ''}")
        if incomplete:
            parts.append("incomplete")
        usage = _summarize_usage_events(self.events)
        if usage["requests"]:
            token_text = (
                f"{_format_token_count(usage['total_tokens'])} tokens "
                f"({_format_token_count(usage['input_tokens'])} in / "
                f"{_format_token_count(usage['output_tokens'])} out)"
            )
            if usage["estimated_requests"]:
                token_text += " estimated"
            parts.append(token_text)
        return " | ".join(parts)


class ConsoleRenderer:
    def __init__(self, stream: TextIO | None = None, verbose: bool = False, max_output_chars: int = 240):
        self.stream = stream
        self.verbose = verbose
        self.max_output_chars = max_output_chars

    def render(self, event: AgentEvent) -> None:
        line = self.format(event)
        if not line:
            return
        print(line, file=self.stream)

    def format(self, event: AgentEvent) -> str:
        if event.type == "thinking":
            if self.verbose and event.step and event.total:
                return f"Thinking... ({event.step}/{event.total})"
            return "Thinking..."

        if event.type == "compact":
            return event.message

        if event.type == "usage":
            return _format_usage_event(event.usage)

        if event.type == "plan":
            path = f"\nPath: {event.output}" if event.output else ""
            return f"Plan ({event.status or 'pending'}): {event.summary or 'Proposed plan'}{path}\n{event.message}"

        if event.type == "tool_start":
            suffix = f" {self._format_input(event.input)}" if event.input else ""
            return f"-> {event.name}{suffix}"

        if event.type == "tool_result":
            if self.verbose and event.output:
                return f"OK {event.name}: {self._compact(event.output)}"
            return f"OK {event.name} completed"

        if event.type == "tool_error":
            return f"Error {event.name}: {event.message}"

        if event.type in {"final", "stopped"}:
            return event.message

        return ""

    def _format_input(self, tool_input: dict[str, object]) -> str:
        try:
            rendered = json.dumps(tool_input, ensure_ascii=False)
        except TypeError:
            rendered = str(tool_input)
        return self._compact(rendered)

    def _compact(self, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= self.max_output_chars:
            return normalized
        return f"{normalized[: self.max_output_chars]}..."


class TuiRenderer:
    def __init__(
        self,
        banner: str,
        verbose: bool = False,
        max_output_chars: int = 240,
        stream: TextIO | None = None,
        console: Console | None = None,
    ):
        self.banner = banner
        self.verbose = verbose
        self.max_output_chars = max_output_chars
        self.stream = stream
        self.console = console or Console(
            file=stream or sys.stdout,
            force_terminal=stream is None,
            color_system="auto" if stream is None else None,
            soft_wrap=True,
            width=100,
        )
        self.turns: list[TurnViewState] = []
        self.system_messages: list[str] = []
        self.running = False
        self.transcript_lines: list[str] = []

    def print_banner(self) -> None:
        self._record("banner", self.banner)
        self._print(Panel(Text(self.banner), title="uedev", border_style="cyan", box=box.ASCII, expand=False))

    def print_user(self, message: str) -> None:
        self._record("user", message)
        self._print(_block("user", Text(message), style="bold cyan"))

    def print_user_prompt(self, message: str) -> None:
        self._record("user", message)
        rendered = Text("> ", style="bold cyan")
        rendered.append(message)
        self._print(rendered)

    def print_system(self, message: str) -> None:
        self.add_system_message(message)
        self._record("system", message)
        self._print(_block("system", Text(message), style="bold blue"))

    def print_approval(self, command: str, reason: str) -> None:
        body = Text()
        body.append("Reason: ", style="bold")
        body.append(reason)
        body.append("\nCommand: ", style="bold")
        body.append(command)
        self._record("approval", f"Reason: {reason}\nCommand: {command}")
        self._print(_block("approval required", body, style="bold yellow"))

    def start_turn(self, turn_id: str, user_message: str, *, echo_user: bool = False) -> None:
        self.turns.append(TurnViewState(turn_id=turn_id, user_message=user_message))
        if echo_user:
            self.print_user_prompt(user_message)
        else:
            self._record("user", user_message)
        self.running = True

    def add_system_message(self, message: str) -> None:
        self.system_messages.append(message)

    def render(self, event: AgentEvent) -> None:
        turn = self._find_turn(event.turn_id)
        if turn is None:
            turn = TurnViewState(turn_id=event.turn_id or f"turn-{len(self.turns) + 1}", user_message="")
            self.turns.append(turn)

        if event.type == "thinking":
            if self.verbose or not any(existing.type == "thinking" for existing in turn.events):
                self._record("thinking", self._thinking_text(event))
                self._print(Text(self._thinking_text(event), style="dim"))
        elif event.type == "budget":
            if self.verbose:
                self._record("budget", event.message)
                self._print(Text(event.message, style="dim"))
        elif event.type == "usage":
            if self.verbose:
                rendered = _format_usage_event(event.usage)
                self._record("usage", rendered)
                self._print(Text(rendered, style="dim"))
        elif event.type == "compact":
            self._record("compact", event.message)
            self._print(_block("system", Text(event.message), style="bold blue"))
        elif event.type == "plan":
            self._record("plan", self._plan_record_text(event))
            self._print(self._plan_block(event))
        elif event.type == "tool_start":
            self._record("tool_start", self._tool_start_text(event))
            self._print(self._tool_block(event.name, self._tool_start_text(event), "yellow"))
        elif event.type == "tool_result":
            self._record("tool_result", self._tool_result_text(event))
            self._print(self._tool_block(event.name, self._tool_result_text(event), "green"))
        elif event.type == "tool_error":
            self._record("tool_error", self._tool_error_text(event))
            self._print(self._tool_block(event.name, self._tool_error_text(event), "red"))
        elif event.type in {"final", "stopped"}:
            plan_already_rendered = any(existing.type == "plan" for existing in turn.events)
            turn.add_event(event)
            self.running = False
            summary = turn.summary()
            self._record("summary", summary)
            self._print(Rule(summary, style="dim"))
            if event.type == "final" and plan_already_rendered and is_proposed_plan(event.message):
                return
            record_type = "assistant" if event.type == "final" else event.status or "stopped"
            self._record(record_type, event.message)
            self._print(
                self._assistant_block(
                    event.message,
                    is_error=event.is_error,
                    is_incomplete=event.status == "incomplete",
                )
            )
            return

        turn.add_event(event)

    def clear(self) -> None:
        self.turns.clear()
        self.system_messages.clear()
        self.transcript_lines.clear()
        self.running = False

    def render_history(self, messages: list[ChatMessage], source: str) -> None:
        self.clear()
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
                    self._record("compact", content)
                    self._print(_block("system", Text(content), style="bold blue"))
                    continue
                self.print_user_prompt(message.content)
                continue
            if message.role == "assistant":
                if message.content.strip():
                    if is_proposed_plan(message.content):
                        event = AgentEvent(
                            type="plan",
                            message=extract_proposed_plan_content(message.content),
                            status="pending",
                            summary="Proposed plan",
                        )
                        self._record("plan", self._plan_record_text(event))
                        self._print(self._plan_block(event))
                    else:
                        self._record("assistant", message.content)
                        self._print(self._assistant_block(message.content))
                elif message.tool_calls:
                    names = ", ".join(tool_call.name for tool_call in message.tool_calls)
                    rendered = f"Tool calls: {names}"
                    self._record("assistant", rendered)
                    self._print(_block("assistant", Text(rendered), style="bold green"))
                continue
            if message.role == "tool":
                rendered = message.content.strip()
                name = message.name or "tool"
                self._record("tool_result", rendered)
                self._print(self._tool_block(name, rendered, "green"))

    def render_display_history(self, records: list[dict[str, object]], source: str) -> None:
        self.clear()
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

    def status_text(self) -> str:
        return "uedev chat | running" if self.running else "uedev chat | ready"

    def render_text(self) -> str:
        return "\n\n".join(self.transcript_lines)

    def _find_turn(self, turn_id: str) -> TurnViewState | None:
        if not self.turns:
            return None
        if not turn_id:
            return self.turns[-1]
        for turn in reversed(self.turns):
            if turn.turn_id == turn_id:
                return turn
        return None

    def _thinking_text(self, event: AgentEvent) -> str:
        return "Thinking..."

    def _tool_start_text(self, event: AgentEvent) -> str:
        if self.verbose and event.input:
            return f"Running {event.name}\n{self._format_input(event.input)}"
        return f"Running {event.name}"

    def _tool_result_text(self, event: AgentEvent) -> str:
        output = self._format_output(event.output, verbose=self.verbose)
        if output:
            return f"OK {event.name}\n{output}"
        return f"OK {event.name} completed"

    def _tool_error_text(self, event: AgentEvent) -> str:
        detail = self._format_output(event.message or event.output, verbose=True)
        if detail:
            return f"Failed {event.name}\n{detail}"
        return f"Failed {event.name}"

    def _tool_block(self, name: str, body: str, style: str) -> RenderableType:
        renderable: RenderableType
        if self.verbose and "\n" in body:
            renderable = Syntax(body, "text", word_wrap=True)
        else:
            renderable = Text(body)
        return Panel(renderable, title=f"tool: {name}", border_style=style, box=box.ASCII, expand=False)

    def _plan_block(self, event: AgentEvent) -> RenderableType:
        title = event.summary or "Proposed plan"
        header = Text()
        header.append(f"{title}\n", style="bold")
        header.append(f"status: {event.status or 'pending'}")
        if event.output:
            header.append(f"\npath: {event.output}")
        try:
            body: RenderableType = Markdown(event.message)
        except Exception:
            body = Text(event.message)
        return Panel(Group(header, body), title="plan", border_style="magenta", box=box.ASCII, expand=False)

    def _plan_record_text(self, event: AgentEvent) -> str:
        lines = [
            f"title: {event.summary or 'Proposed plan'}",
            f"status: {event.status or 'pending'}",
        ]
        if event.output:
            lines.append(f"path: {event.output}")
        if event.message:
            lines.extend(["", event.message])
        return "\n".join(lines)

    def _assistant_block(self, message: str, *, is_error: bool = False, is_incomplete: bool = False) -> RenderableType:
        if is_error:
            return _block("stopped", Text(message), style="bold red")
        if is_incomplete:
            return _block("incomplete", Text(message), style="bold yellow")
        try:
            renderable: RenderableType = Markdown(message)
        except Exception:
            renderable = Text(message)
        return _block("assistant", renderable, style="bold green")

    def _format_input(self, tool_input: dict[str, object]) -> str:
        try:
            rendered = json.dumps(tool_input, ensure_ascii=False, indent=2)
        except TypeError:
            rendered = str(tool_input)
        return _compact_text(rendered, self.max_output_chars)

    def _format_output(self, value: str, *, verbose: bool) -> str:
        if verbose:
            return value.rstrip()
        return _compact_lines(value, self.max_output_chars)

    def _print(self, renderable: RenderableType) -> None:
        self.console.print(renderable)

    def _record(self, label: str, text: str) -> None:
        if not text:
            return
        self.transcript_lines.append(f"{label}:\n{text}")


def _format_duration(duration_ms: int) -> str:
    seconds = max(1, round(duration_ms / 1000))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining = divmod(seconds, 60)
    if remaining:
        return f"{minutes}m {remaining}s"
    return f"{minutes}m"


def _format_token_count(value: int) -> str:
    return f"{max(0, value):,}"


def _summarize_usage_events(events: list[AgentEvent]) -> dict[str, int]:
    summary = {
        "requests": 0,
        "estimated_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    for event in events:
        if not event.usage:
            continue
        summary["requests"] += 1
        if event.usage.get("source") == "estimated":
            summary["estimated_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
            summary[key] += _int_value(event.usage.get(key))
    return summary


def _format_usage_event(usage: dict[str, object]) -> str:
    purpose = str(usage.get("purpose") or "model")
    source = str(usage.get("source") or "provider")
    return (
        f"Tokens [{purpose}]: {_format_token_count(_int_value(usage.get('total_tokens')))} total · "
        f"{_format_token_count(_int_value(usage.get('input_tokens')))} in · "
        f"{_format_token_count(_int_value(usage.get('output_tokens')))} out · {source}"
    )


def _compact_text(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars]}..."


def _compact_lines(value: str, max_chars: int) -> str:
    if not value:
        return ""
    lines = value.rstrip().splitlines()
    if len(lines) <= 3 and len(value) <= max_chars:
        return "\n".join(lines)
    visible = lines[:3]
    hidden_count = max(0, len(lines) - len(visible))
    rendered = "\n".join(_compact_text(line, max_chars) for line in visible)
    if hidden_count:
        rendered += f"\n... +{hidden_count} lines (use --verbose to show full output)"
    elif len(value) > max_chars:
        rendered = _compact_text(value, max_chars)
    return rendered


def _block(label: str, renderable: RenderableType, *, style: str) -> RenderableType:
    return Group(Text(label, style=style), renderable)


def _event_from_dict(raw: dict[str, object]) -> AgentEvent:
    event_type = str(raw.get("type") or "stopped")
    status = str(raw.get("status") or "")
    raw_is_error = raw.get("is_error")
    is_error = bool(raw_is_error) if raw_is_error is not None else event_type == "stopped" and status != "incomplete"
    return AgentEvent(
        type=event_type,  # type: ignore[arg-type]
        message=str(raw.get("message") or ""),
        name=str(raw.get("name") or ""),
        input=raw.get("input") if isinstance(raw.get("input"), dict) else {},
        output=str(raw.get("output") or ""),
        step=_int_value(raw.get("step")),
        total=_int_value(raw.get("total")),
        turn_id=str(raw.get("turn_id") or ""),
        status=status,
        summary=str(raw.get("summary") or ""),
        duration_ms=_int_value(raw.get("duration_ms")),
        is_error=is_error,
        usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
    )


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
