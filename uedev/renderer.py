from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TextIO

from .events import AgentEvent


@dataclass
class TurnViewState:
    turn_id: str
    user_message: str
    events: list[AgentEvent] = field(default_factory=list)
    collapsed: bool = False
    final_answer: str = ""

    # 外部函数：记录一条事件，并在 final/stopped 后折叠本轮过程。
    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)
        if event.type in {"final", "stopped"}:
            self.final_answer = event.message
            self.collapsed = True

    # 外部函数：生成本轮摘要，负责最终回答前的过程收束展示。
    def summary(self) -> str:
        steps = max((event.step for event in self.events), default=0)
        tool_events = [event for event in self.events if event.type in {"tool_result", "tool_error"}]
        edited_files = sum(1 for event in tool_events if event.name in {"edit_file", "write_file"})
        errors = sum(1 for event in self.events if event.is_error)
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
        return " · ".join(parts)


class ConsoleRenderer:
    def __init__(self, stream: TextIO | None = None, verbose: bool = False, max_output_chars: int = 240):
        self.stream = stream
        self.verbose = verbose
        self.max_output_chars = max_output_chars

    # 外部函数：渲染一条 agent 事件，负责普通控制台过程输出。
    def render(self, event: AgentEvent) -> None:
        line = self.format(event)
        if not line:
            return
        print(line, file=self.stream)

    # 外部函数：把 agent 事件格式化为一行文本，负责控制台输出内容组织。
    def format(self, event: AgentEvent) -> str:
        if event.type == "thinking":
            if self.verbose and event.step and event.total:
                return f"Thinking... ({event.step}/{event.total})"
            return "Thinking..."

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

    # 内部函数：格式化工具入参，只展示短 JSON，避免过程输出刷屏。
    def _format_input(self, tool_input: dict[str, object]) -> str:
        try:
            rendered = json.dumps(tool_input, ensure_ascii=False)
        except TypeError:
            rendered = str(tool_input)
        return self._compact(rendered)

    # 内部函数：压缩长文本，负责保持事件输出简洁。
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
    ):
        self.banner = banner
        self.verbose = verbose
        self.max_output_chars = max_output_chars
        self.stream = stream
        self.turns: list[TurnViewState] = []
        self.system_messages: list[str] = []
        self.running = False
        self.transcript_lines: list[str] = []

    # 外部函数：打印启动信息卡片，负责 chat 开始时的版本、模型和目录展示。
    def print_banner(self) -> None:
        self._write(self.banner)

    # 外部函数：追加用户输入行，主要供测试或非 PromptSession 驱动场景复用。
    def print_user(self, message: str) -> None:
        self._write(f"› {message}")

    # 外部函数：打印本地系统消息，负责 slash command、/clear 等本地命令输出。
    def print_system(self, message: str) -> None:
        self.add_system_message(message)
        self._write(_prefix_block(message, "• "))

    # 外部函数：开始一轮用户请求，负责创建 transcript 状态并标记运行中。
    def start_turn(self, turn_id: str, user_message: str) -> None:
        self.turns.append(TurnViewState(turn_id=turn_id, user_message=user_message))
        self.running = True

    # 外部函数：记录本地系统消息，负责 slash command 和状态提示的历史保存。
    def add_system_message(self, message: str) -> None:
        self.system_messages.append(message)

    # 外部函数：渲染一条 agent 事件，负责按时间顺序追加阶段动作、工具信息和最终回答。
    def render(self, event: AgentEvent) -> None:
        turn = self._find_turn(event.turn_id)
        if turn is None:
            turn = TurnViewState(turn_id=event.turn_id or f"turn-{len(self.turns) + 1}", user_message="")
            self.turns.append(turn)

        if event.type == "thinking":
            if not any(existing.type == "thinking" for existing in turn.events):
                self._write(self._format_thinking(event))
        elif event.type == "tool_start":
            self._write(self._format_tool_start(event))
        elif event.type == "tool_result":
            self._write(self._format_tool_result(event))
        elif event.type == "tool_error":
            self._write(self._format_tool_error(event))
        elif event.type in {"final", "stopped"}:
            turn.add_event(event)
            self.running = False
            self._write(self._format_summary(turn))
            if event.message:
                self._write(_prefix_block(event.message, "• "))
            return

        turn.add_event(event)

    # 外部函数：清空 renderer 状态，负责 /clear 后重置 transcript 历史。
    def clear(self) -> None:
        self.turns.clear()
        self.system_messages.clear()
        self.transcript_lines.clear()
        self.running = False

    # 外部函数：生成简短状态文本，兼容旧测试和未来状态栏调用。
    def status_text(self) -> str:
        return "uedev chat · running" if self.running else "uedev chat · ready"

    # 外部函数：返回顺序 transcript 文本，负责测试和日志场景的完整内容查看。
    def render_text(self) -> str:
        return "\n\n".join(self.transcript_lines)

    # 内部函数：查找轮次状态，支持事件按 turn_id 回填。
    def _find_turn(self, turn_id: str) -> TurnViewState | None:
        if not self.turns:
            return None
        if not turn_id:
            return self.turns[-1]
        for turn in reversed(self.turns):
            if turn.turn_id == turn_id:
                return turn
        return None

    # 内部函数：格式化 thinking 事件，负责显示当前轮正在工作。
    def _format_thinking(self, event: AgentEvent) -> str:
        if self.verbose and event.step and event.total:
            return f"• Working ({event.step}/{event.total})"
        return "• Working"

    # 内部函数：格式化工具开始事件，负责展示工具名和简短参数。
    def _format_tool_start(self, event: AgentEvent) -> str:
        suffix = f" {self._format_input(event.input)}" if event.input else ""
        return f"• Running {event.name}{suffix}"

    # 内部函数：格式化工具成功事件，负责展示工具完成状态和可选输出摘要。
    def _format_tool_result(self, event: AgentEvent) -> str:
        lines = [f"• Ran {event.name}"]
        output = self._format_output(event.output)
        if output and (self.verbose or "\n" in output):
            lines.append(f"  └ {output}")
        return "\n".join(lines)

    # 内部函数：格式化工具失败事件，负责展示失败状态和错误摘要。
    def _format_tool_error(self, event: AgentEvent) -> str:
        detail = self._format_output(event.message or event.output)
        return f"• Failed {event.name}\n  └ {detail}" if detail else f"• Failed {event.name}"

    # 内部函数：格式化最终折叠摘要，负责在 final 后收束过程信息。
    def _format_summary(self, turn: TurnViewState) -> str:
        summary = turn.summary()
        width = max(20, 100 - len(summary))
        return f"─ {summary} " + "─" * width

    # 内部函数：格式化工具入参，负责避免长 JSON 撑爆 transcript。
    def _format_input(self, tool_input: dict[str, object]) -> str:
        try:
            rendered = json.dumps(tool_input, ensure_ascii=False)
        except TypeError:
            rendered = str(tool_input)
        return _compact_text(rendered, self.max_output_chars)

    # 内部函数：格式化工具输出，负责长输出折叠为摘要提示。
    def _format_output(self, value: str) -> str:
        return _compact_lines(value, self.max_output_chars)

    # 内部函数：写入 transcript 和输出流，负责保持历史与终端显示一致。
    def _write(self, text: str) -> None:
        if not text:
            return
        self.transcript_lines.append(text)
        print(text, file=self.stream)


def _format_duration(duration_ms: int) -> str:
    seconds = max(1, round(duration_ms / 1000))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining = divmod(seconds, 60)
    if remaining:
        return f"{minutes}m {remaining}s"
    return f"{minutes}m"


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
        return "\n  ".join(lines)
    visible = lines[:3]
    hidden_count = max(0, len(lines) - len(visible))
    rendered = "\n  ".join(_compact_text(line, max_chars) for line in visible)
    if hidden_count:
        rendered += f"\n  … +{hidden_count} lines (ctrl+t to view transcript)"
    elif len(value) > max_chars:
        rendered = _compact_text(value, max_chars)
    return rendered


def _prefix_block(message: str, first_prefix: str) -> str:
    lines = message.splitlines() or [""]
    if len(lines) == 1:
        return first_prefix + lines[0]
    return "\n".join([first_prefix + lines[0], *[f"  {line}" for line in lines[1:]]])
