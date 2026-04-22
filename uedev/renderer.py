from __future__ import annotations

import json
from typing import TextIO

from .events import AgentEvent


class ConsoleRenderer:
    def __init__(self, stream: TextIO | None = None, verbose: bool = False, max_output_chars: int = 240):
        self.stream = stream
        self.verbose = verbose
        self.max_output_chars = max_output_chars

    # 外部函数：渲染一条 agent 事件，负责第一阶段的普通控制台过程输出。
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

        if event.type == "final":
            return event.message

        if event.type == "stopped":
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
