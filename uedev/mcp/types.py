from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


McpServerState = Literal["disabled", "unconfigured", "connected", "failed"]


class McpError(RuntimeError):
    pass


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    transport: str
    command: str
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    enabled: bool = True
    timeout_seconds: int = 10


@dataclass(frozen=True)
class McpTool:
    server_name: str
    name: str
    agent_name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolCallResult:
    server_name: str
    tool_name: str
    result: dict[str, Any]

    def render(self) -> str:
        content = self.result.get("content")
        if isinstance(content, list):
            lines: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    lines.append(str(item.get("text", "")))
                else:
                    lines.append(json.dumps(item, ensure_ascii=False, indent=2))
            if lines:
                return "\n".join(lines)
        return json.dumps(self.result, ensure_ascii=False, indent=2)


@dataclass
class McpServerStatus:
    name: str
    state: McpServerState
    error: str = ""
    tools: list[McpTool] = field(default_factory=list)

    @property
    def tool_count(self) -> int:
        return len(self.tools)
