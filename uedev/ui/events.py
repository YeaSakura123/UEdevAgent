from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AgentEventType = Literal[
    "thinking",
    "assistant_delta",
    "budget",
    "compact",
    "plan",
    "tool_start",
    "tool_result",
    "tool_error",
    "final",
    "stopped",
]


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    message: str = ""
    name: str = ""
    input: dict[str, object] = field(default_factory=dict)
    output: str = ""
    step: int = 0
    total: int = 0
    turn_id: str = ""
    status: str = ""
    summary: str = ""
    duration_ms: int = 0
    is_error: bool = False


def thinking_event(step: int, total: int, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="thinking", step=step, total=total, turn_id=turn_id, status="thinking")


def assistant_delta_event(delta: str, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="assistant_delta", message=delta, turn_id=turn_id, status="streaming")


def budget_event(message: str, turn_id: str = "", duration_ms: int = 0, summary: str = "") -> AgentEvent:
    return AgentEvent(
        type="budget",
        message=message,
        turn_id=turn_id,
        status="running",
        duration_ms=duration_ms,
        summary=summary,
    )


def compact_event(message: str, turn_id: str = "", output: str = "") -> AgentEvent:
    return AgentEvent(type="compact", message=message, output=output, turn_id=turn_id, status="compacted")


def plan_event(message: str, path: str, title: str, status: str = "pending", turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="plan", message=message, output=path, summary=title, status=status, turn_id=turn_id)


def tool_start_event(name: str, tool_input: dict[str, object], turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_start", name=name, input=tool_input, turn_id=turn_id, status="running")


def tool_result_event(name: str, output: str, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_result", name=name, output=output, turn_id=turn_id, status="completed")


def tool_error_event(name: str, message: str, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_error", name=name, message=message, turn_id=turn_id, status="failed", is_error=True)


def final_event(answer: str, turn_id: str = "", duration_ms: int = 0, summary: str = "") -> AgentEvent:
    return AgentEvent(
        type="final",
        message=answer,
        turn_id=turn_id,
        status="final",
        duration_ms=duration_ms,
        summary=summary,
    )


def stopped_event(message: str, turn_id: str = "", duration_ms: int = 0) -> AgentEvent:
    return AgentEvent(
        type="stopped",
        message=message,
        turn_id=turn_id,
        status="stopped",
        duration_ms=duration_ms,
        is_error=True,
    )


def incomplete_event(message: str, turn_id: str = "", duration_ms: int = 0) -> AgentEvent:
    return AgentEvent(
        type="stopped",
        message=message,
        turn_id=turn_id,
        status="incomplete",
        duration_ms=duration_ms,
        is_error=False,
    )
