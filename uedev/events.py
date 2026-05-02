from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AgentEventType = Literal[
    "thinking",
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


# 外部函数：创建模型思考进度事件，负责 agent 运行过程的状态展示。
def thinking_event(step: int, total: int, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="thinking", step=step, total=total, turn_id=turn_id, status="thinking")


# 外部函数：创建工具开始事件，负责展示即将调用的工具和参数。
def tool_start_event(name: str, tool_input: dict[str, object], turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_start", name=name, input=tool_input, turn_id=turn_id, status="running")


# 外部函数：创建工具完成事件，负责展示工具执行后的摘要结果。
def tool_result_event(name: str, output: str, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_result", name=name, output=output, turn_id=turn_id, status="completed")


# 外部函数：创建工具失败事件，负责展示工具执行错误。
def tool_error_event(name: str, message: str, turn_id: str = "") -> AgentEvent:
    return AgentEvent(type="tool_error", name=name, message=message, turn_id=turn_id, status="failed", is_error=True)


# 外部函数：创建最终回答事件，负责展示 agent 完成后的用户可见回答。
def final_event(answer: str, turn_id: str = "", duration_ms: int = 0, summary: str = "") -> AgentEvent:
    return AgentEvent(
        type="final",
        message=answer,
        turn_id=turn_id,
        status="final",
        duration_ms=duration_ms,
        summary=summary,
    )


# 外部函数：创建停止事件，负责展示达到最大轮次等未完成状态。
def stopped_event(message: str, turn_id: str = "", duration_ms: int = 0) -> AgentEvent:
    return AgentEvent(
        type="stopped",
        message=message,
        turn_id=turn_id,
        status="stopped",
        duration_ms=duration_ms,
        is_error=True,
    )
