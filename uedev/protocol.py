from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Union


class ActionParseError(ValueError):
    # 内部函数：初始化当前类实例，准备 模型 JSON action 协议解析 所需状态。
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class ShellAction:
    type: Literal["shell"]
    command: str
    reason: str


@dataclass(frozen=True)
class ToolAction:
    type: Literal["tool"]
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class FinalAction:
    type: Literal["final"]
    answer: str


AgentAction = Union[ShellAction, ToolAction, FinalAction]


# 外部函数：解析模型返回的 JSON action，负责 模型 JSON action 协议解析。
def parse_agent_action(raw: str) -> AgentAction:
    try:
        json_text = extract_json(raw)
        data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"Model did not return valid JSON: {error}", raw) from error

    if not isinstance(data, dict):
        raise ActionParseError("Model JSON response must be an object.", raw)

    action_type = data.get("type")
    if action_type is None and "answer" in data:
        action_type = "final"
    if action_type == "tool":
        name = data.get("name")
        tool_input = data.get("input", {})
        if not isinstance(name, str) or not name.strip():
            raise ActionParseError("tool action requires a non-empty name", raw)
        if not isinstance(tool_input, dict):
            raise ActionParseError("tool action input must be an object", raw)
        return ToolAction(type="tool", name=name, input=tool_input)

    if action_type == "shell":
        command = data.get("command")
        reason = data.get("reason")
        if not isinstance(command, str) or not command.strip():
            raise ActionParseError("shell action requires a non-empty command", raw)
        if not isinstance(reason, str) or not reason.strip():
            raise ActionParseError("shell action requires a non-empty reason", raw)
        return ShellAction(type="shell", command=command, reason=reason)

    if action_type == "final":
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ActionParseError("final action requires a non-empty answer", raw)
        return FinalAction(type="final", answer=answer)

    raise ActionParseError(f"Unknown action type: {action_type}", raw)


# 内部函数：从模型回复中提取 JSON 对象，兼容代码块包裹。
def extract_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first = stripped.find("{")
    if first >= 0:
        candidate = stripped[first:]
        try:
            _, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            last = stripped.rfind("}")
            if last > first:
                return stripped[first : last + 1]
        else:
            return candidate[:end].strip()

    return stripped
