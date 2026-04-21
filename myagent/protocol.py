from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Union


class ActionParseError(ValueError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class ShellAction:
    type: Literal["shell"]
    command: str
    reason: str


@dataclass(frozen=True)
class FinalAction:
    type: Literal["final"]
    answer: str


AgentAction = Union[ShellAction, FinalAction]


def parse_agent_action(raw: str) -> AgentAction:
    try:
        json_text = extract_json(raw)
        data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"Model did not return valid JSON: {error}", raw) from error

    if not isinstance(data, dict):
        raise ActionParseError("Model JSON response must be an object.", raw)

    action_type = data.get("type")
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


def extract_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        return stripped[first : last + 1]

    return stripped
