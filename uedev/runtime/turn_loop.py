from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..state.config import RuntimeBudgetConfig


@dataclass(frozen=True)
class ToolAction:
    name: str
    input: dict[str, Any]


@dataclass
class TurnBudgetState:
    config: RuntimeBudgetConfig
    started_at: float
    model_requests: int = 0
    tool_calls: int = 0
    total_output_tokens: int = 0
    consecutive_tool_failures: int = 0
    permission_denials: int = 0
    no_progress_rounds: int = 0
    tool_soft_limit_reminded: bool = False
    output_soft_limit_reminded: bool = False
    no_progress_reminded: bool = False

    def __post_init__(self) -> None:
        self.per_tool_calls: dict[str, int] = {}

    def elapsed_ms(self) -> int:
        return _duration_ms(self.started_at)

    def elapsed_seconds(self) -> int:
        return max(0, int(time.perf_counter() - self.started_at))

    def can_request_model(self) -> bool:
        return self.model_requests < self.config.model_request_hard_limit

    def next_model_request(self) -> int:
        self.model_requests += 1
        return self.model_requests

    def record_tool_call(self, name: str) -> int:
        self.tool_calls += 1
        self.per_tool_calls[name] = self.per_tool_calls.get(name, 0) + 1
        return self.per_tool_calls[name]

    def tool_limit_for(self, name: str) -> int | None:
        return self.config.tool_call_limits.get(name)

    def wall_clock_exceeded(self) -> bool:
        return self.elapsed_seconds() >= self.config.wall_clock_seconds

    def record_tool_result(self, *, is_error: bool, permission_denied: bool, progress: bool) -> None:
        if permission_denied:
            self.permission_denials += 1
        if is_error:
            self.consecutive_tool_failures += 1
        else:
            self.consecutive_tool_failures = 0
        if progress:
            self.no_progress_rounds = 0
        else:
            self.no_progress_rounds += 1

    def status(self, phase: str = "") -> str:
        parts = [
            f"model {self.model_requests}/{self.config.model_request_hard_limit}",
            f"tools {self.tool_calls}/{self.config.tool_call_soft_limit}",
        ]
        if phase:
            parts.append(phase)
        return " 路 ".join(parts)


def truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


def _duration_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def defers_tool_confirmation(goal: str, answer: str) -> bool:
    goal_text = goal.lower()
    answer_text = answer.lower()
    if not any(
        token in goal_text
        for token in [
            "ue",
            "unreal",
            "editor",
            "script",
            "execute",
            "launch",
            "run",
            "build",
            "compile",
            "uht",
            "\u811a\u672c",
            "\u6267\u884c",
            "\u542f\u52a8",
            "\u8fd0\u884c",
            "\u7f16\u8bd1",
            ".py",
        ]
    ):
        return False
    confirmation_tokens = ["confirm", "confirmation", "\u786e\u8ba4", "\u662f\u5426", "y/n", "[y/n]"]
    action_tokens = [
        "run",
        "execute",
        "launch",
        "start",
        "continue",
        "build",
        "compile",
        "\u6267\u884c",
        "\u542f\u52a8",
        "\u8fd0\u884c",
        "\u7ee7\u7eed",
        "\u7f16\u8bd1",
    ]
    return any(token in answer_text for token in confirmation_tokens) and any(
        token in answer_text for token in action_tokens
    )


def is_acknowledgement_answer(answer: str) -> bool:
    normalized = " ".join(answer.strip().lower().split())
    if not normalized:
        return False
    result_tokens = [
        "project=",
        "ue doctor",
        "engine",
        "perforce",
        "\u9879\u76ee",
        "\u5f15\u64ce",
        "\u7248\u672c",
        "\u5b58\u5728",
        "\u4e0d\u5b58\u5728",
        "\u7ed3\u679c",
        "error",
        "failed",
    ]
    if any(token in normalized for token in result_tokens):
        return False
    acknowledgement_tokens = [
        "understood",
        "got it",
        "i'll",
        "i will",
        "i\u2019ll",
        "will follow",
        "will directly invoke",
        "future",
        "harness",
        "\u6536\u5230",
        "\u660e\u767d",
        "\u4e86\u89e3",
        "\u5df2\u6309\u4f60\u7684\u8981\u6c42",
        "\u4f1a\u9075\u5faa",
        "\u9075\u5faa\u8be5\u884c\u4e3a",
        "\u4ee5\u540e\u4f1a",
        "\u4e0b\u6b21\u4f1a",
    ]
    return any(token in normalized for token in acknowledgement_tokens)


__all__ = [
    "ToolAction",
    "TurnBudgetState",
    "_duration_ms",
    "defers_tool_confirmation",
    "is_acknowledgement_answer",
    "truncate",
]
