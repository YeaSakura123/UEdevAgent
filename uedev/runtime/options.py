from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..state.config import RuntimeBudgetConfig


@dataclass(frozen=True)
class AgentOptions:
    task: str
    max_steps: int
    auto_approve: bool
    cwd: Path
    timeout_seconds: int
    verbose: bool
    context_threshold: int | None = None
    plain: bool = False
    runtime_budget: RuntimeBudgetConfig | None = None

__all__ = ["AgentOptions"]
