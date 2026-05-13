from __future__ import annotations

from typing import Callable

ToolHandler = Callable[[dict[str, object]], str]

__all__ = ["ToolHandler"]
