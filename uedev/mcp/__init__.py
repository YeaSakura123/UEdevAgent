from __future__ import annotations

from .client import McpClient
from .transport import McpStdioTransport
from .types import McpError, McpServerConfig, McpServerStatus, McpTool, McpToolCallResult

__all__ = [
    "McpClient",
    "McpError",
    "McpServerConfig",
    "McpServerStatus",
    "McpStdioTransport",
    "McpTool",
    "McpToolCallResult",
    "McpToolRegistry",
]


def __getattr__(name: str):
    if name == "McpToolRegistry":
        from .registry import McpToolRegistry

        return McpToolRegistry
    raise AttributeError(name)
