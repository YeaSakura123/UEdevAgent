from __future__ import annotations

from typing import Any

from .transport import McpStdioTransport
from .types import McpServerConfig, McpTool, McpToolCallResult


class McpClient:
    def __init__(self, config: McpServerConfig, transport: McpStdioTransport | None = None):
        self.config = config
        self.transport = transport or McpStdioTransport(config)
        self.initialized = False

    def initialize(self) -> None:
        self.transport.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "uedev-cli", "version": "0.1.0"},
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        self.transport.notify("notifications/initialized")
        self.initialized = True

    def list_tools(self) -> list[McpTool]:
        if not self.initialized:
            self.initialize()
        result = self.transport.request("tools/list", timeout_seconds=self.config.timeout_seconds)
        raw_tools = result.get("tools", [])
        tools: list[McpTool] = []
        if not isinstance(raw_tools, list):
            return tools
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            input_schema = raw.get("inputSchema")
            tools.append(
                McpTool(
                    server_name=self.config.name,
                    name=name,
                    agent_name="",
                    description=str(raw.get("description") or ""),
                    input_schema=input_schema if isinstance(input_schema, dict) else {"type": "object", "properties": {}},
                )
            )
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        if not self.initialized:
            self.initialize()
        result = self.transport.request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_seconds=self.config.timeout_seconds,
        )
        return McpToolCallResult(self.config.name, tool_name, result)

    def close(self) -> None:
        self.transport.close()
