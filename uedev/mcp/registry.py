from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..config import ConfigError, load_system_config
from ..tools.specs import ToolSpec
from .client import McpClient
from .types import McpError, McpServerConfig, McpServerStatus, McpTool

MCP_TOOL_PREFIX = "mcp__"


class McpToolRegistry:
    def __init__(self, configs: dict[str, McpServerConfig] | None = None):
        self.configs = configs or {}
        self.clients: dict[str, McpClient] = {}
        self.statuses: dict[str, McpServerStatus] = {}
        self.tools: dict[str, McpTool] = {}

    @classmethod
    def from_system_config(cls) -> "McpToolRegistry":
        try:
            config = load_system_config()
        except ConfigError as error:
            registry = cls()
            if "Config file not found" not in str(error):
                registry.statuses["config"] = McpServerStatus("config", "failed", str(error))
            return registry
        registry = cls(config.mcp_servers)
        registry.startup_check()
        return registry

    def startup_check(self) -> None:
        if not self.configs:
            return
        for name, config in self.configs.items():
            if not config.enabled:
                self.statuses[name] = McpServerStatus(name, "disabled")
                continue
            try:
                client = McpClient(config)
                raw_tools = client.list_tools()
                tools = self._assign_agent_names(raw_tools)
            except Exception as error:
                self.statuses[name] = McpServerStatus(name, "failed", _error_text(error))
                continue
            self.clients[name] = client
            self.statuses[name] = McpServerStatus(name, "connected", tools=tools)
            for tool in tools:
                self.tools[tool.agent_name] = tool

    def tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for tool in sorted(self.tools.values(), key=lambda item: item.agent_name):
            schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
            if schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.agent_name,
                        "description": f"[MCP:{tool.server_name}] {tool.description or tool.name}",
                        "parameters": schema,
                    },
                }
            )
        return specs

    def handlers(self) -> dict[str, Callable[[dict[str, object]], str]]:
        return {agent_name: self._make_handler(agent_name) for agent_name in self.tools}

    def call_tool(self, agent_name: str, arguments: dict[str, object]) -> str:
        tool = self.tools.get(agent_name)
        if tool is None:
            raise McpError(f"Unknown MCP tool: {agent_name}")
        client = self.clients.get(tool.server_name)
        if client is None:
            raise McpError(f"MCP server is not connected: {tool.server_name}")
        return client.call_tool(tool.name, dict(arguments)).render()

    def render_status(self) -> str:
        if not self.statuses:
            return "MCP: no configured servers."
        connected = sum(1 for status in self.statuses.values() if status.state == "connected")
        failed = sum(1 for status in self.statuses.values() if status.state == "failed")
        disabled = sum(1 for status in self.statuses.values() if status.state == "disabled")
        lines = [f"MCP: {connected} connected, {failed} failed, {disabled} disabled"]
        for name, status in sorted(self.statuses.items()):
            if status.state == "connected":
                lines.append(f"- {name}: connected, tools={status.tool_count}")
                for tool in status.tools:
                    lines.append(f"  - {tool.agent_name}")
            elif status.error:
                lines.append(f"- {name}: {status.state}, error={status.error}")
            else:
                lines.append(f"- {name}: {status.state}")
        return "\n".join(lines)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()

    def _make_handler(self, agent_name: str) -> Callable[[dict[str, object]], str]:
        def handler(tool_input: dict[str, object]) -> str:
            return self.call_tool(agent_name, tool_input)

        return handler

    def _assign_agent_names(self, tools: list[McpTool]) -> list[McpTool]:
        assigned: list[McpTool] = []
        seen: set[str] = set(self.tools)
        for tool in tools:
            base = f"{MCP_TOOL_PREFIX}{_safe_part(tool.server_name)}__{_safe_part(tool.name)}"
            agent_name = base
            suffix = 2
            while agent_name in seen:
                agent_name = f"{base}_{suffix}"
                suffix += 1
            seen.add(agent_name)
            assigned.append(
                McpTool(
                    server_name=tool.server_name,
                    name=tool.name,
                    agent_name=agent_name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                )
            )
        return assigned


def is_mcp_tool_name(name: str) -> bool:
    return name.startswith(MCP_TOOL_PREFIX)


def _safe_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    safe = safe.strip("_")
    return safe or "tool"


def _error_text(error: Exception) -> str:
    text = str(error).strip()
    return text or error.__class__.__name__
