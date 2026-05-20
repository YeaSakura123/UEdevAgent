from __future__ import annotations

import json
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from uedev.state.config import ConfigError, load_system_config
from uedev.runtime.agent import AgentOptions, AgentRuntime
from uedev.mcp.client import McpClient
from uedev.mcp.registry import McpToolRegistry
from uedev.mcp.transport import McpStdioTransport
from uedev.mcp.types import McpServerConfig, McpServerStatus, McpTool
from uedev.policy.permissions import classify_tool_permission


@contextmanager
def workspace_temp_dir():
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    yield str(path)


def write_system_config(path: Path, mcp: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "first": {
                        "model": "test-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    }
                },
                "ue": {"engines": {}},
                "mcp": mcp or {"servers": {}},
            }
        ),
        encoding="utf-8",
    )


class McpConfigTests(unittest.TestCase):
    def test_load_system_config_accepts_optional_mcp_servers(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(
                config_path,
                {
                    "servers": {
                        "unreal": {
                            "transport": "stdio",
                            "command": "python",
                            "args": ["-m", "mcp_unreal"],
                            "cwd": temp,
                            "enabled": True,
                        }
                    }
                },
            )

            config = load_system_config(config_path)

        server = config.mcp_servers["unreal"]
        self.assertEqual(server.command, "python")
        self.assertEqual(server.args, ("-m", "mcp_unreal"))
        self.assertEqual(server.cwd, Path(temp).resolve())

    def test_load_system_config_rejects_invalid_mcp_transport(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path, {"servers": {"bad": {"transport": "http", "command": "server"}}})

            with self.assertRaisesRegex(ConfigError, "unsupported transport"):
                load_system_config(config_path)


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        pass


class FakeProcess:
    def __init__(self, stdout_lines: list[str]) -> None:
        self.stdin = FakeStdin()
        self.stdout = iter(stdout_lines)
        self.stderr = iter([])
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None) -> None:
        self.returncode = 0


class McpTransportClientTests(unittest.TestCase):
    def test_stdio_transport_sends_json_rpc_and_returns_result(self) -> None:
        process = FakeProcess(['{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'])
        config = McpServerConfig(name="test", transport="stdio", command="server")
        transport = McpStdioTransport(config)

        with patch("uedev.mcp.transport.subprocess.Popen", return_value=process):
            result = transport.request("ping", {"x": 1})

        self.assertEqual(result, {"ok": True})
        sent = json.loads(process.stdin.writes[0])
        self.assertEqual(sent["method"], "ping")
        self.assertEqual(sent["params"], {"x": 1})

    def test_client_lists_and_calls_tools(self) -> None:
        class FakeTransport:
            def __init__(self) -> None:
                self.requests: list[tuple[str, dict[str, object] | None]] = []
                self.notifications: list[str] = []

            def request(self, method, params=None, timeout_seconds=None):
                self.requests.append((method, params))
                if method == "tools/list":
                    return {
                        "tools": [
                            {
                                "name": "get_status",
                                "description": "status",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    }
                if method == "tools/call":
                    return {"content": [{"type": "text", "text": "ok"}]}
                return {}

            def notify(self, method, params=None):
                self.notifications.append(method)

            def close(self):
                pass

        transport = FakeTransport()
        client = McpClient(McpServerConfig(name="unreal", transport="stdio", command="server"), transport)  # type: ignore[arg-type]

        tools = client.list_tools()
        result = client.call_tool("get_status", {"verbose": True})

        self.assertEqual(tools[0].name, "get_status")
        self.assertEqual(result.render(), "ok")
        self.assertEqual(transport.notifications, ["notifications/initialized"])
        self.assertEqual(transport.requests[-1], ("tools/call", {"name": "get_status", "arguments": {"verbose": True}}))


class McpRegistryRuntimeTests(unittest.TestCase):
    def test_registry_missing_system_config_is_no_configured_servers(self) -> None:
        with patch("uedev.mcp.registry.load_system_config", side_effect=ConfigError("Config file not found: missing")):
            registry = McpToolRegistry.from_system_config()

        self.assertEqual(registry.render_status(), "MCP: no configured servers.")

    def test_registry_registers_tools_and_renders_status(self) -> None:
        class FakeClient:
            def __init__(self, config):
                self.config = config

            def list_tools(self):
                return [
                    McpTool(
                        server_name=self.config.name,
                        name="get.editor-status",
                        agent_name="",
                        description="Get editor status",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, tool_name, arguments):
                return type("Result", (), {"render": lambda self: "editor ok"})()

        registry = McpToolRegistry({"unreal": McpServerConfig("unreal", "stdio", "server")})
        with patch("uedev.mcp.registry.McpClient", FakeClient):
            registry.startup_check()

        self.assertIn("mcp__unreal__get_editor-status", registry.tools)
        self.assertIn("connected", registry.render_status())
        self.assertEqual(registry.call_tool("mcp__unreal__get_editor-status", {}), "editor ok")

    def test_runtime_includes_mcp_tools_and_mcp_slash_status(self) -> None:
        class FakeRegistry:
            def tool_specs(self):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "mcp__unreal__get_status",
                            "description": "status",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            def handlers(self):
                return {"mcp__unreal__get_status": lambda data: "ok"}

            def render_status(self):
                return "MCP: 1 connected\n- unreal: connected, tools=1"

        with workspace_temp_dir() as temp:
            with patch("uedev.runtime.agent.McpToolRegistry.from_system_config", return_value=FakeRegistry()):
                runtime = AgentRuntime(AgentOptions("", 1, True, Path(temp), 120, False))

            output: list[str] = []
            self.assertIn("mcp__unreal__get_status", runtime.tools)
            self.assertTrue(any(spec["function"]["name"] == "mcp__unreal__get_status" for spec in runtime.tool_specs))
            self.assertEqual(runtime.tools["mcp__unreal__get_status"]({}), "ok")
            self.assertTrue(runtime.handle_slash_command("/mcp", emit=output.append))
            self.assertIn("1 connected", output[-1])

    def test_mcp_tool_permissions_follow_external_execution_rules(self) -> None:
        plan = classify_tool_permission("mcp__server__tool", {}, collaboration_mode="plan", permission_mode="full_access")
        read_only = classify_tool_permission("mcp__server__tool", {}, collaboration_mode="default", permission_mode="read_only")
        default = classify_tool_permission("mcp__server__tool", {}, collaboration_mode="default", permission_mode="default")

        self.assertEqual(plan.action, "deny")
        self.assertEqual(read_only.action, "ask")
        self.assertEqual(default.action, "allow")


if __name__ == "__main__":
    unittest.main()
