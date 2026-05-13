from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from contextlib import contextmanager


@contextmanager
def workspace_temp_dir():
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    yield str(path)



from uedev.background import BackgroundManager
from uedev.config import ConfigError, agent_dir, load_project_config, load_system_config, resolve_model_profile
from uedev.context import compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
from uedev.events import final_event, thinking_event, tool_error_event, tool_result_event, tool_start_event
from uedev.llm import ChatMessage, ModelResponse, ToolCall, _serialize_message
from uedev.loop import (
    SLASH_COMMANDS,
    AgentOptions,
    AgentRuntime,
    SlashCommandCompleter,
    create_chat_prompt_options,
    defers_tool_confirmation,
    is_acknowledgement_answer,
    render_chat_banner,
    render_slash_help,
)
from uedev.permissions import classify_shell_command
from uedev.prompts import (
    _join_sections,
    build_prompt_bundle,
    build_subagent_prompt,
    build_system_prompt as build_prompt_system_prompt,
    build_tool_confirmation_reminder,
)
from uedev.renderer import ConsoleRenderer, TuiRenderer
from uedev.shell import ShellResult, run_shell
from uedev.skills import SkillLoader
from uedev.tasks import TaskManager
from uedev.team import MessageBus, TeamManager
from uedev.tool_specs import get_tool_names, get_tool_specs
from uedev.workspace import edit_file, read_file, write_file
from uedev.worktrees import WorktreeManager


def write_system_config(config_path: Path, *, models: dict[str, dict[str, str]] | None = None, ue_engines: dict[str, dict[str, object]] | None = None) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": models
                or {
                    "first-model": {
                        "model": "gpt-test",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "test-key",
                    }
                },
                "ue": {"engines": ue_engines or {}},
            }
        ),
        encoding="utf-8",
    )


def create_ue_engine_root(root: Path, *, commandlet: bool = True, gui: bool = True) -> None:
    bin_dir = root / "Engine" / "Binaries" / "Win64"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if commandlet:
        (bin_dir / "UnrealEditor-Cmd.exe").write_text("", encoding="utf-8")
    if gui:
        (bin_dir / "UnrealEditor.exe").write_text("", encoding="utf-8")


class LlmMessageTests(unittest.TestCase):
    def test_serialize_assistant_tool_call_message(self) -> None:
        message = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
        )

        serialized = _serialize_message(message)

        self.assertEqual(serialized["role"], "assistant")
        self.assertEqual(serialized["tool_calls"][0]["function"]["name"], "read_file")

    def test_serialize_tool_result_message(self) -> None:
        message = ChatMessage(role="tool", content="ok", tool_call_id="call_1", name="read_file")

        serialized = _serialize_message(message)

        self.assertEqual(serialized["role"], "tool")
        self.assertEqual(serialized["tool_call_id"], "call_1")

class AgentEventLoopTests(unittest.TestCase):
    def test_run_turn_events_emits_tool_flow(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            write_file(root, "a.txt", "hello")
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=3,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content="read a.txt"),
            ]
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})]),
                ModelResponse("done"),
            ]

            with patch("uedev.config.default_system_config_path", return_value=config_path):
                with patch("uedev.loop.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test"))

            event_types = [event.type for event in events]
            self.assertEqual(event_types[:3], ["thinking", "tool_start", "tool_result"])
            self.assertEqual(event_types[-1], "final")
            self.assertEqual(events[1].name, "read_file")
            self.assertEqual(events[-1].message, "done")

    def test_run_turn_events_emits_tool_error(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=3,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content="use missing tool"),
            ]
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="missing_tool", arguments={})]),
                ModelResponse("failed cleanly"),
            ]

            with patch("uedev.config.default_system_config_path", return_value=config_path):
                with patch("uedev.loop.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "use missing tool", turn_id="turn-test"))

            self.assertIn("tool_error", [event.type for event in events])
            self.assertEqual(events[-1].type, "final")
            self.assertTrue(any(message.role == "tool" for message in messages))

    def test_run_turn_events_rejects_acknowledgement_final_after_tool_result(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            write_file(root, "a.txt", "hello")
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=4,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content="read a.txt"),
            ]
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})]),
                ModelResponse("已按你的要求执行，并会遵循该行为。"),
                ModelResponse("a.txt contains: hello"),
            ]

            with patch("uedev.config.default_system_config_path", return_value=config_path):
                with patch("uedev.loop.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test"))

            self.assertEqual(events[-1].type, "final")
            self.assertEqual(events[-1].message, "a.txt contains: hello")
            self.assertTrue(any(message.role == "system" and "Invalid final answer" in message.content for message in messages))

    def test_plan_mode_requires_proposed_plan_final(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=3,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            runtime.collaboration_mode = "plan"
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content="make a plan"),
            ]
            responses = [
                ModelResponse("plain plan"),
                ModelResponse("<proposed_plan>\n# Plan\n</proposed_plan>"),
            ]

            with patch("uedev.config.default_system_config_path", return_value=config_path):
                with patch("uedev.loop.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "make a plan", turn_id="turn-test"))

            self.assertEqual(events[-1].type, "final")
            self.assertEqual(events[-1].message, "<proposed_plan>\n# Plan\n</proposed_plan>")
            self.assertTrue(any(message.role == "system" and "Plan Mode final answers" in message.content for message in messages))

class BackgroundTests(unittest.TestCase):
    def test_background_check_empty(self) -> None:
        with workspace_temp_dir() as temp:
            manager = BackgroundManager(Path(temp))
            self.assertIn("No background", manager.check())

class ToolRequirementTests(unittest.TestCase):
    def test_ue_confirmation_deferral_is_not_final(self) -> None:
        self.assertTrue(
            defers_tool_confirmation(
                '执行"D:\\Code\\myAgentCli\\examples\\hello_editor.py"，使用 full_editor',
                "请确认启动，我就继续执行。",
            )
        )

    def test_ordinary_answer_is_not_confirmation_deferral(self) -> None:
        self.assertFalse(defers_tool_confirmation("解释 kind 是什么", "kind 是内部模板选择器。"))


    def test_acknowledgement_answer_is_not_valid_final_after_tools(self) -> None:
        self.assertTrue(is_acknowledgement_answer("Understood. I’ll directly invoke the needed tool when required."))
        self.assertTrue(is_acknowledgement_answer("已按你的要求执行，并会遵循该行为。"))
        self.assertFalse(is_acknowledgement_answer("项目存在，EngineAssociation 是 5.7，Perforce 为 workspace/tracked。"))
