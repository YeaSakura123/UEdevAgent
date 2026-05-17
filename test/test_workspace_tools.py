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
from uedev.context import (
    SUMMARY_PREFIX,
    build_compacted_history,
    build_compaction_request,
    compact_locally,
    estimate_tokens,
    micro_compact,
    repair_tool_call_messages,
)
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


class WorkspaceToolTests(unittest.TestCase):
    def test_read_write_edit_are_sandboxed(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            write_file(root, "a.txt", "hello")
            edit_file(root, "a.txt", "hello", "world")

            self.assertEqual(read_file(root, "a.txt"), "world")
            with self.assertRaises(ValueError):
                read_file(root, "../outside.txt")

    def test_edit_file_tool_accepts_edits_list(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            write_file(root, "test/main.py", "print('hello')\n")
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=root,
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)

            result = runtime.tools["edit_file"](
                {
                    "path": "test/main.py",
                    "edits": [{"oldText": "print('hello')", "newText": "print('updated')"}],
                }
            )

            self.assertIn("Edited", result)
            self.assertEqual(read_file(root, "test/main.py"), "print('updated')")

class SkillLoaderTests(unittest.TestCase):
    def test_load_skill_from_frontmatter(self) -> None:
        with workspace_temp_dir() as temp:
            skill_dir = Path(temp) / "skills" / "ue"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: ue\ndescription: UE workflow\n---\nUse UE Python.",
                encoding="utf-8",
            )

            loader = SkillLoader(Path(temp) / "skills")
            self.assertIn("ue", loader.descriptions())
            self.assertIn("Use UE Python", loader.load("ue"))

    def test_load_skill_normalizes_name(self) -> None:
        with workspace_temp_dir() as temp:
            skill_dir = Path(temp) / "skills" / "ue-editor"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: ue-editor\ndescription: UE workflow\n---\nUse UE Python.",
                encoding="utf-8",
            )

            loader = SkillLoader(Path(temp) / "skills")

            self.assertIn("Use UE Python", loader.load(" UE-EDITOR "))

class ContextTests(unittest.TestCase):
    def test_micro_compact_old_tool_results(self) -> None:
        messages = [ChatMessage(role="user", content=f"Tool result for: read_file\n{'x' * 5000}") for _ in range(10)]

        micro_compact(messages, keep_recent=2, max_content=100)

        self.assertIn("compacted", messages[0].content)
        self.assertGreater(estimate_tokens(messages), 0)

    def test_micro_compact_preserves_tool_message_identity(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
            ),
            ChatMessage(
                role="tool",
                content=f"Tool result for: read_file\n{'x' * 5000}",
                tool_call_id="call_1",
                name="read_file",
            ),
        ]

        micro_compact(messages, keep_recent=0, max_content=100)

        self.assertEqual(messages[1].role, "tool")
        self.assertEqual(messages[1].tool_call_id, "call_1")
        self.assertEqual(messages[1].name, "read_file")
        self.assertIn("compacted", messages[1].content)

    def test_repair_tool_call_messages_downgrades_damaged_history(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
            ),
            ChatMessage(role="user", content="Tool result for: read_file\n[older observation compacted]"),
        ]

        repair_tool_call_messages(messages)

        self.assertEqual(messages[0].role, "assistant")
        self.assertFalse(messages[0].tool_calls)
        self.assertIn("tool calls omitted", messages[0].content)

    def test_compact_locally_saves_transcript(self) -> None:
        with workspace_temp_dir() as temp:
            messages = [ChatMessage(role="user", content="hello")]
            compacted = compact_locally(messages, Path(temp), "test")

            self.assertEqual(len(compacted), 1)
            self.assertTrue(list(Path(temp).glob("transcript_*.jsonl")))

    def test_compact_locally_preserves_system_prompt(self) -> None:
        with workspace_temp_dir() as temp:
            messages = [
                ChatMessage(role="system", content="system rules"),
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="assistant", content="hi"),
            ]

            compacted = compact_locally(messages, Path(temp), "test")

            self.assertEqual(compacted[0].role, "system")
            self.assertEqual(compacted[0].content, "system rules")
            self.assertEqual(compacted[1].role, "user")
            self.assertIn("[Compressed locally: test]", compacted[1].content)

    def test_build_compaction_request_omits_runtime_state(self) -> None:
        messages = [
            ChatMessage(role="system", content="system rules"),
            ChatMessage(role="system", content="<runtime-state>\nmode"),
            ChatMessage(role="user", content="old request"),
        ]

        request = build_compaction_request(messages, "test")

        self.assertFalse(any(message.content.startswith("<runtime-state>") for message in request))
        self.assertIn("Compaction reason: test", request[-1].content)

    def test_build_compacted_history_uses_summary_prefix_and_filters_internal_users(self) -> None:
        messages = [
            ChatMessage(role="system", content="system rules"),
            ChatMessage(role="user", content="Working directory: D:/Code\nShell: PowerShell"),
            ChatMessage(role="user", content=f"{SUMMARY_PREFIX}\nold summary"),
            ChatMessage(role="user", content="keep this request"),
            ChatMessage(role="user", content="Tool result for: read_file\nold observation"),
            ChatMessage(role="system", content="<runtime-state>\nmode"),
        ]

        compacted = build_compacted_history(messages, "new summary")

        self.assertEqual(compacted[0].role, "system")
        self.assertEqual(compacted[0].content, "system rules")
        rendered = "\n".join(message.content for message in compacted)
        self.assertIn("keep this request", rendered)
        self.assertIn(f"{SUMMARY_PREFIX}\nnew summary", rendered)
        self.assertNotIn("Working directory:", rendered)
        self.assertNotIn("old summary", rendered)
        self.assertNotIn("Tool result for:", rendered)
        self.assertNotIn("<runtime-state>", rendered)

    def test_build_compacted_history_respects_user_token_budget(self) -> None:
        messages = [
            ChatMessage(role="system", content="system rules"),
            ChatMessage(role="user", content="small one"),
            ChatMessage(role="user", content="x" * 1000),
            ChatMessage(role="user", content="small two"),
        ]

        compacted = build_compacted_history(messages, "summary", max_user_tokens=50)
        rendered = "\n".join(message.content for message in compacted)

        self.assertIn("small one", rendered)
        self.assertIn("small two", rendered)
        self.assertNotIn("x" * 1000, rendered)

    def test_estimate_tokens_accepts_native_tool_calls(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
            )
        ]

        self.assertGreater(estimate_tokens(messages), 0)
