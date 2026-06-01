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



from uedev.tools.background import BackgroundManager
from uedev.state.config import ConfigError, agent_dir, load_project_config, load_system_config, resolve_model_profile
from uedev.runtime.context import (
    SUMMARY_PREFIX,
    build_compacted_history,
    build_compaction_request,
    compact_locally,
    estimate_tokens,
    micro_compact,
    repair_tool_call_messages,
)
from uedev.ui.events import final_event, thinking_event, tool_error_event, tool_result_event, tool_start_event
from uedev.llm.client import ChatMessage, ModelResponse, ToolCall, _serialize_message
from uedev.runtime.agent import (
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
from uedev.policy.permissions import classify_shell_command
from uedev.runtime.prompts import (
    _join_sections,
    build_prompt_bundle,
    build_subagent_prompt,
    build_system_prompt as build_prompt_system_prompt,
    build_tool_confirmation_reminder,
)
from uedev.ui.renderer import ConsoleRenderer, TuiRenderer
from uedev.tools.shell import ShellResult, run_shell
from uedev.runtime.skills import SkillLoader
from uedev.state.tasks import TaskManager
from uedev.state.team import MessageBus, TeamManager
from uedev.tools.specs import get_tool_names, get_tool_specs
from uedev.tools.workspace import edit_file, grep, list_files, read_file, safe_path, write_file
from uedev.tools.worktrees import WorktreeManager


def write_system_config(
    config_path: Path,
    *,
    models: dict[str, dict[str, str]] | None = None,
    ue_engines: dict[str, dict[str, object]] | None = None,
    workspace: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
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
    if workspace is not None:
        payload["workspace"] = workspace
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload),
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

    def test_workspace_tools_exclude_internal_and_generated_dirs(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            for directory in [".agent", ".git", ".vs", "Binaries", "Intermediate", "Saved", "DerivedDataCache"]:
                hidden = root / directory
                hidden.mkdir()
                (hidden / "hidden.txt").write_text("hidden", encoding="utf-8")
            source = root / "Source"
            source.mkdir()
            (source / "Game.cpp").write_text("code", encoding="utf-8")

            listed = list_files(root).replace("\\", "/")

            self.assertIn("Source/Game.cpp", listed)
            for directory in [".agent", ".git", ".vs", "Binaries", "Intermediate", "Saved", "DerivedDataCache"]:
                self.assertNotIn(f"{directory}/", listed)

    def test_workspace_tools_reject_direct_access_to_excluded_dirs(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            (root / ".agent").mkdir()
            (root / ".agent" / "messages.jsonl").write_text("secret", encoding="utf-8")
            (root / ".vs").mkdir()
            (root / ".vs" / "state.txt").write_text("state", encoding="utf-8")
            (root / ".git").mkdir()

            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                list_files(root, ".agent")
            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                read_file(root, ".agent/messages.jsonl")
            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                write_file(root, ".git/config", "config")
            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                edit_file(root, ".vs/state.txt", "state", "updated")

    def test_safe_path_does_not_resolve_workspace_links(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            source = root / "SourceContent"
            source.mkdir()
            (source / "Map.umap").write_text("asset", encoding="utf-8")
            worktree = root / "Worktree"
            worktree.mkdir()
            link = worktree / "Content"
            try:
                link.symlink_to(source, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are not available in this environment")

            path = safe_path(worktree, "Content")

            self.assertEqual(path, worktree / "Content")
            self.assertIn("Content", list_files(worktree, "Content"))

    def test_runtime_uses_configured_workspace_excludes(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path, workspace={"excluded_dirs": ["Hidden"]})
            (root / "Hidden").mkdir()
            (root / "Hidden" / "secret.txt").write_text("secret", encoding="utf-8")
            (root / "Visible").mkdir()
            (root / "Visible" / "ok.txt").write_text("ok", encoding="utf-8")

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(
                    AgentOptions(
                        task="",
                        max_steps=1,
                        auto_approve=True,
                        cwd=root,
                        timeout_seconds=120,
                        verbose=False,
                    )
                )

            listed = runtime.tools["list_files"]({"path": ".", "limit": 200}).replace("\\", "/")

            self.assertIn("Visible/ok.txt", listed)
            self.assertNotIn("Hidden/secret.txt", listed)
            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                runtime.tools["read_file"]({"path": "Hidden/secret.txt"})

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

    def test_grep_python_fallback_searches_text_with_locations(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            source = root / "Source"
            source.mkdir()
            (source / "Game.cpp").write_text("alpha\nbeta target\n", encoding="utf-8")

            with patch("uedev.tools.workspace.subprocess.run", side_effect=FileNotFoundError):
                result = grep(root, "target", excluded_dirs=())

            self.assertIn("Source", result)
            self.assertIn("Game.cpp:2:6: beta target", result)

    def test_grep_honors_glob_limit_case_and_output_modes(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            (root / "a.py").write_text("Target one\nTARGET two\n", encoding="utf-8")
            (root / "b.txt").write_text("Target ignored\n", encoding="utf-8")

            with patch("uedev.tools.workspace.subprocess.run", side_effect=FileNotFoundError):
                content = grep(root, "target", glob="*.py", limit=1, case_sensitive=False, excluded_dirs=())
                files = grep(root, "target", glob="*.py", output_mode="files", case_sensitive=False, excluded_dirs=())
                counts = grep(root, "target", glob="*.py", output_mode="count", case_sensitive=False, excluded_dirs=())

            self.assertIn("a.py:1:1: Target one", content)
            self.assertIn("... (1 more matches)", content)
            self.assertEqual(files, "a.py")
            self.assertEqual(counts, "a.py: 2")
            self.assertNotIn("b.txt", content)

    def test_grep_rejects_escaped_or_excluded_paths_and_skips_excluded_dirs(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            (root / ".agent").mkdir()
            (root / ".agent" / "secret.txt").write_text("needle", encoding="utf-8")
            (root / "Source").mkdir()
            (root / "Source" / "visible.txt").write_text("needle", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Path escapes workspace"):
                grep(root, "needle", "../outside")
            with self.assertRaisesRegex(ValueError, "Path is excluded"):
                grep(root, "needle", ".agent")

            with patch("uedev.tools.workspace.subprocess.run", side_effect=FileNotFoundError):
                result = grep(root, "needle")

            self.assertIn("Source", result)
            self.assertIn("visible.txt", result)
            self.assertNotIn("secret.txt", result)

    def test_grep_skips_binary_content_and_includes_ue_asset_path_matches(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            content = root / "Content"
            content.mkdir()
            (content / "Vehicle.uasset").write_bytes(b"\0needle inside binary")
            (root / "blob.bin").write_bytes(b"\0needle inside binary")

            with patch("uedev.tools.workspace.subprocess.run", side_effect=FileNotFoundError):
                asset_result = grep(root, "Vehicle")
                binary_result = grep(root, "needle", include_asset_paths=False)

            self.assertIn("Vehicle.uasset: asset path match", asset_result)
            self.assertEqual(binary_result, "(no matches)")

    def test_grep_parses_ripgrep_json_results(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            (root / "Source").mkdir()
            (root / "Source" / "Game.cpp").write_text("int target;\n", encoding="utf-8")
            stdout = json.dumps(
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "Source/Game.cpp"},
                        "lines": {"text": "int target;\n"},
                        "line_number": 7,
                        "submatches": [{"match": {"text": "target"}, "start": 4, "end": 10}],
                    },
                }
            )
            completed = type("Completed", (), {"returncode": 0, "stdout": stdout + "\n", "stderr": ""})()

            with patch("uedev.tools.workspace.subprocess.run", return_value=completed) as run:
                result = grep(root, "target", glob="*.cpp", excluded_dirs=())

            command = run.call_args.args[0]
            self.assertIn("--json", command)
            self.assertIn("-g", command)
            self.assertIn("Source", result)
            self.assertIn("Game.cpp:7:5: int target;", result)

    def test_grep_invalid_regex_is_clear(self) -> None:
        with workspace_temp_dir() as temp:
            with self.assertRaisesRegex(ValueError, "Invalid grep pattern"):
                grep(Path(temp), "[")

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

    def test_repair_tool_call_messages_downgrades_missing_reasoning_for_thinking_models(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
            ),
            ChatMessage(role="tool", content="ok", tool_call_id="call_1", name="read_file"),
            ChatMessage(role="user", content="next"),
        ]

        repair_tool_call_messages(messages, require_reasoning_content=True)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "assistant")
        self.assertFalse(messages[0].tool_calls)
        self.assertIn("tool calls omitted", messages[0].content)
        self.assertEqual(messages[1].content, "next")

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
