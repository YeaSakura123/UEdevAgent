from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
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
from uedev.state.config import ConfigError, ModelProfile, agent_dir, load_project_config, load_system_config, resolve_model_profile
from uedev.runtime.context import SUMMARY_PREFIX, compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
from uedev.ui.events import final_event, thinking_event, tool_error_event, tool_result_event, tool_start_event
from uedev.runtime.history import HistoryRecorder, load_display_history, load_history_file
from uedev.runtime.subagents import SubagentSpec
from uedev.llm.client import ChatMessage, ModelResponse, ToolCall, _serialize_message, call_model
from uedev.runtime.agent import (
    SLASH_COMMANDS,
    AgentOptions,
    AgentRuntime,
    SlashCommandCompleter,
    create_chat_prompt_options,
    defers_tool_confirmation,
    is_acknowledgement_answer,
    render_chat_banner,
    render_workspace_diff,
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
from uedev.tools.workspace import edit_file, read_file, write_file
from uedev.tools.worktrees import WorktreeManager


def write_system_config(
    config_path: Path,
    *,
    models: dict[str, dict[str, object]] | None = None,
    ue_engines: dict[str, dict[str, object]] | None = None,
    display: dict[str, object] | None = None,
    subagents: dict[str, object] | None = None,
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
    if display is not None:
        payload["display"] = display
    if subagents is not None:
        payload["subagents"] = subagents
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

    def test_serialize_reasoning_content_only_when_profile_requires_it(self) -> None:
        message = ChatMessage(role="assistant", content="done", reasoning_content="thinking")
        plain = ModelProfile(
            name="plain",
            model="plain-model",
            base_url="https://api.openai.com/v1",
            api_key="key",
        )
        reasoning = ModelProfile(
            name="deepseek",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            api_key="key",
            requires_reasoning_content=True,
        )

        self.assertNotIn("reasoning_content", _serialize_message(message, plain))
        self.assertEqual(_serialize_message(message, reasoning)["reasoning_content"], "thinking")
        tool_call_message = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
            reasoning_content="thinking",
        )
        self.assertEqual(_serialize_message(tool_call_message, reasoning)["reasoning_content"], "thinking")

    def test_serialize_tool_result_message(self) -> None:
        message = ChatMessage(role="tool", content="ok", tool_call_id="call_1", name="read_file")

        serialized = _serialize_message(message)

        self.assertEqual(serialized["role"], "tool")
        self.assertEqual(serialized["tool_call_id"], "call_1")

    def test_call_model_reads_reasoning_content_from_direct_attribute(self) -> None:
        captured: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="done", tool_calls=[], reasoning_content="thinking")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        profile = ModelProfile(
            name="deepseek",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            api_key="key",
            requires_reasoning_content=True,
        )

        with patch("uedev.llm.client.OpenAI", return_value=fake_client):
            response = call_model([ChatMessage(role="user", content="hello")], profile)

        self.assertEqual(response.reasoning_content, "thinking")
        self.assertNotIn("reasoning_content", captured["messages"][0])

    def test_call_model_reads_reasoning_content_from_model_extra(self) -> None:
        class FakeCompletions:
            def create(self, **kwargs):
                message = SimpleNamespace(content="done", tool_calls=[], model_extra={"reasoning_content": "extra thinking"})
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        profile = ModelProfile(
            name="deepseek",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            api_key="key",
        )

        with patch("uedev.llm.client.OpenAI", return_value=fake_client):
            response = call_model([ChatMessage(role="user", content="hello")], profile)

        self.assertEqual(response.reasoning_content, "extra thinking")


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

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test"))

            event_types = [event.type for event in events]
            self.assertEqual(event_types[:3], ["thinking", "tool_start", "tool_result"])
            self.assertEqual(event_types[-1], "final")
            self.assertEqual(events[1].name, "read_file")
            self.assertEqual(events[-1].message, "done")

    def test_run_turn_events_records_reasoning_content_on_tool_call_messages(self) -> None:
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse(
                    "",
                    [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
                    reasoning_content="thinking",
                ),
                ModelResponse("done"),
            ]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
                    list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test", history=history))

            loaded = load_history_file(history.path or Path())
            assistant_tool = next(message for message in loaded if message.tool_calls)
            self.assertEqual(assistant_tool.reasoning_content, "thinking")

    def test_run_turn_events_records_session_history(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", return_value=ModelResponse("done")):
                    events = list(runtime.run_turn_events(messages, "remember this", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "final")
            loaded = load_history_file(history.path or Path())
            rendered = "\n".join(message.content for message in loaded)
            self.assertIn("remember this", rendered)
            self.assertIn("done", rendered)

    def test_run_turn_events_compacts_before_new_goal_when_threshold_is_exceeded(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=2,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                    context_threshold=3000,
                )
            )
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="assistant", content="old observation " + ("x" * 8000)),
            ]
            responses = [ModelResponse("short summary"), ModelResponse("done")]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses) as mock_call:
                    events = list(runtime.run_turn_events(messages, "new task", turn_id="turn-test"))

            self.assertEqual(events[0].type, "compact")
            self.assertEqual(events[-1].type, "final")
            compaction_messages = mock_call.call_args_list[0].args[0]
            self.assertNotIn("new task", "\n".join(message.content for message in compaction_messages))
            normal_messages = mock_call.call_args_list[1].args[0]
            rendered = "\n".join(message.content for message in normal_messages)
            self.assertIn(SUMMARY_PREFIX, rendered)
            self.assertIn("new task", rendered)
            self.assertIn("tools", mock_call.call_args_list[1].kwargs)

    def test_default_context_threshold_uses_model_context_window(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "small": {
                        "model": "small-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 1000,
                    }
                },
            )
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

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                self.assertEqual(runtime._context_threshold(), 900)

    def test_explicit_context_threshold_overrides_model_context_window(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                    context_threshold=123,
                )
            )

            self.assertEqual(runtime._context_threshold(), 123)

    def test_context_slash_command_reports_current_context_usage(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "small": {
                        "model": "small-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 1000,
                    }
                },
            )
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                    context_threshold=700,
                )
            )
            messages = [
                ChatMessage(role="system", content="system prompt"),
                ChatMessage(role="user", content="hello"),
            ]
            original = list(messages)
            output: list[str] = []

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                self.assertTrue(runtime.handle_slash_command("/context", emit=output.append, messages=messages))

            rendered = output[-1]
            self.assertIn("Context:", rendered)
            self.assertIn("model: small-model (profile: small)", rendered)
            self.assertIn("estimated tokens:", rendered)
            self.assertIn("context window: 1,000", rendered)
            self.assertIn("context usage:", rendered)
            self.assertIn("auto compact threshold: 700", rendered)
            self.assertIn("threshold usage:", rendered)
            self.assertIn("remaining to threshold:", rendered)
            self.assertIn("remaining to window:", rendered)
            self.assertNotIn("approximate JSON length", rendered)
            self.assertEqual(messages, original)

    def test_context_slash_command_uses_model_window_default_threshold(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "small": {
                        "model": "small-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 1000,
                    }
                },
            )
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
            output: list[str] = []

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                self.assertTrue(
                    runtime.handle_slash_command(
                        "/context",
                        emit=output.append,
                        messages=[ChatMessage(role="system", content="system prompt")],
                    )
                )

            self.assertIn("auto compact threshold: 900", output[-1])

    def test_context_slash_command_requires_chat_messages_and_rejects_arguments(self) -> None:
        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=True,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            output: list[str] = []

            self.assertTrue(runtime.handle_slash_command("/context", emit=output.append))
            self.assertIn("Use /context inside chat", output[-1])

            self.assertTrue(runtime.handle_slash_command("/context now", emit=output.append, messages=[]))
            self.assertEqual(output[-1], "Usage: /context")

    def test_subagents_slash_command_lists_only_from_bare_command(self) -> None:
        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=True,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            output: list[str] = []

            self.assertTrue(runtime.handle_slash_command("/subagents", emit=output.append, messages=[]))
            self.assertIn("Main conversation", output[-1])
            self.assertFalse((Path(temp) / ".agent" / "sessions").exists())
            self.assertFalse((Path(temp) / ".agent" / "subagents").exists())

            self.assertTrue(runtime.handle_slash_command("/subagents main", emit=output.append, messages=[]))
            self.assertEqual(output[-1], "Usage: /subagents")

    def test_diff_slash_command_renders_git_and_perforce_status(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path, display={"diff_output_max_chars": 50})
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
            output: list[str] = []

            def fake_run(args, **kwargs):
                if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(args, 0, "true\n", "")
                if args == ["git", "status", "--short", "--branch"]:
                    return subprocess.CompletedProcess(args, 0, "## No commits yet on master\n M Source/A.cpp\nA  README.md\n?? Content/\n", "")
                if args == ["git", "diff", "--no-ext-diff"]:
                    return subprocess.CompletedProcess(args, 0, "x" * 80, "")
                if args == ["git", "diff", "--cached", "--no-ext-diff"]:
                    return subprocess.CompletedProcess(args, 0, "", "")
                raise AssertionError(f"unexpected command: {args}")

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.subprocess.run", side_effect=fake_run),
                patch(
                    "uedev.runtime.agent.p4_status",
                    return_value=json.dumps(
                        {
                            "ok": True,
                            "available": True,
                            "in_workspace": True,
                            "project_tracked": True,
                            "client_name": "UnrealCode_WS",
                            "client_root": str(root),
                            "user_name": "admin",
                            "server_address": "perforce:1666",
                            "project_depot_path": "//depot/Project.uproject",
                            "opened_count": 1,
                            "opened_preview": ["SHOULD_NOT_PRINT"],
                            "notes": [],
                        }
                    ),
                ) as status,
                patch(
                    "uedev.runtime.agent.p4_opened",
                    return_value=json.dumps(
                        {
                            "ok": True,
                            "status": "completed",
                            "command": "p4 opened",
                            "exit_code": 0,
                            "stdout": "SHOULD_NOT_PRINT",
                            "stderr": "",
                            "opened_count": 1,
                            "opened": ["//depot/A.cpp#1 - edit default change (text)"],
                        }
                    ),
                ) as opened,
            ):
                self.assertTrue(runtime.handle_slash_command("/diff", emit=output.append))

            rendered = output[-1]
            self.assertIn("branch: master (no commits)", rendered)
            self.assertIn("status: staged 1, unstaged 1, untracked 1", rendered)
            self.assertIn("note: untracked files are not included in git diff output.", rendered)
            self.assertIn("unstaged diff:", rendered)
            self.assertIn("staged diff: none", rendered)
            self.assertIn("truncated at 50 chars", rendered)
            self.assertIn("workspace: UnrealCode_WS", rendered)
            self.assertIn("project: tracked //depot/Project.uproject", rendered)
            self.assertIn("opened: 1", rendered)
            self.assertIn("edit", rendered)
            self.assertIn("text", rendered)
            self.assertIn("//depot/A.cpp", rendered)
            self.assertNotIn('"stdout"', rendered)
            self.assertNotIn("opened_preview", rendered)
            self.assertNotIn("SHOULD_NOT_PRINT", rendered)
            status.assert_called_once_with(root)
            opened.assert_called_once_with(root)

    def test_diff_slash_command_rejects_arguments(self) -> None:
        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=True,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            output: list[str] = []

            self.assertTrue(runtime.handle_slash_command("/diff Source/A.cpp", emit=output.append))

            self.assertEqual(output[-1], "Usage: /diff")

    def test_workspace_diff_continues_to_p4_when_not_git_repository(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)

            def fake_run(args, **kwargs):
                if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(args, 128, "", "not a git repository")
                raise AssertionError(f"unexpected command: {args}")

            with (
                patch("uedev.runtime.agent.subprocess.run", side_effect=fake_run),
                patch("uedev.runtime.agent.p4_status", return_value='{"ok": false, "available": false}') as status,
                patch("uedev.runtime.agent.p4_opened", return_value='{"ok": true, "opened_count": 0}') as opened,
            ):
                rendered = render_workspace_diff(root, 120, 20000)

            self.assertIn("Git: not a repository", rendered)
            self.assertIn("status: unavailable", rendered)
            self.assertIn("opened: none", rendered)
            status.assert_called_once_with(root)
            opened.assert_called_once_with(root)

    def test_workspace_diff_falls_back_to_raw_p4_output_for_invalid_json(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)

            def fake_run(args, **kwargs):
                if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(args, 128, "", "not a git repository")
                raise AssertionError(f"unexpected command: {args}")

            with (
                patch("uedev.runtime.agent.subprocess.run", side_effect=fake_run),
                patch("uedev.runtime.agent.p4_status", return_value="not-json-" + ("x" * 80)),
                patch("uedev.runtime.agent.p4_opened", return_value='{"ok": true, "opened": []}'),
            ):
                rendered = render_workspace_diff(root, 120, 30)

            self.assertIn("p4_status: raw output", rendered)
            self.assertIn("truncated at 30 chars", rendered)

    def test_workspace_diff_renders_p4_command_failure_without_duplicate_stdout(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)

            def fake_run(args, **kwargs):
                if args == ["git", "rev-parse", "--is-inside-work-tree"]:
                    return subprocess.CompletedProcess(args, 128, "", "not a git repository")
                raise AssertionError(f"unexpected command: {args}")

            with (
                patch("uedev.runtime.agent.subprocess.run", side_effect=fake_run),
                patch("uedev.runtime.agent.p4_status", return_value='{"ok": true, "available": true, "in_workspace": true, "project_tracked": false}'),
                patch(
                    "uedev.runtime.agent.p4_opened",
                    return_value=json.dumps(
                        {
                            "ok": False,
                            "command": "p4 opened",
                            "exit_code": 1,
                            "stdout": "duplicate stdout",
                            "stderr": "client unknown",
                        }
                    ),
                ),
            ):
                rendered = render_workspace_diff(root, 120, 20000)

            self.assertIn("opened: failed", rendered)
            self.assertIn("command: p4 opened", rendered)
            self.assertIn("exitCode: 1", rendered)
            self.assertIn("client unknown", rendered)
            self.assertNotIn("duplicate stdout", rendered)

    def test_compact_slash_command_rewrites_model_context(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
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
            messages = [
                ChatMessage(role="system", content=runtime.system_prompt),
                ChatMessage(role="user", content="old request"),
            ]
            history = HistoryRecorder(agent_dir(root), messages)
            output: list[str] = []

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", return_value=ModelResponse("manual summary")) as mock_call:
                    self.assertTrue(runtime.handle_slash_command("/compact", emit=output.append, messages=messages, history=history))

            self.assertIn("Conversation compacted", output[-1])
            self.assertIn(SUMMARY_PREFIX, "\n".join(message.content for message in messages))
            self.assertEqual((history.transcript_path or Path()).name, "transcript.jsonl")
            self.assertTrue((history.transcript_path or Path()).exists())
            self.assertFalse((agent_dir(root) / "transcripts").exists())
            self.assertEqual(mock_call.call_count, 1)
            self.assertNotIn("tools", mock_call.call_args.kwargs)

    def test_compact_transcript_includes_normal_final_answers(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", return_value=ModelResponse("previous answer")):
                    events = list(runtime.run_turn_events(messages, "first task", turn_id="turn-test", history=history))
                self.assertEqual(events[-1].type, "final")

                with patch("uedev.runtime.agent.call_model", return_value=ModelResponse("manual summary")):
                    self.assertTrue(
                        runtime.handle_slash_command(
                            "/compact",
                            emit=lambda _message: None,
                            messages=messages,
                            history=history,
                        )
                    )

            transcript = history.transcript_path or Path()
            transcript_text = transcript.read_text(encoding="utf-8")

            self.assertIn('"role": "assistant"', transcript_text)
            self.assertIn("previous answer", transcript_text)

    def test_session_compact_overwrites_transcript(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt), ChatMessage(role="user", content="first")]
            history = HistoryRecorder(agent_dir(root), messages)

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=[ModelResponse("summary one"), ModelResponse("summary two")]):
                    self.assertTrue(runtime.handle_slash_command("/compact", emit=lambda _message: None, messages=messages, history=history))
                    messages.append(ChatMessage(role="user", content="second"))
                    self.assertTrue(runtime.handle_slash_command("/compact", emit=lambda _message: None, messages=messages, history=history))

            transcript = history.transcript_path or Path()
            transcript_text = transcript.read_text(encoding="utf-8")

            self.assertEqual(transcript.name, "transcript.jsonl")
            self.assertEqual(len(list((history.session_dir or Path()).glob("transcript*.jsonl"))), 1)
            self.assertIn("second", transcript_text)

    def test_compact_tool_uses_model_summary_and_preserves_current_goal(self) -> None:
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="compact", arguments={})]),
                ModelResponse("tool summary"),
                ModelResponse("done"),
            ]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses) as mock_call:
                    events = list(runtime.run_turn_events(messages, "continue current task", turn_id="turn-test"))

            self.assertIn("compact", [event.type for event in events])
            self.assertEqual(events[-1].type, "final")
            rendered = "\n".join(message.content for message in messages)
            self.assertIn(SUMMARY_PREFIX, rendered)
            self.assertIn("continue current task", rendered)
            self.assertNotIn("tools", mock_call.call_args_list[1].kwargs)

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

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "use missing tool", turn_id="turn-test"))

            self.assertIn("tool_error", [event.type for event in events])
            self.assertEqual(events[-1].type, "final")
            self.assertTrue(any(message.role == "tool" for message in messages))

    def test_subagent_uses_configured_model_profile_and_records_history(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "main": {
                        "model": "main-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    },
                    "child": {
                        "model": "child-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    },
                },
                subagents={"model_profile": "child"},
            )
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse(
                    "",
                    [
                        ToolCall(
                            id="call_1",
                            name="subagent",
                            arguments={"agent_type": "explorer", "task": "inspect files", "inherit_context": False},
                        )
                    ],
                ),
                ModelResponse("done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", return_value=ModelResponse("child result")) as child_call,
            ):
                events = list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "final")
            child_profile = child_call.call_args.args[1]
            self.assertEqual(child_profile.name, "child")
            self.assertEqual(child_profile.model, "child-model")
            subagents_dir = (history.session_dir or Path()) / "subagents"
            records = runtime.subagents.list_records(subagents_dir)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "complete")
            self.assertEqual(records[0].model_profile, "child")
            self.assertEqual(records[0].model, "child-model")
            metadata = json.loads(Path(records[0].metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_profile"], "child")
            self.assertEqual(metadata["status"], "complete")
            self.assertIn("display_history_path", metadata)
            self.assertEqual(metadata["parent_session_id"], (history.session_dir or Path()).name)
            self.assertEqual(Path(records[0].history_path).name, "messages.jsonl")
            self.assertTrue(Path(records[0].history_path).is_relative_to(subagents_dir))
            self.assertFalse((agent_dir(root) / "subagents").exists())
            history_text = Path(records[0].history_path).read_text(encoding="utf-8")
            self.assertIn("inspect files", history_text)
            self.assertIn("child result", history_text)
            display_records = load_display_history(Path(records[0].display_history_path))
            self.assertEqual(display_records[0]["type"], "turn_start")
            self.assertEqual([record["event"]["type"] for record in display_records if record["type"] == "event"], ["thinking", "final"])
            self.assertIn("subagent_id:", "\n".join(message.content for message in messages if message.role == "tool"))

    def test_subagent_defaults_to_main_active_profile(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "main": {
                        "model": "main-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    }
                },
            )
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=2,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="subagent", arguments={"task": "inspect"})]),
                ModelResponse("done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", return_value=ModelResponse("child result")) as child_call,
            ):
                list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test", history=history))

            self.assertEqual(child_call.call_args.args[1].name, "main")
            self.assertEqual(runtime.subagents.list_records((history.session_dir or Path()) / "subagents")[0].model_profile, "main")

    def test_subagent_display_history_records_tool_errors(self) -> None:
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="subagent", arguments={"agent_type": "explorer", "task": "inspect"})]),
                ModelResponse("done"),
            ]
            child_responses = [
                ModelResponse(
                    "",
                    [ToolCall(id="child_call_1", name="write_file", arguments={"path": "x.txt", "content": "x"})],
                    reasoning_content="child thinking",
                ),
                ModelResponse("child done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", side_effect=child_responses),
            ):
                events = list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "final")
            record = runtime.subagents.list_records((history.session_dir or Path()) / "subagents")[0]
            child_messages = load_history_file(Path(record.history_path))
            child_tool_call = next(message for message in child_messages if message.tool_calls)
            self.assertEqual(child_tool_call.reasoning_content, "child thinking")
            display_records = load_display_history(Path(record.display_history_path))
            event_types = [item["event"]["type"] for item in display_records if item["type"] == "event"]

            self.assertIn("tool_start", event_types)
            self.assertIn("tool_error", event_types)

    def test_multiple_subagent_tool_calls_run_as_one_parallel_batch(self) -> None:
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse(
                    "",
                    [
                        ToolCall(id="call_1", name="subagent", arguments={"agent_type": "explorer", "task": "inspect A"}),
                        ToolCall(id="call_2", name="subagent", arguments={"agent_type": "explorer", "task": "inspect B"}),
                    ],
                ),
                ModelResponse("done"),
            ]
            barrier = threading.Barrier(2)

            def child_call(_messages, _profile, tools=None):
                barrier.wait(timeout=2)
                return ModelResponse("child complete")

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", side_effect=child_call) as child_model,
            ):
                events = list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "final")
            self.assertEqual(child_model.call_count, 2)
            records = runtime.subagents.list_records((history.session_dir or Path()) / "subagents")
            self.assertEqual(len(records), 2)
            self.assertEqual({record.status for record in records}, {"complete"})
            tool_messages = [message for message in messages if message.role == "tool" and message.name == "subagent"]
            self.assertEqual(len(tool_messages), 2)

    def test_subagents_are_scoped_to_parent_session(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=2,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            first_messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            second_messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            first_history = HistoryRecorder(agent_dir(root), first_messages)
            second_history = HistoryRecorder(agent_dir(root), second_messages)
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="subagent", arguments={"task": "inspect first"})]),
                ModelResponse("first done"),
                ModelResponse("", [ToolCall(id="call_2", name="subagent", arguments={"task": "inspect second"})]),
                ModelResponse("second done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", return_value=ModelResponse("child result")),
            ):
                list(runtime.run_turn_events(first_messages, "delegate first", turn_id="turn-first", history=first_history))
                list(runtime.run_turn_events(second_messages, "delegate second", turn_id="turn-second", history=second_history))

            first_records = runtime.subagents.list_records((first_history.session_dir or Path()) / "subagents")
            second_records = runtime.subagents.list_records((second_history.session_dir or Path()) / "subagents")

            self.assertEqual(len(first_records), 1)
            self.assertEqual(len(second_records), 1)
            self.assertIn("inspect first", first_records[0].task)
            self.assertIn("inspect second", second_records[0].task)

    def test_subagent_id_collision_retries_with_next_counter(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=2,
                    auto_approve=True,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            history = HistoryRecorder(agent_dir(root), [ChatMessage(role="system", content=runtime.system_prompt)])
            subagents_dir = history.ensure_session() / "subagents"
            (subagents_dir / "sa_123_1_deadbeef").mkdir(parents=True)

            class FixedUuid:
                hex = "deadbeefcafebabe"

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.subagents.time.time_ns", return_value=123),
                patch("uedev.runtime.subagents.uuid4", return_value=FixedUuid()),
                patch("uedev.runtime.subagents.call_model", return_value=ModelResponse("child result")),
            ):
                results = runtime.subagents.run_batch(
                    [SubagentSpec(agent_type="explorer", task="inspect collision")],
                    [],
                    subagents_dir,
                )

            self.assertEqual(results[0].record.id, "sa_123_2_deadbeef")
            self.assertTrue((subagents_dir / "sa_123_2_deadbeef").exists())
            self.assertEqual(runtime.subagents.list_records(subagents_dir)[0].id, "sa_123_2_deadbeef")

    def test_worker_subagent_requires_responsibility_and_paths(self) -> None:
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="subagent", arguments={"agent_type": "worker", "task": "edit"})]),
                ModelResponse("done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model") as child_model,
            ):
                events = list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test"))

            self.assertIn("tool_error", [event.type for event in events])
            self.assertFalse(child_model.called)
            self.assertIn("worker subagent requires responsibility", "\n".join(message.content for message in messages))

    def test_subagent_inherit_context_passes_main_context_to_child(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "main": {
                        "model": "main-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    },
                    "child": {
                        "model": "child-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 100,
                    },
                },
                subagents={"model_profile": "child"},
            )
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
                ChatMessage(role="user", content="important prior context"),
            ]
            responses = [
                ModelResponse(
                    "",
                    [ToolCall(id="call_1", name="subagent", arguments={"task": "inspect", "inherit_context": True})],
                ),
                ModelResponse("done"),
            ]

            with (
                patch("uedev.state.config.default_system_config_path", return_value=config_path),
                patch("uedev.runtime.agent.call_model", side_effect=responses),
                patch("uedev.runtime.subagents.call_model", return_value=ModelResponse("child result")) as child_call,
            ):
                list(runtime.run_turn_events(messages, "delegate", turn_id="turn-test"))

            child_messages = child_call.call_args.args[0]
            rendered = "\n".join(message.content for message in child_messages)
            self.assertIn("Main conversation context:", rendered)
            self.assertIn("important prior context", rendered)
            self.assertEqual(child_call.call_args.args[1].context_window, 100)

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
                ModelResponse("Done, I will follow this behavior."),
                ModelResponse("a.txt contains: hello"),
            ]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
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

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "make a plan", turn_id="turn-test"))

            self.assertEqual(events[-1].type, "final")
            self.assertEqual(events[-1].message, "<proposed_plan>\n# Plan\n</proposed_plan>")
            self.assertTrue(any(message.role == "system" and "Plan Mode final answers" in message.content for message in messages))

    def test_max_steps_forces_final_answer_without_tools(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            write_file(root, "a.txt", "hello")
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})]),
                ModelResponse("Forced final summary."),
            ]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses) as mock_call:
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "final")
            self.assertEqual(events[-1].message, "Forced final summary.")
            self.assertNotIn("tools", mock_call.call_args_list[1].kwargs)
            loaded = load_history_file(history.path or Path())
            self.assertEqual(loaded[-1].role, "assistant")
            self.assertEqual(loaded[-1].content, "Forced final summary.")

    def test_max_steps_tool_call_during_finalization_becomes_incomplete(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
            write_file(root, "a.txt", "hello")
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]
            history = HistoryRecorder(agent_dir(root), messages)
            responses = [
                ModelResponse("", [ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})]),
                ModelResponse("", [ToolCall(id="call_2", name="read_file", arguments={"path": "a.txt"})]),
            ]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=responses):
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test", history=history))

            self.assertEqual(events[-1].type, "stopped")
            self.assertEqual(events[-1].status, "incomplete")
            self.assertFalse(events[-1].is_error)
            self.assertIn("Forced finalization returned tool calls", events[-1].message)
            loaded = load_history_file(history.path or Path())
            self.assertFalse(any(message.content.startswith("Incomplete turn.") for message in loaded))

    def test_model_request_error_is_stopped_error(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)
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
            messages = [ChatMessage(role="system", content=runtime.system_prompt)]

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                with patch("uedev.runtime.agent.call_model", side_effect=RuntimeError("model exploded")):
                    events = list(runtime.run_turn_events(messages, "hello", turn_id="turn-test"))

            self.assertEqual(events[-1].type, "stopped")
            self.assertEqual(events[-1].status, "stopped")
            self.assertTrue(events[-1].is_error)
            self.assertIn("model exploded", events[-1].message)

class BackgroundTests(unittest.TestCase):
    def test_background_check_empty(self) -> None:
        with workspace_temp_dir() as temp:
            manager = BackgroundManager(Path(temp))
            self.assertIn("No background", manager.check())

class ToolRequirementTests(unittest.TestCase):
    def test_ue_confirmation_deferral_is_not_final(self) -> None:
        self.assertTrue(
            defers_tool_confirmation(
                'Run "D:\\Code\\myAgentCli\\examples\\hello_editor.py" using full_editor',
                "Please confirm startup, then I will continue.",
            )
        )

    def test_ordinary_answer_is_not_confirmation_deferral(self) -> None:
        self.assertFalse(defers_tool_confirmation("Explain what kind means", "kind is an internal template selector."))


    def test_acknowledgement_answer_is_not_valid_final_after_tools(self) -> None:
        self.assertTrue(is_acknowledgement_answer("Understood. I鈥檒l directly invoke the needed tool when required."))
        self.assertTrue(is_acknowledgement_answer("Done, I will follow this behavior."))
        self.assertFalse(is_acknowledgement_answer("Project exists, EngineAssociation is 5.7, Perforce is workspace/tracked."))
