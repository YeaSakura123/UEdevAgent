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


try:
    from prompt_toolkit.cursor_shapes import CursorShape
    from prompt_toolkit.document import Document
    from prompt_toolkit.shortcuts.prompt import CompleteStyle
except ModuleNotFoundError as error:
    raise unittest.SkipTest("prompt_toolkit is not installed") from error

from uedev.tools.background import BackgroundManager
from uedev.state.config import ConfigError, agent_dir, load_project_config, load_system_config, resolve_model_profile
from uedev.runtime.context import compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
from uedev.ui.events import compact_event, final_event, thinking_event, tool_error_event, tool_result_event, tool_start_event
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
from uedev.ui.tui import ChatTuiApplication
from uedev.tools.workspace import edit_file, read_file, write_file
from uedev.tools.worktrees import WorktreeManager


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


class PromptBuilderTests(unittest.TestCase):
    def test_system_prompt_includes_static_and_dynamic_sections(self) -> None:
        prompt = build_prompt_system_prompt(Path("ProjectRoot"), "PowerShell", "- ue-editor: UE workflow")

        self.assertIn("You are a UE development agent", prompt)
        self.assertIn("Perforce UE source control:", prompt)
        self.assertIn("use ue_doctor's Perforce result", prompt)
        self.assertIn("do not run shell `p4 info` just to detect whether the project uses Perforce", prompt)
        self.assertIn("Before modifying any Perforce-controlled file, use p4_checkout.", prompt)
        self.assertIn("Do not run shell `p4 submit`, and do not submit by default.", prompt)
        self.assertIn("p4_status, p4_file_state, p4_opened, p4_checkout", prompt)
        self.assertIn("prefer p4_* tools over shell p4 commands", prompt)
        self.assertIn("UE safety:", prompt)
        self.assertIn("Never answer with acknowledgements about future behavior", prompt)
        self.assertIn("Do not use todo_update to acknowledge instructions", prompt)
        self.assertIn("meaningful multi-step task tracking only", prompt)
        self.assertIn("call ue_doctor directly and do not call list_files or shell `p4 info`", prompt)
        self.assertIn("Only use shell `p4 info` for raw Perforce diagnostics", prompt)
        self.assertIn("Available skills:\n- ue-editor: UE workflow", prompt)
        self.assertIn("Working directory: ProjectRoot", prompt)
        self.assertIn("Shell: PowerShell", prompt)
        self.assertIn("never an inline runpy.run_path loader", prompt)

    def test_prompt_bundle_exposes_subagent_and_reminders(self) -> None:
        bundle = build_prompt_bundle(Path("ProjectRoot"), "PowerShell", "(no skills found)")

        self.assertEqual(bundle.subagent_prompt, build_subagent_prompt())
        self.assertEqual(bundle.tool_confirmation_reminder, build_tool_confirmation_reminder())
        self.assertIn("(no skills found)", bundle.system_prompt)

    def test_prompt_sections_filter_none_and_empty_content(self) -> None:
        prompt = _join_sections([lambda: "alpha", lambda: None, lambda: "   ", lambda: "beta"])

        self.assertEqual(prompt, "alpha\n\nbeta")

class RendererTests(unittest.TestCase):
    def test_tui_renderer_collapses_turn_after_final(self) -> None:
        stream = StringIO()
        renderer = TuiRenderer("banner", verbose=True, stream=stream)
        renderer.start_turn("turn-1", "read a file")

        renderer.render(thinking_event(1, 3, "turn-1"))
        renderer.render(tool_result_event("read_file", "hello", "turn-1"))
        renderer.render(final_event("done", "turn-1"))

        text = renderer.render_text()
        rendered = stream.getvalue()

        self.assertIn("thinking:\nThinking... (1/3)", text)
        self.assertIn("summary:\nWorked", text)
        self.assertIn("1 tool used", text)
        self.assertIn("assistant:\ndone", text)
        self.assertIn("assistant", rendered)

    def test_tui_renderer_renders_markdown_final_answer(self) -> None:
        stream = StringIO()
        renderer = TuiRenderer("banner", verbose=False, stream=stream)
        markdown = "# Title\n\n- item\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('x')\n```"

        renderer.start_turn("turn-1", "show markdown")
        renderer.render(final_event(markdown, "turn-1"))

        transcript = renderer.render_text()
        rendered = stream.getvalue()

        self.assertIn("assistant:\n# Title", transcript)
        self.assertIn("Title", rendered)
        self.assertIn("item", rendered)
        self.assertIn("print", rendered)

    def test_tui_renderer_renders_tool_errors(self) -> None:
        stream = StringIO()
        renderer = TuiRenderer("banner", verbose=False, stream=stream)
        renderer.start_turn("turn-1", "fail")

        renderer.render(tool_error_event("shell", "boom", "turn-1"))

        text = renderer.render_text()
        rendered = stream.getvalue()

        self.assertIn("tool_error:\nFailed shell\nboom", text)
        self.assertIn("Failed shell", rendered)

    def test_console_renderer_compacts_long_output(self) -> None:
        renderer = ConsoleRenderer(verbose=True, max_output_chars=24)

        line = renderer.format(tool_result_event("shell", "x" * 100))

        self.assertIn("...", line)
        self.assertLess(len(line), 80)

    def test_tui_renderer_transcript_flow_order(self) -> None:
        renderer = TuiRenderer("banner", verbose=True, stream=StringIO())

        renderer.print_banner()
        renderer.start_turn("turn-1", "read a file")
        renderer.render(thinking_event(1, 3, "turn-1"))
        renderer.render(tool_start_event("read_file", {"path": "a.txt"}, "turn-1"))
        renderer.render(tool_result_event("read_file", "hello", "turn-1"))
        renderer.render(final_event("done", "turn-1"))

        transcript = renderer.render_text()

        self.assertIn("user:\nread a file", transcript)
        self.assertLess(transcript.index("banner:\nbanner"), transcript.index("thinking:\nThinking"))
        self.assertLess(transcript.index("user:\nread a file"), transcript.index("thinking:\nThinking"))
        self.assertLess(transcript.index("thinking:\nThinking"), transcript.index("tool_start:\nRunning read_file"))
        self.assertLess(transcript.index("tool_start:\nRunning read_file"), transcript.index("tool_result:\nOK read_file"))
        self.assertLess(transcript.index("tool_result:\nOK read_file"), transcript.index("summary:\nWorked"))
        self.assertLess(transcript.index("summary:\nWorked"), transcript.index("assistant:\ndone"))

    def test_tui_renderer_records_compact_without_clearing_transcript(self) -> None:
        renderer = TuiRenderer("banner", verbose=True, stream=StringIO())

        renderer.print_banner()
        renderer.start_turn("turn-1", "new task")
        renderer.render(compact_event("Conversation compacted.", "turn-1"))
        renderer.render(final_event("done", "turn-1"))

        transcript = renderer.render_text()

        self.assertIn("banner:\nbanner", transcript)
        self.assertIn("user:\nnew task", transcript)
        self.assertIn("compact:\nConversation compacted.", transcript)
        self.assertIn("assistant:\ndone", transcript)

class TaskAndTeamTests(unittest.TestCase):
    def test_task_dependencies_clear_on_completion(self) -> None:
        with workspace_temp_dir() as temp:
            state_dir = agent_dir(Path(temp))
            manager = TaskManager(state_dir / "tasks")
            manager.create("A")
            manager.create("B", blocked_by=[1])
            manager.update(1, status="completed")

            self.assertNotIn("blockedBy=[1]", manager.list_all())

    def test_team_message_and_claim(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            state_dir = agent_dir(root)
            task_manager = TaskManager(state_dir / "tasks")
            bus = MessageBus(state_dir / "team")
            team = TeamManager(state_dir / "team", task_manager, bus)
            task_manager.create("Ready task")
            team.spawn("alice", "coder")

            self.assertIn("alice", team.list_all())
            self.assertIn("claimed task #1", team.claim_ready_task("alice"))
            bus.send("lead", "alice", "hello")
            self.assertEqual(bus.read_inbox("alice")[0]["content"], "hello")

    def test_team_state_tolerates_empty_or_bad_json(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            state_dir = agent_dir(root)
            task_manager = TaskManager(state_dir / "tasks")
            bus = MessageBus(state_dir / "team")
            team = TeamManager(state_dir / "team", task_manager, bus)
            team.config_path.write_text("", encoding="utf-8")
            team.requests_path.write_text("{bad", encoding="utf-8")
            (bus.inbox_dir / "lead.jsonl").write_text("{bad\n{\"content\":\"ok\"}\n", encoding="utf-8")

            self.assertIn("No teammates", team.list_all())
            self.assertEqual(bus.read_inbox("lead"), [{"content": "ok"}])
            self.assertIn("unknown shutdown request", self._raises_text(lambda: team.shutdown_response("missing", True)))

    def _raises_text(self, callback) -> str:
        try:
            callback()
        except Exception as error:
            return str(error)
        self.fail("expected callback to raise")
