from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.document import Document
from prompt_toolkit.shortcuts.prompt import CompleteStyle

from uedev.background import BackgroundManager
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
    render_chat_banner,
    render_slash_help,
)
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


class WorkspaceToolTests(unittest.TestCase):
    def test_read_write_edit_are_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_file(root, "a.txt", "hello")
            edit_file(root, "a.txt", "hello", "world")

            self.assertEqual(read_file(root, "a.txt"), "world")
            with self.assertRaises(ValueError):
                read_file(root, "../outside.txt")

    def test_edit_file_tool_accepts_edits_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
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
        with tempfile.TemporaryDirectory() as temp:
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
        with tempfile.TemporaryDirectory() as temp:
            skill_dir = Path(temp) / "skills" / "ue-editor"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: ue-editor\ndescription: UE workflow\n---\nUse UE Python.",
                encoding="utf-8",
            )

            loader = SkillLoader(Path(temp) / "skills")

            self.assertIn("Use UE Python", loader.load(" UE-EDITOR "))


class PromptBuilderTests(unittest.TestCase):
    def test_system_prompt_includes_static_and_dynamic_sections(self) -> None:
        prompt = build_prompt_system_prompt(Path("ProjectRoot"), "PowerShell", "- ue-editor: UE workflow")

        self.assertIn("You are a UE development agent", prompt)
        self.assertIn("Perforce UE source control:", prompt)
        self.assertIn("Before modifying any Perforce-controlled file, run `p4 edit <path>`.", prompt)
        self.assertIn("Do not run `p4 submit` unless the user explicitly asks for submit.", prompt)
        self.assertIn("UE safety:", prompt)
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
        with tempfile.TemporaryDirectory() as temp:
            messages = [ChatMessage(role="user", content="hello")]
            compacted = compact_locally(messages, Path(temp), "test")

            self.assertEqual(len(compacted), 1)
            self.assertTrue(list(Path(temp).glob("transcript_*.jsonl")))

    def test_compact_locally_preserves_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
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

    def test_estimate_tokens_accepts_native_tool_calls(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
            )
        ]

        self.assertGreater(estimate_tokens(messages), 0)


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
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
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

            with patch("uedev.loop.call_model", side_effect=responses):
                events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-test"))

            event_types = [event.type for event in events]
            self.assertEqual(event_types[:3], ["thinking", "tool_start", "tool_result"])
            self.assertEqual(event_types[-1], "final")
            self.assertEqual(events[1].name, "read_file")
            self.assertEqual(events[-1].message, "done")

    def test_run_turn_events_emits_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
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

            with patch("uedev.loop.call_model", side_effect=responses):
                events = list(runtime.run_turn_events(messages, "use missing tool", turn_id="turn-test"))

            self.assertIn("tool_error", [event.type for event in events])
            self.assertEqual(events[-1].type, "final")
            self.assertTrue(any(message.role == "tool" for message in messages))


class ShellAndApprovalTests(unittest.TestCase):
    def test_run_shell_returns_output_without_printing(self) -> None:
        class FakeProcess:
            returncode = 0

            def communicate(self, timeout=None):
                return "stdout text\n", "stderr text\n"

            def kill(self):
                pass

        stdout = StringIO()
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as temp:
            with patch("uedev.shell.subprocess.Popen", return_value=FakeProcess()):
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    result = run_shell("echo test", Path(temp), 1)

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(result.stdout, "stdout text\n")
        self.assertEqual(result.stderr, "stderr text\n")

    def test_runtime_uses_injected_approval_provider_for_shell(self) -> None:
        approvals: list[tuple[str, str]] = []

        def approve(command: str, reason: str) -> bool:
            approvals.append((command, reason))
            return True

        with tempfile.TemporaryDirectory() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                ),
                approval_provider=approve,
            )

            with patch("uedev.loop.run_shell", return_value=ShellResult("echo ok", 0, "ok\n", "")):
                result = runtime.tools["shell"]({"command": "echo ok", "reason": "test approval"})

        self.assertEqual(approvals, [("echo ok", "test approval")])
        self.assertIn("exitCode: 0", result)
        self.assertIn("ok", result)

    def test_runtime_rejects_shell_when_approval_provider_declines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                ),
                approval_provider=lambda command, reason: False,
            )

            result = runtime.tools["shell"]({"command": "echo no", "reason": "test rejection"})

        self.assertIn("rejected", result)


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

        self.assertNotIn("user:\nread a file", transcript)
        self.assertLess(transcript.index("banner:\nbanner"), transcript.index("thinking:\nThinking"))
        self.assertLess(transcript.index("thinking:\nThinking"), transcript.index("tool_start:\nRunning read_file"))
        self.assertLess(transcript.index("tool_start:\nRunning read_file"), transcript.index("tool_result:\nOK read_file"))
        self.assertLess(transcript.index("tool_result:\nOK read_file"), transcript.index("summary:\nWorked"))
        self.assertLess(transcript.index("summary:\nWorked"), transcript.index("assistant:\ndone"))


class TaskAndTeamTests(unittest.TestCase):
    def test_task_dependencies_clear_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = TaskManager(Path(temp) / ".tasks")
            manager.create("A")
            manager.create("B", blocked_by=[1])
            manager.update(1, status="completed")

            self.assertNotIn("blockedBy=[1]", manager.list_all())

    def test_team_message_and_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_manager = TaskManager(root / ".tasks")
            bus = MessageBus(root / ".team")
            team = TeamManager(root / ".team", task_manager, bus)
            task_manager.create("Ready task")
            team.spawn("alice", "coder")

            self.assertIn("alice", team.list_all())
            self.assertIn("claimed task #1", team.claim_ready_task("alice"))
            bus.send("lead", "alice", "hello")
            self.assertEqual(bus.read_inbox("alice")[0]["content"], "hello")

    def test_team_state_tolerates_empty_or_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_manager = TaskManager(root / ".tasks")
            bus = MessageBus(root / ".team")
            team = TeamManager(root / ".team", task_manager, bus)
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


class BackgroundTests(unittest.TestCase):
    def test_background_check_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = BackgroundManager(Path(temp))
            self.assertIn("No background", manager.check())


class SlashCommandTests(unittest.TestCase):
    def test_help_includes_command_descriptions(self) -> None:
        help_text = render_slash_help()

        self.assertIn("/help", help_text)
        self.assertIn("Show available chat slash commands.", help_text)
        self.assertIn("/ue doctor", help_text)
        self.assertIn("Inspect Unreal Engine project and editor configuration.", help_text)
        self.assertIn("/clear", help_text)
        self.assertIn("Reset the current chat conversation context.", help_text)

    def test_chat_banner_includes_runtime_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=False,
                cwd=Path(temp),
                timeout_seconds=120,
                verbose=False,
            )

            banner = render_chat_banner(options)

            self.assertIn("uedev", banner)
            self.assertIn("model:", banner)
            self.assertIn(str(Path(temp)), banner)
            self.assertIn('Type "/" for commands', banner)

    def test_slash_completer_returns_all_commands_for_slash(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/"), None))

        self.assertEqual([completion.text for completion in completions], [command for command, _ in SLASH_COMMANDS])
        self.assertIn("Show available chat slash commands.", str(completions[0].display_meta))

    def test_slash_completer_filters_by_prefix(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/ue"), None))

        self.assertEqual([completion.text for completion in completions], ["/ue doctor"])

    def test_slash_completer_matches_fuzzy_ue_doctor(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/ud"), None))

        self.assertEqual([completion.text for completion in completions], ["/ue doctor"])

    def test_slash_completer_prefers_direct_doctor_match(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/doc"), None))

        self.assertEqual(completions[0].text, "/doctor")
        self.assertIn("Inspect Unreal Engine project", str(completions[0].display_meta))

    def test_chat_prompt_options_enable_block_cursor_and_completion(self) -> None:
        options = create_chat_prompt_options()

        self.assertTrue(options["complete_while_typing"])
        self.assertEqual(options["complete_style"], CompleteStyle.MULTI_COLUMN)
        self.assertEqual(options["cursor"].get_cursor_shape(None), CursorShape.BLINKING_BLOCK)


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


class ToolSpecTests(unittest.TestCase):
    def test_native_tool_specs_match_runtime_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=Path(temp),
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)

            self.assertEqual(get_tool_names(), set(runtime.tools))
            self.assertEqual(runtime.system_prompt, runtime.prompt_bundle.system_prompt)
            self.assertIn("UE safety:", runtime.system_prompt)

    def test_ue_run_python_schema_hides_internal_execution_flags(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        properties = specs["ue_run_python"]["function"]["parameters"]["properties"]
        description = specs["ue_run_python"]["function"]["description"]

        self.assertIn("script", properties)
        self.assertIn("script_path", properties)
        self.assertIn("mode", properties)
        self.assertIn("_uedev_result", description)
        self.assertIn("_uedev_emit", description)
        self.assertIn("do not pass inline runpy.run_path loader scripts", description)
        self.assertNotIn("kind", properties)
        self.assertNotIn("execute", properties)


class UeRuntimeToolTests(unittest.TestCase):
    def _runtime(self, cwd: Path) -> AgentRuntime:
        return AgentRuntime(
            AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=cwd,
                timeout_seconds=120,
                verbose=False,
            )
        )

    def test_ue_doctor_uses_explicit_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launch_dir = root / "agent"
            ue_dir = root / "UEAgentDemo"
            launch_dir.mkdir()
            ue_dir.mkdir()
            (ue_dir / "UEAgentDemo.uproject").write_text("{}", encoding="utf-8")

            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=launch_dir,
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)

            with patch.dict("os.environ", {}, clear=True):
                result = runtime.tools["ue_doctor"]({"cwd": str(ue_dir)})

            self.assertIn(str((ue_dir / "UEAgentDemo.uproject").resolve()), result)

    def test_ue_run_python_reads_script_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script_path = root / "hello_editor.py"
            script_path.write_text("print('hello from file')\n", encoding="utf-8")
            runtime = self._runtime(root)

            script = runtime._resolve_ue_script({"script_path": str(script_path)}, root)

            self.assertEqual(script, "print('hello from file')\n")

    def test_ue_run_python_script_input_returns_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script_path = root / "hello_editor.py"
            script_path.write_text("print('hello from file')\n", encoding="utf-8")
            runtime = self._runtime(root)

            script, source = runtime._resolve_ue_script_input({"script_path": str(script_path)}, root)

            self.assertEqual(script, "print('hello from file')\n")
            self.assertEqual(source, script_path.resolve())

    def test_ue_run_python_accepts_inline_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = self._runtime(root)

            script = runtime._resolve_ue_script({"script": "print('hello inline')"}, root)

            self.assertEqual(script, "print('hello inline')")

    def test_ue_run_python_rejects_inline_runpy_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = self._runtime(root)

            with self.assertRaisesRegex(ValueError, "Use script_path"):
                runtime._resolve_ue_script(
                    {
                        "script": "import runpy\nrunpy.run_path('Scripts/hello.py', run_name='__main__')\n",
                    },
                    root,
                )

    def test_ue_run_python_allows_inline_business_script_with_runpy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = self._runtime(root)
            inline_script = (
                "import runpy\n"
                "print('before child')\n"
                "runpy.run_path('Scripts/child.py', run_name='__main__')\n"
            )

            script = runtime._resolve_ue_script({"script": inline_script}, root)

            self.assertEqual(script, inline_script)

    def test_ue_run_python_requires_script_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = self._runtime(root)

            with self.assertRaises(ValueError):
                runtime._resolve_ue_script({}, root)

    def test_ue_run_python_accepts_mode_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(Path(temp))

            self.assertEqual(runtime._resolve_ue_mode("cmd"), "commandlet")
            self.assertEqual(runtime._resolve_ue_mode("editor"), "full_editor")


class WorktreeTests(unittest.TestCase):
    def test_worktree_index_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = TaskManager(root / ".tasks")
            manager = WorktreeManager(root, root / ".worktrees", tasks)

            self.assertIn("No managed", manager.list_all())


if __name__ == "__main__":
    unittest.main()
