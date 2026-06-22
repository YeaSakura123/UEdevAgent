from __future__ import annotations

import json
import subprocess
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
from uedev.runtime.context import estimate_tokens, micro_compact, repair_tool_call_messages
from uedev.ui.events import final_event, thinking_event, tool_error_event, tool_result_event, tool_start_event
from uedev.llm.client import ChatMessage, ModelResponse, ToolCall, _serialize_message
from uedev.runtime.history import (
    HistoryRecorder,
    load_display_history,
    load_history_file,
    load_session_metadata,
    update_session_active_plan,
)
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
from uedev.ui.tui import ChatTuiApplication
from uedev.tools.shell import ShellResult, run_shell
from uedev.runtime.skills import SkillLoader
from uedev.state.tasks import TaskManager
from uedev.tools.specs import get_tool_names, get_tool_specs
from uedev.tools.workspace import edit_file, read_file, write_file
from uedev.tools.worktrees import WorktreeManager


def write_system_config(
    config_path: Path,
    *,
    models: dict[str, dict[str, object]] | None = None,
    ue_engines: dict[str, dict[str, object]] | None = None,
    display: dict[str, object] | None = None,
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


class SlashCommandTests(unittest.TestCase):
    def test_help_includes_command_descriptions(self) -> None:
        help_text = render_slash_help()

        self.assertIn("/help", help_text)
        self.assertIn("Show available chat slash commands.", help_text)
        self.assertIn("/context", help_text)
        self.assertIn("Show current conversation context usage.", help_text)
        self.assertIn("/diff", help_text)
        self.assertIn("Show Git and Perforce workspace changes.", help_text)
        self.assertIn("/worktree", help_text)
        self.assertIn("/model", help_text)
        self.assertIn("/plan", help_text)
        self.assertIn("/permissions", help_text)
        self.assertIn("/history", help_text)
        self.assertIn("Load a previous conversation from this project.", help_text)
        self.assertIn("/ue doctor", help_text)
        self.assertIn("Inspect Unreal Engine project and editor configuration.", help_text)
        self.assertIn("/compact", help_text)
        self.assertIn("Compact the current conversation context.", help_text)
        self.assertIn("/clear", help_text)
        self.assertIn("Reset the current chat conversation context.", help_text)
        self.assertIn("/exit", help_text)
        self.assertIn("Exit interactive chat.", help_text)

    def test_chat_banner_includes_runtime_details(self) -> None:
        with workspace_temp_dir() as temp:
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
            self.assertNotIn("Tip:", banner)

    def test_slash_completer_returns_all_commands_for_slash(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/"), None))

        self.assertEqual([completion.text for completion in completions], [command for command, _ in SLASH_COMMANDS])
        self.assertIn("Show available chat slash commands.", str(completions[0].display_meta))

    def test_slash_completer_filters_by_prefix(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/ue"), None))

        self.assertEqual([completion.text for completion in completions], ["/ue doctor"])

    def test_slash_completer_filters_diff_command(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/dif"), None))

        self.assertEqual([completion.text for completion in completions], ["/diff"])

    def test_slash_completer_filters_context_command(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/con"), None))

        self.assertEqual([completion.text for completion in completions], ["/context"])

    def test_slash_completer_matches_fuzzy_ue_doctor(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/ud"), None))

        self.assertEqual([completion.text for completion in completions], ["/ue doctor"])

    def test_slash_completer_prefers_direct_doctor_match(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/doc"), None))

        self.assertEqual(completions[0].text, "/doctor")
        self.assertIn("Inspect Unreal Engine project", str(completions[0].display_meta))

    def test_slash_completer_lists_permission_modes(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/permissions "), None))

        self.assertEqual(
            [completion.text for completion in completions],
            [
                "/permissions read-only",
                "/permissions default",
                "/permissions auto-review",
                "/permissions full-access",
            ],
        )
        self.assertIn("Can read files", str(completions[0].display_meta))

    def test_slash_completer_filters_permission_modes(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/permissions a"), None))

        self.assertEqual([completion.text for completion in completions], ["/permissions auto-review"])

    def test_model_slash_command_lists_switches_and_resets_project_model(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "fast": {
                        "model": "fast-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "main-key",
                    },
                    "gpt-alt": {
                        "model": "gpt-alt-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "alt-key",
                    },
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
                self.assertTrue(runtime.handle_slash_command("/model", emit=output.append))
                self.assertIn("fast", output[-1])
                self.assertIn("default", output[-1])

                self.assertTrue(runtime.handle_slash_command("/model gpt-alt", emit=output.append))
                self.assertIn("Active model set to gpt-alt", output[-1])
                project_config = json.loads((agent_dir(root) / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(project_config["active_model"], "gpt-alt")
                self.assertEqual(runtime.current_model_profile().name, "gpt-alt")

                self.assertTrue(runtime.handle_slash_command("/model reset", emit=output.append))
                self.assertIn("Active model reset to default profile fast", output[-1])
                self.assertEqual(runtime.current_model_profile().name, "fast")

    def test_plan_slash_command_switches_collaboration_mode(self) -> None:
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

            self.assertTrue(runtime.handle_slash_command("/plan", emit=output.append))
            self.assertEqual(runtime.collaboration_mode, "plan")
            self.assertIn("Plan Mode enabled", output[-1])

            self.assertTrue(runtime.handle_slash_command("/plan status", emit=output.append))
            self.assertIn("plan", output[-1])

            self.assertTrue(runtime.handle_slash_command("/plan off", emit=output.append))
            self.assertEqual(runtime.collaboration_mode, "default")

    def test_plan_approve_updates_active_plan_and_leaves_plan_mode(self) -> None:
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
            runtime.collaboration_mode = "plan"
            history = HistoryRecorder(agent_dir(root), [ChatMessage(role="system", content=runtime.system_prompt)])
            session_dir = history.ensure_session()

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                record = runtime.plan_manager.save_proposed_plan(session_dir.name, "turn-1", "# Ready Plan")
                update_session_active_plan(session_dir, record.to_dict())
                output: list[str] = []

                self.assertTrue(runtime.handle_slash_command("/plan approve", emit=output.append, history=history))

            metadata = load_session_metadata(session_dir)
            display_records = load_display_history(history.display_path or Path())
            self.assertEqual(runtime.collaboration_mode, "default")
            self.assertIn("Plan approved", output[-1])
            self.assertEqual(metadata["active_plan"]["status"], "approved")
            self.assertEqual(display_records[-1]["event"]["type"], "plan")
            self.assertEqual(display_records[-1]["event"]["status"], "approved")

    def test_plan_reject_updates_active_plan_and_stays_in_plan_mode(self) -> None:
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
            history = HistoryRecorder(agent_dir(root), [ChatMessage(role="system", content=runtime.system_prompt)])
            session_dir = history.ensure_session()

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                record = runtime.plan_manager.save_proposed_plan(session_dir.name, "turn-1", "# Needs Work")
                update_session_active_plan(session_dir, record.to_dict())
                output: list[str] = []

                self.assertTrue(runtime.handle_slash_command("/plan reject", emit=output.append, history=history))

            metadata = load_session_metadata(session_dir)
            self.assertEqual(runtime.collaboration_mode, "plan")
            self.assertIn("Plan rejected", output[-1])
            self.assertEqual(metadata["active_plan"]["status"], "rejected")

    def test_paln_is_not_supported(self) -> None:
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

            self.assertTrue(runtime.handle_slash_command("/paln", emit=output.append))

            self.assertEqual(runtime.collaboration_mode, "default")
            self.assertIn("Unknown slash command", output[-1])

    def test_worktree_slash_command_requires_interactive_chat(self) -> None:
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

            self.assertTrue(runtime.handle_slash_command("/worktree", emit=output.append))
            self.assertEqual(output[-1], "Use /worktree in interactive chat to create a UE Git linked worktree.")

            self.assertTrue(runtime.handle_slash_command("/worktree name", emit=output.append))
            self.assertEqual(output[-1], "Usage: /worktree")

    def test_permissions_slash_command_switches_session_mode(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=root,
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            output: list[str] = []

            self.assertTrue(runtime.handle_slash_command("/permissions", emit=output.append))
            self.assertIn("Permission mode: default", output[-1])
            self.assertIn("session only", output[-1])

            self.assertTrue(runtime.handle_slash_command("/permissions auto-review", emit=output.append))
            self.assertEqual(runtime.permission_mode, "auto_review")
            self.assertIn("Permission mode set to auto-review", output[-1])
            self.assertFalse((agent_dir(root) / "config.json").exists())

            self.assertTrue(runtime.handle_slash_command("/permissions invalid", emit=output.append))
            self.assertIn("Unknown permission mode", output[-1])

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

    def test_chat_prompt_options_enable_block_cursor_and_completion(self) -> None:
        options = create_chat_prompt_options()

        self.assertTrue(options["complete_while_typing"])
        self.assertEqual(options["complete_style"], CompleteStyle.COLUMN)
        self.assertEqual(options["cursor"].get_cursor_shape(None), CursorShape.BLINKING_BLOCK)

    def test_tui_status_toolbar_and_shift_tab_exit_hook(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "fast": {
                        "model": "fast-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    }
                },
            )
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=root,
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)
            app = ChatTuiApplication(options, runtime, "banner", SlashCommandCompleter())

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                toolbar = "".join(fragment[1] for fragment in app.status_bottom_toolbar())

                self.assertIn("fast-model", toolbar)
                self.assertIn(str(root), toolbar)
                self.assertNotIn("Plan mode", toolbar)

                runtime.collaboration_mode = "plan"
                toolbar = "".join(fragment[1] for fragment in app.status_bottom_toolbar())

                self.assertIn("fast-model", toolbar)
                self.assertIn(str(root), toolbar)
                self.assertIn("Plan mode", toolbar)
                self.assertTrue(app.exit_plan_mode())
                self.assertEqual(runtime.collaboration_mode, "default")

class ToolSpecTests(unittest.TestCase):
    def test_native_tool_specs_match_runtime_handlers(self) -> None:
        with workspace_temp_dir() as temp:
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

    def test_grep_schema_declares_structured_search(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        properties = specs["grep"]["function"]["parameters"]["properties"]
        description = specs["grep"]["function"]["description"]

        for name in ["pattern", "path", "glob", "limit", "case_sensitive", "output_mode", "include_asset_paths"]:
            self.assertIn(name, properties)
        self.assertIn("structured", description)
        self.assertIn("UE asset path", description)

    def test_ue_build_schema_declares_fixed_editor_build(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        properties = specs["ue_build"]["function"]["parameters"]["properties"]
        description = specs["ue_build"]["function"]["description"]

        self.assertIn("cwd", properties)
        self.assertIn("timeout_seconds", properties)
        self.assertIn("Editor target", description)
        self.assertIn("Win64 Development", description)
        self.assertIn("diagnostics", description)
        self.assertNotIn("target", properties)
        self.assertNotIn("platform", properties)
        self.assertNotIn("configuration", properties)

    def test_ue_doctor_schema_declares_perforce_status_scope(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        description = specs["ue_doctor"]["function"]["description"]

        self.assertIn(".uproject discovery", description)
        self.assertIn("EngineAssociation", description)
        self.assertIn("Perforce read-only status", description)
        self.assertIn("sole default check", description)
        self.assertIn("do not follow with shell p4 info", description)

    def test_p4_tool_schemas_declare_structured_perforce_workflow(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        for name in [
            "p4_status",
            "p4_file_state",
            "p4_opened",
            "p4_checkout",
            "p4_add",
            "p4_delete",
            "p4_reconcile",
            "p4_diff",
        ]:
            self.assertIn(name, specs)

        self.assertIn("read-only", specs["p4_status"]["function"]["description"])
        self.assertIn("lock/open conflict", specs["p4_file_state"]["function"]["description"])
        self.assertIn("Stops on UE binary asset lock conflicts", specs["p4_checkout"]["function"]["description"])
        self.assertIn("requires explicit approval", specs["p4_delete"]["function"]["description"])
        self.assertIn("UE binary assets are reported as skipped", specs["p4_diff"]["function"]["description"])

    def test_todo_update_schema_rejects_acknowledgement_usage(self) -> None:
        specs = {spec["function"]["name"]: spec for spec in get_tool_specs()}

        description = specs["todo_update"]["function"]["description"]

        self.assertIn("meaningful multi-step progress", description)
        self.assertIn("Do not use this tool to acknowledge instructions", description)
        self.assertIn("single status check", description)

class WorktreeTests(unittest.TestCase):
    def test_worktree_index_starts_empty(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            state_dir = agent_dir(root)
            tasks = TaskManager(state_dir / "tasks")
            manager = WorktreeManager(root, state_dir / "worktrees", tasks)

            self.assertIn("No managed", manager.list_all())

    def test_create_ue_git_linked_worktree_branches_links_content_and_copies_current_agent_session(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            project = root / "UEAgentDemo"
            (project / "Source").mkdir(parents=True)
            (project / "Config").mkdir()
            (project / "Content").mkdir()
            (project / "UEAgentDemo.uproject").write_text("{}", encoding="utf-8")
            (project / "Source" / "Game.cpp").write_text("// code", encoding="utf-8")
            (project / "Config" / "DefaultEngine.ini").write_text("[Engine]", encoding="utf-8")
            (project / "Content" / "Map.umap").write_text("asset", encoding="utf-8")

            state_dir = agent_dir(project)
            state_dir.mkdir(parents=True)
            (state_dir / "config.json").write_text('{"active_model": "main"}', encoding="utf-8")
            (state_dir / "worktrees").mkdir()
            (state_dir / "tasks").mkdir()
            (state_dir / "sessions" / "2000" / "01" / "01" / "session_old").mkdir(parents=True)
            history = HistoryRecorder(
                state_dir,
                [
                    ChatMessage(role="system", content="old system"),
                    ChatMessage(role="user", content=f"Working directory: {project}\nShell: PowerShell"),
                    ChatMessage(role="user", content="continue this session"),
                ],
            )
            session_dir = history.ensure_session()
            manager = WorktreeManager(project, state_dir / "worktrees", TaskManager(state_dir / "tasks"))
            default_root = root / ".uedev-worktrees"

            def fake_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
                if args == ["rev-parse", "--show-toplevel"]:
                    return subprocess.CompletedProcess(["git", *args], 0, str(project), "")
                if args[:2] == ["check-ref-format", "--branch"]:
                    return subprocess.CompletedProcess(["git", *args], 0, args[-1], "")
                if args[:3] == ["show-ref", "--verify", "--quiet"]:
                    return subprocess.CompletedProcess(["git", *args], 1, "", "")
                if args[:2] == ["status", "--porcelain"]:
                    return subprocess.CompletedProcess(["git", *args], 0, "", "")
                if args[:2] == ["ls-files", "--"]:
                    return subprocess.CompletedProcess(["git", *args], 0, "", "")
                if args[:4] == ["worktree", "add", "-b", "login-test"]:
                    target = Path(args[4])
                    target.mkdir(parents=True)
                    (target / "Source").mkdir()
                    (target / "Config").mkdir()
                    (target / "UEAgentDemo.uproject").write_text("{}", encoding="utf-8")
                    (target / "Source" / "Game.cpp").write_text("// code", encoding="utf-8")
                    (target / "Config" / "DefaultEngine.ini").write_text("[Engine]", encoding="utf-8")
                    return subprocess.CompletedProcess(["git", *args], 0, "", "")
                if args == ["rev-parse", "--git-path", "info/exclude"]:
                    return subprocess.CompletedProcess(["git", *args], 0, str(Path(cwd) / ".git" / "info" / "exclude"), "")
                raise AssertionError(f"unexpected git command: {args}")

            with (
                patch("uedev.tools.worktrees.os.name", "nt"),
                patch("uedev.tools.worktrees._run_git", side_effect=fake_git),
                patch("uedev.tools.worktrees._create_junction") as create_junction,
            ):
                output = manager.create_ue_git_linked("login-test", default_root=default_root, session_dir=session_dir)

            target = default_root / "UEAgentDemo" / "login-test"
            self.assertIn("Created UE linked worktree: login-test", output)
            self.assertIn("Branch: login-test", output)
            self.assertTrue((target / "UEAgentDemo.uproject").is_file())
            self.assertTrue((target / "Source" / "Game.cpp").is_file())
            self.assertTrue((target / "Config" / "DefaultEngine.ini").is_file())
            create_junction.assert_called_once_with(target.resolve() / "Content", (project / "Content").resolve())
            self.assertIn(".agent/", (target / ".git" / "info" / "exclude").read_text(encoding="utf-8"))
            self.assertIn("Content/", (target / ".git" / "info" / "exclude").read_text(encoding="utf-8"))

            target_agent = target / ".agent"
            target_session = target_agent / session_dir.relative_to(state_dir)
            self.assertTrue((target_agent / "config.json").is_file())
            self.assertTrue(target_session.is_dir())
            self.assertFalse((target_agent / "sessions" / "2000").exists())
            self.assertFalse((target_agent / "worktrees").exists())
            self.assertFalse((target_agent / "tasks").exists())
            loaded = load_history_file(target_session / "messages.jsonl")
            self.assertIn(str(target.resolve()), loaded[0].content)
            self.assertEqual(loaded[1].content, f"Working directory: {target.resolve()}\nShell: PowerShell")

            index = json.loads((state_dir / "worktrees" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["login-test"]["kind"], "ue-linked-worktree")
            self.assertEqual(index["login-test"]["mode"], "git-worktree-p4-content")
            self.assertEqual(index["login-test"]["branch"], "login-test")
            self.assertEqual(index["login-test"]["content_source"], str((project / "Content").resolve()))
            self.assertEqual(index["login-test"]["agent_session_source"], str(session_dir))
            self.assertEqual(index["login-test"]["agent_session_target"], str(target_session))
            self.assertIn("Content is shared", index["login-test"]["warnings"][0])
            listing = manager.list_all()
            self.assertIn("kind=ue-linked-worktree", listing)
            self.assertIn("mode=git-worktree-p4-content", listing)
            self.assertIn("branch=login-test", listing)

    def test_create_ue_linked_worktree_requires_content_and_windows(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            project = root / "UEAgentDemo"
            (project / "Source").mkdir(parents=True)
            (project / "Config").mkdir()
            (project / "UEAgentDemo.uproject").write_text("{}", encoding="utf-8")
            state_dir = agent_dir(project)
            manager = WorktreeManager(project, state_dir / "worktrees", TaskManager(state_dir / "tasks"))

            with patch("uedev.tools.worktrees.os.name", "nt"):
                with self.assertRaisesRegex(RuntimeError, "Content directory"):
                    manager.create_ue_git_linked("missing-content", default_root=root / ".uedev-worktrees")

            self.assertFalse((state_dir / "worktrees" / "index.json").exists())

            (project / "Content").mkdir()
            with patch("uedev.tools.worktrees.os.name", "posix"):
                with self.assertRaisesRegex(RuntimeError, "Windows junctions only"):
                    manager.create_ue_git_linked("not-windows", default_root=root / ".uedev-worktrees")

            with self.assertRaisesRegex(RuntimeError, "not implemented"):
                manager.create_ue_git_linked("p4-full", default_root=root / ".uedev-worktrees", mode="p4-full")

    def test_create_ue_git_linked_worktree_requires_clean_text_and_untracked_content(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            project = root / "UEAgentDemo"
            (project / "Source").mkdir(parents=True)
            (project / "Config").mkdir()
            (project / "Content").mkdir()
            (project / "UEAgentDemo.uproject").write_text("{}", encoding="utf-8")
            state_dir = agent_dir(project)
            manager = WorktreeManager(project, state_dir / "worktrees", TaskManager(state_dir / "tasks"))

            def run_case(status_stdout: str, ls_stdout: str, expected: str) -> None:
                def fake_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
                    if args == ["rev-parse", "--show-toplevel"]:
                        return subprocess.CompletedProcess(["git", *args], 0, str(project), "")
                    if args[:2] == ["check-ref-format", "--branch"]:
                        return subprocess.CompletedProcess(["git", *args], 0, args[-1], "")
                    if args[:3] == ["show-ref", "--verify", "--quiet"]:
                        return subprocess.CompletedProcess(["git", *args], 1, "", "")
                    if args[:2] == ["status", "--porcelain"]:
                        return subprocess.CompletedProcess(["git", *args], 0, status_stdout, "")
                    if args[:2] == ["ls-files", "--"]:
                        return subprocess.CompletedProcess(["git", *args], 0, ls_stdout, "")
                    raise AssertionError(f"unexpected git command: {args}")

                with patch("uedev.tools.worktrees._run_git", side_effect=fake_git):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        manager.create_ue_git_linked("login-test", default_root=root / ".uedev-worktrees")

            with patch("uedev.tools.worktrees.os.name", "nt"):
                run_case(" M Source/Game.cpp\n", "", "uncommitted changes")
                run_case("", "Content/Map.umap\n", "tracked by Git")

            self.assertFalse((state_dir / "worktrees" / "index.json").exists())

    def test_remove_ue_git_linked_worktree_removes_worktree_but_keeps_branch(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            source = root / "UEAgentDemo"
            target = root / ".uedev-worktrees" / "UEAgentDemo" / "login-test"
            target.mkdir(parents=True)
            (target / ".agent").mkdir()
            (target / ".agent" / "config.json").write_text("{}", encoding="utf-8")
            state_dir = agent_dir(source)
            manager = WorktreeManager(source, state_dir / "worktrees", TaskManager(state_dir / "tasks"))
            manager._save_index(
                {
                    "login-test": {
                        "kind": "ue-linked-worktree",
                        "mode": "git-worktree-p4-content",
                        "name": "login-test",
                        "branch": "login-test",
                        "path": str(target),
                        "source_repo_path": str(source),
                        "worktree_repo_path": str(target),
                        "worktree_project_path": str(target),
                        "content_link": str(target / "Content"),
                        "status": "active",
                    }
                }
            )

            def fake_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
                self.assertEqual(args, ["worktree", "remove", str(target)])
                return subprocess.CompletedProcess(["git", *args], 0, "", "")

            with (
                patch("uedev.tools.worktrees._run_git", side_effect=fake_git),
                patch("uedev.tools.worktrees._remove_junction") as remove_junction,
            ):
                output = manager.remove("login-test")

            remove_junction.assert_called_once_with(target / "Content", missing_ok=True)
            self.assertFalse((target / ".agent").exists())
            self.assertIn("Branch login-test was not deleted", output)
            self.assertEqual(manager.list_all(), "No managed worktrees.")


if __name__ == "__main__":
    unittest.main()
