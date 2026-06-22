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
from uedev.runtime.context import estimate_tokens, micro_compact, repair_tool_call_messages
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
from uedev.policy.permissions import classify_shell_command, classify_tool_permission
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
from uedev.tools.specs import get_tool_names, get_tool_specs
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
        with workspace_temp_dir() as temp:
            with patch("uedev.tools.shell.subprocess.Popen", return_value=FakeProcess()):
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

        with workspace_temp_dir() as temp:
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

            with patch("uedev.runtime.agent.run_shell", return_value=ShellResult("curl https://example.com", 0, "ok\n", "")):
                result = runtime.tools["shell"]({"command": "curl https://example.com", "reason": "test approval"})

        self.assertEqual(approvals, [("curl https://example.com", "test approval")])
        self.assertIn("exitCode: 0", result)
        self.assertIn("ok", result)

    def test_runtime_rejects_shell_when_approval_provider_declines(self) -> None:
        with workspace_temp_dir() as temp:
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

            result = runtime.tools["shell"]({"command": "curl https://example.com", "reason": "test rejection"})

        self.assertIn("rejected", result)

    def test_default_permission_allows_local_shell_without_approval(self) -> None:
        approvals: list[tuple[str, str]] = []

        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                ),
                approval_provider=lambda command, reason: approvals.append((command, reason)) or False,
            )

            with patch("uedev.runtime.agent.run_shell", return_value=ShellResult("echo ok", 0, "ok\n", "")):
                result = runtime.tools["shell"]({"command": "echo ok", "reason": "local"})

        self.assertEqual(approvals, [])
        self.assertIn("exitCode: 0", result)

    def test_read_only_permission_requires_approval_for_write_file(self) -> None:
        approvals: list[tuple[str, str]] = []

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
                ),
                approval_provider=lambda command, reason: approvals.append((command, reason)) or False,
            )
            runtime.permission_mode = "read_only"

            result = runtime.tools["write_file"]({"path": "blocked.txt", "content": "no"})

            self.assertIn("rejected", result)
            self.assertEqual(approvals, [("write_file blocked.txt", "read-only mode requires approval before editing files")])
            self.assertFalse((root / "blocked.txt").exists())

    def test_auto_review_permission_denies_dangerous_shell(self) -> None:
        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                )
            )
            runtime.permission_mode = "auto_review"

            result = runtime.tools["shell"]({"command": "git reset --hard", "reason": "danger"})

        self.assertIn("Tool denied by policy", result)

    def test_full_access_permission_skips_network_approval(self) -> None:
        approvals: list[tuple[str, str]] = []

        with workspace_temp_dir() as temp:
            runtime = AgentRuntime(
                AgentOptions(
                    task="",
                    max_steps=1,
                    auto_approve=False,
                    cwd=Path(temp),
                    timeout_seconds=120,
                    verbose=False,
                ),
                approval_provider=lambda command, reason: approvals.append((command, reason)) or False,
            )
            runtime.permission_mode = "full_access"

            with patch("uedev.runtime.agent.run_shell", return_value=ShellResult("curl https://example.com", 0, "ok\n", "")):
                result = runtime.tools["shell"]({"command": "curl https://example.com", "reason": "network"})

        self.assertEqual(approvals, [])
        self.assertIn("exitCode: 0", result)

    def test_plan_mode_denies_write_file(self) -> None:
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
                )
            )
            runtime.collaboration_mode = "plan"

            result = runtime.tools["write_file"]({"path": "blocked.txt", "content": "no"})

            self.assertIn("Tool denied by policy", result)
            self.assertFalse((root / "blocked.txt").exists())

    def test_shell_command_classifier_marks_common_risks(self) -> None:
        self.assertEqual(classify_shell_command("rg permission uedev"), "readonly")
        self.assertEqual(classify_shell_command("git reset --hard"), "dangerous")
        self.assertEqual(classify_shell_command("curl https://example.com"), "network")
        self.assertEqual(classify_shell_command("p4 diff Source/A.cpp"), "readonly")
        self.assertEqual(classify_shell_command("p4 delete Source/A.cpp"), "dangerous")

    def test_p4_tool_permissions_match_local_pending_workflow(self) -> None:
        grep_read_only = classify_tool_permission(
            "grep",
            {"pattern": "needle"},
            collaboration_mode="default",
            permission_mode="read_only",
        )
        grep_default = classify_tool_permission(
            "grep",
            {"pattern": "needle"},
            collaboration_mode="default",
            permission_mode="default",
        )
        grep_plan = classify_tool_permission(
            "grep",
            {"pattern": "needle"},
            collaboration_mode="plan",
            permission_mode="full_access",
        )
        read_decision = classify_tool_permission(
            "p4_file_state",
            {"paths": ["Source/A.cpp"]},
            collaboration_mode="default",
            permission_mode="read_only",
        )
        checkout_decision = classify_tool_permission(
            "p4_checkout",
            {"paths": ["Source/A.cpp"]},
            collaboration_mode="default",
            permission_mode="read_only",
        )
        delete_decision = classify_tool_permission(
            "p4_delete",
            {"paths": ["Source/A.cpp"]},
            collaboration_mode="default",
            permission_mode="full_access",
        )
        plan_decision = classify_tool_permission(
            "p4_checkout",
            {"paths": ["Source/A.cpp"]},
            collaboration_mode="plan",
            permission_mode="full_access",
        )

        self.assertEqual(grep_read_only.action, "allow")
        self.assertEqual(grep_default.action, "allow")
        self.assertEqual(grep_plan.action, "allow")
        self.assertEqual(read_decision.action, "allow")
        self.assertEqual(checkout_decision.action, "ask")
        self.assertEqual(delete_decision.action, "ask")
        self.assertEqual(plan_decision.action, "deny")

    def test_ue_build_permission_matches_local_execution(self) -> None:
        read_only_decision = classify_tool_permission(
            "ue_build",
            {},
            collaboration_mode="default",
            permission_mode="read_only",
        )
        default_decision = classify_tool_permission(
            "ue_build",
            {},
            collaboration_mode="default",
            permission_mode="default",
        )
        plan_decision = classify_tool_permission(
            "ue_build",
            {},
            collaboration_mode="plan",
            permission_mode="full_access",
        )

        self.assertEqual(read_only_decision.action, "ask")
        self.assertEqual(default_decision.action, "allow")
        self.assertEqual(plan_decision.action, "deny")

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
