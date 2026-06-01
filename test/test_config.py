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
from uedev.state.config import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_DIFF_OUTPUT_MAX_CHARS,
    DEFAULT_MAX_STEPS,
    DEFAULT_WORKTREE_ROOT,
    DEFAULT_WORKSPACE_EXCLUDED_DIRS,
    ConfigError,
    agent_dir,
    format_model_profiles,
    load_project_config,
    load_system_config,
    resolve_model_profile,
    resolve_subagent_model_profile,
    system_config_template,
)
from uedev.cli import _resolve_max_steps
from uedev.runtime.context import compact_locally, estimate_tokens, micro_compact, repair_tool_call_messages
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
from uedev.tools.workspace import edit_file, read_file, write_file
from uedev.tools.worktrees import WorktreeManager


def write_system_config(
    config_path: Path,
    *,
    models: dict[str, dict[str, object]] | None = None,
    ue_engines: dict[str, dict[str, object]] | None = None,
    display: dict[str, object] | None = None,
    runtime: dict[str, object] | None = None,
    workspace: dict[str, object] | None = None,
    worktrees: dict[str, object] | None = None,
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
    if runtime is not None:
        payload["runtime"] = runtime
    if workspace is not None:
        payload["workspace"] = workspace
    if worktrees is not None:
        payload["worktrees"] = worktrees
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


class ConfigTests(unittest.TestCase):
    def test_missing_system_config_raises(self) -> None:
        with workspace_temp_dir() as temp:
            with self.assertRaisesRegex(ConfigError, "Config file not found"):
                load_system_config(Path(temp) / "missing.json")

    def test_invalid_system_config_raises(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "bad.json"
            config_path.write_text("{bad", encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "Invalid JSON"):
                load_system_config(config_path)

    def test_project_active_model_overrides_default(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "main": {
                        "model": "main-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "main-key",
                    },
                    "gpt-alt": {
                        "model": "alt-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "alt-key",
                    },
                },
            )
            (agent_dir(root) / "config.json").parent.mkdir(parents=True)
            (agent_dir(root) / "config.json").write_text(json.dumps({"version": 1, "active_model": "gpt-alt"}), encoding="utf-8")

            profile = resolve_model_profile(root, load_system_config(config_path))

            self.assertEqual(profile.name, "gpt-alt")
            self.assertEqual(profile.model, "alt-model")

    def test_first_model_is_default_when_default_model_is_omitted(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "first": {
                        "model": "first-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "first-key",
                    },
                    "second": {
                        "model": "second-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "second-key",
                    },
                },
            )

            config = load_system_config(config_path)
            profile = resolve_model_profile(root, config)

            self.assertEqual(config.default_model, "first")
            self.assertEqual(profile.name, "first")

    def test_model_context_window_defaults_to_256k_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "default-window": {
                        "model": "default-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    },
                    "small-window": {
                        "model": "small-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 4096,
                    },
                },
            )

            config = load_system_config(config_path)

            self.assertEqual(config.models["default-window"].context_window, DEFAULT_CONTEXT_WINDOW)
            self.assertEqual(config.models["small-window"].context_window, 4096)
            self.assertIn("context_window=4096", format_model_profiles(root, config))

    def test_model_reasoning_content_replay_defaults_false_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "plain": {
                        "model": "plain-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                    },
                    "deepseek": {
                        "model": "deepseek-reasoner",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key": "key",
                        "requires_reasoning_content": True,
                    },
                },
            )

            config = load_system_config(config_path)

            self.assertFalse(config.models["plain"].requires_reasoning_content)
            self.assertTrue(config.models["deepseek"].requires_reasoning_content)
            self.assertIn("requires_reasoning_content=True", format_model_profiles(root, config))

    def test_invalid_model_reasoning_content_replay_raises(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "bad": {
                        "model": "bad-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "requires_reasoning_content": "yes",
                    },
                },
            )

            with self.assertRaisesRegex(ConfigError, "models.bad.requires_reasoning_content"):
                load_system_config(config_path)

    def test_invalid_model_context_window_raises(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(
                config_path,
                models={
                    "bad": {
                        "model": "bad-model",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "key",
                        "context_window": 0,
                    },
                },
            )

            with self.assertRaisesRegex(ConfigError, "models.bad.context_window"):
                load_system_config(config_path)

    def test_diff_output_max_chars_defaults_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path)

            config = load_system_config(config_path)

            self.assertEqual(config.diff_output_max_chars, DEFAULT_DIFF_OUTPUT_MAX_CHARS)

            write_system_config(config_path, display={"diff_output_max_chars": 1234})

            self.assertEqual(load_system_config(config_path).diff_output_max_chars, 1234)

    def test_invalid_diff_output_max_chars_raises(self) -> None:
        for value in (0, -1, "many"):
            with self.subTest(value=value):
                with workspace_temp_dir() as temp:
                    config_path = Path(temp) / "system-config.json"
                    write_system_config(config_path, display={"diff_output_max_chars": value})

                    with self.assertRaisesRegex(ConfigError, "display.diff_output_max_chars"):
                        load_system_config(config_path)

    def test_system_config_template_includes_diff_output_limit(self) -> None:
        template = system_config_template()

        self.assertEqual(template["display"]["diff_output_max_chars"], DEFAULT_DIFF_OUTPUT_MAX_CHARS)
        self.assertFalse(template["models"]["my-model"]["requires_reasoning_content"])

    def test_runtime_default_max_steps_defaults_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path)

            self.assertEqual(load_system_config(config_path).runtime_default_max_steps, DEFAULT_MAX_STEPS)

            write_system_config(config_path, runtime={"default_max_steps": 24})

            self.assertEqual(load_system_config(config_path).runtime_default_max_steps, 24)

    def test_invalid_runtime_default_max_steps_raises(self) -> None:
        for value in (0, -1, "many", True):
            with self.subTest(value=value):
                with workspace_temp_dir() as temp:
                    config_path = Path(temp) / "system-config.json"
                    write_system_config(config_path, runtime={"default_max_steps": value})

                    with self.assertRaisesRegex(ConfigError, "runtime.default_max_steps"):
                        load_system_config(config_path)

    def test_system_config_template_includes_runtime_default_max_steps(self) -> None:
        template = system_config_template()

        self.assertEqual(template["runtime"]["default_max_steps"], DEFAULT_MAX_STEPS)

    def test_cli_max_steps_uses_system_config_when_omitted(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path, runtime={"default_max_steps": 13})

            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                self.assertEqual(_resolve_max_steps(None), 13)
            self.assertEqual(_resolve_max_steps(5), 5)

    def test_workspace_excluded_dirs_default_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path)

            self.assertEqual(load_system_config(config_path).workspace_excluded_dirs, DEFAULT_WORKSPACE_EXCLUDED_DIRS)

            write_system_config(config_path, workspace={"excluded_dirs": [".cache", "Generated"]})

            self.assertEqual(load_system_config(config_path).workspace_excluded_dirs, (".cache", "Generated"))

    def test_invalid_workspace_excluded_dirs_raise(self) -> None:
        values = (
            "Generated",
            [""],
            [123],
            ["Nested/Dir"],
            ["Nested\\Dir"],
            ["/absolute"],
            ["C:/absolute"],
            ["C:"],
            [".."],
        )
        for value in values:
            with self.subTest(value=value):
                with workspace_temp_dir() as temp:
                    config_path = Path(temp) / "system-config.json"
                    write_system_config(config_path, workspace={"excluded_dirs": value})

                    with self.assertRaisesRegex(ConfigError, "workspace.excluded_dirs"):
                        load_system_config(config_path)

    def test_system_config_template_includes_workspace_excluded_dirs(self) -> None:
        template = system_config_template()

        self.assertEqual(tuple(template["workspace"]["excluded_dirs"]), DEFAULT_WORKSPACE_EXCLUDED_DIRS)

    def test_worktree_default_root_defaults_and_can_be_configured(self) -> None:
        with workspace_temp_dir() as temp:
            root = Path(temp)
            config_path = root / "system-config.json"
            write_system_config(config_path)

            self.assertIsNone(load_system_config(config_path).worktree_default_root)

            configured = root / "worktrees"
            write_system_config(config_path, worktrees={"default_root": str(configured)})

            self.assertEqual(load_system_config(config_path).worktree_default_root, configured.resolve())

    def test_invalid_worktree_default_root_raises(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path, worktrees={"default_root": 123})

            with self.assertRaisesRegex(ConfigError, "worktrees.default_root"):
                load_system_config(config_path)

    def test_system_config_template_includes_worktree_default_root(self) -> None:
        template = system_config_template()

        self.assertEqual(template["worktrees"]["default_root"], DEFAULT_WORKTREE_ROOT)

    def test_subagent_model_profile_defaults_to_main_profile(self) -> None:
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
            )
            config = load_system_config(config_path)

            profile = resolve_subagent_model_profile(root, config.models["child"], config)

            self.assertEqual(profile.name, "child")
            self.assertIsNone(config.subagent_model_profile)

            write_system_config(config_path, subagents={"model_profile": ""})
            config = load_system_config(config_path)

            self.assertIsNone(config.subagent_model_profile)

    def test_subagent_model_profile_can_reference_existing_profile(self) -> None:
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
            config = load_system_config(config_path)

            profile = resolve_subagent_model_profile(root, config.models["main"], config)

            self.assertEqual(config.subagent_model_profile, "child")
            self.assertEqual(profile.name, "child")
            self.assertEqual(profile.model, "child-model")

    def test_invalid_subagent_model_profile_raises(self) -> None:
        with workspace_temp_dir() as temp:
            config_path = Path(temp) / "system-config.json"
            write_system_config(config_path, subagents={"model_profile": "missing"})

            with self.assertRaisesRegex(ConfigError, "subagents.model_profile"):
                load_system_config(config_path)

    def test_system_config_template_includes_subagent_profile(self) -> None:
        template = system_config_template()

        self.assertIsNone(template["subagents"]["model_profile"])

    def test_project_permission_mode_defaults_and_persists(self) -> None:
        from uedev.state.config import save_project_permission_mode

        with workspace_temp_dir() as temp:
            root = Path(temp)

            self.assertEqual(load_project_config(root).permission_mode, "default")

            save_project_permission_mode(root, "auto_review")
            payload = json.loads((agent_dir(root) / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["permission_mode"], "auto_review")
            self.assertEqual(load_project_config(root).permission_mode, "auto_review")
