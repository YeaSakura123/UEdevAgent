from __future__ import annotations

import unittest
import uuid
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

from uedev.tasks import TodoManager
from uedev.ue import (
    UeRunResult,
    _discover_perforce,
    _discover_preforce,
    build_editor_executor_script,
    build_python_script,
    build_wrapper_script,
    discover_ue,
    enqueue_editor_stop,
    execute_prepared_ue_python,
    generate_run_id,
    prepare_ue_python,
    render_doctor,
    render_run_result,
    run_ue_python,
)


def workspace_temp_path() -> Path:
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


def write_system_config(config_path: Path, engine_name: str, engine_root: Path, aliases: list[str] | None = None) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "first-model": {
                        "model": "gpt-test",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "test-key",
                    }
                },
                "ue": {
                    "engines": {
                        engine_name: {
                            "root": str(engine_root),
                            "aliases": aliases or [],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def create_engine_root(root: Path, *, commandlet: bool = True, gui: bool = True) -> tuple[Path, Path]:
    bin_dir = root / "Engine" / "Binaries" / "Win64"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cmd = bin_dir / "UnrealEditor-Cmd.exe"
    editor = bin_dir / "UnrealEditor.exe"
    if commandlet:
        cmd.write_text("", encoding="utf-8")
    if gui:
        editor.write_text("", encoding="utf-8")
    return cmd, editor


class UeDiscoveryTests(unittest.TestCase):
    def test_discover_project_and_engine_from_config(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        cmd, editor = create_engine_root(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            result = discover_ue(root)

        self.assertEqual(result.project_path, project.resolve())
        self.assertEqual(result.engine_association, "5.4")
        self.assertEqual(result.engine_name, "5.4")
        self.assertEqual(result.editor_cmd_path, cmd.resolve())
        self.assertEqual(result.editor_gui_path, editor.resolve())

    def test_discover_engine_uses_alias(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.5"}), encoding="utf-8")
        engine_root = root / "UE_5.5_Source"
        create_engine_root(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.5-source", engine_root, aliases=["5.5"])

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            result = discover_ue(root)

        self.assertEqual(result.engine_association, "5.5")
        self.assertEqual(result.engine_name, "5.5-source")

    def test_unconfigured_engine_association_does_not_fallback(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.6"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            result = discover_ue(root)

        self.assertEqual(result.engine_association, "5.6")
        self.assertIsNone(result.engine_name)
        self.assertIsNone(result.editor_cmd_path)
        self.assertTrue(any("5.6" in note and "Available UE engines: 5.4" in note for note in result.notes))

    def test_render_doctor_starts_with_compact_summary(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        def fake_run(args, **kwargs):
            command = args[1:]
            if command == ["info"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            "User name: alice",
                            "Client name: demo-client",
                            f"Client root: {root}",
                            "Server address: perforce:1666",
                        ]
                    ),
                    stderr="",
                )
            if command == ["fstat", str(project.resolve())]:
                return subprocess.CompletedProcess(args, 0, stdout="... depotFile //depot/Demo/Demo.uproject\n", stderr="")
            if command == ["opened"]:
                return subprocess.CompletedProcess(args, 1, stdout="File(s) not opened on this client.\n", stderr="")
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            with patch("uedev.ue.subprocess.run", side_effect=fake_run):
                rendered = render_doctor(discover_ue(root))

        lines = rendered.splitlines()
        self.assertEqual(lines[0], "UE doctor")
        self.assertEqual(lines[1], "- summary: project=yes, engine=5.4, perforce=workspace/tracked")
        self.assertIn("- project:", lines[2])

