from __future__ import annotations

import unittest
import uuid
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

from uedev.state.tasks import TodoManager
from uedev.ue import (
    UeRunResult,
    _discover_perforce,
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


class TodoManagerTests(unittest.TestCase):
    def test_update_rejects_multiple_in_progress_items(self) -> None:
        temp = workspace_temp_path()
        manager = TodoManager(temp / ".agent")

        with self.assertRaises(ValueError):
            manager.update(
                [
                    {"id": "1", "text": "A", "status": "in_progress"},
                    {"id": "2", "text": "B", "status": "in_progress"},
                ]
            )

    def test_update_persists_items(self) -> None:
        temp = workspace_temp_path()
        manager = TodoManager(temp / ".agent")
        manager.update([{"id": "1", "text": "Check UE config", "status": "completed"}])

        self.assertIn("Check UE config", manager.render_current())

    def test_empty_todo_file_loads_as_no_todos(self) -> None:
        temp = workspace_temp_path()
        manager = TodoManager(temp / ".agent")
        manager.path.write_text("", encoding="utf-8")

        self.assertEqual(manager.load(), [])
        self.assertFalse(manager.has_open_items())
        self.assertEqual(manager.render_current(), "No todos.")
