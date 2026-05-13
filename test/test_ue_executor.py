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


class UeExecutorTests(unittest.TestCase):
    def test_editor_executor_moves_pending_task_to_done(self) -> None:
        root = workspace_temp_path()
        agent_dir = root / ".agent"
        pending = agent_dir / "ue_queue" / "pending"
        pending.mkdir(parents=True)
        for name in ["running", "done", "failed"]:
            (agent_dir / "ue_queue" / name).mkdir(parents=True)
        marker = root / "marker.txt"
        wrapper = root / "wrapper.py"
        wrapper.write_text(
            f"from pathlib import Path\nPath({json.dumps(str(marker))}).write_text('done', encoding='utf-8')\n",
            encoding="utf-8",
        )
        task = pending / "ue_test.task.json"
        task.write_text(json.dumps({"run_id": "ue_test", "wrapper_path": str(wrapper)}), encoding="utf-8")

        code = build_editor_executor_script(agent_dir)
        exec(compile(code, str(agent_dir / "ue_executor" / "editor_executor.py"), "exec"), {})

        self.assertEqual(marker.read_text(encoding="utf-8"), "done")
        self.assertFalse(task.exists())
        self.assertTrue((agent_dir / "ue_queue" / "done" / "ue_test.task.json").exists())
        self.assertTrue((agent_dir / "ue_executor" / "heartbeat.json").exists())

    def test_enqueue_editor_stop_writes_pending_stop_task(self) -> None:
        root = workspace_temp_path()
        stop_path = enqueue_editor_stop(root / ".agent")

        self.assertTrue(stop_path.exists())
        self.assertTrue(stop_path.name.endswith(".stop.json"))
        self.assertIn("pending", str(stop_path))
        payload = json.loads(stop_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["type"], "stop")

    def test_full_editor_script_does_not_enable_per_run_keep_alive(self) -> None:
        wrapped = build_python_script("custom", "print('ok')", keep_editor_open=True)

        self.assertNotIn("set_keep_python_script_alive(True)", wrapped)
        self.assertIn("print('ok')", wrapped)

