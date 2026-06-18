from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from uedev.state.plans import PlanManager, default_plan_dir, extract_plan_title


def workspace_temp_path() -> Path:
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    return path


class PlanManagerTests(unittest.TestCase):
    def test_default_plan_dir_uses_system_config_parent(self) -> None:
        root = workspace_temp_path()
        config_path = root / ".uedev" / "config.json"

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            self.assertEqual(default_plan_dir(), config_path.parent.resolve() / "plan")

    def test_save_proposed_plan_writes_markdown_record(self) -> None:
        root = workspace_temp_path()
        config_path = root / ".uedev" / "config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"version": 1}), encoding="utf-8")

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            manager = PlanManager()
            record = manager.save_proposed_plan("session_123", "turn-1", "# Launch plan\n\n- Build it")

        path = Path(record.path)
        self.assertEqual(path.parent, config_path.parent.resolve() / "plan")
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(encoding="utf-8"), "# Launch plan\n\n- Build it\n")
        self.assertEqual(record.title, "Launch plan")
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.session_id, "session_123")
        self.assertEqual(record.turn_id, "turn-1")

    def test_extract_plan_title_falls_back_to_first_non_empty_line(self) -> None:
        self.assertEqual(extract_plan_title("\nImplement the API\nMore"), "Implement the API")
