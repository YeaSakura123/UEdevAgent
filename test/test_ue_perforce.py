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
    p4_add,
    p4_checkout,
    p4_delete,
    p4_diff,
    p4_file_state,
    p4_opened,
    p4_reconcile,
    p4_status,
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


class UePerforceTests(unittest.TestCase):
    def test_discover_perforce_handles_missing_p4(self) -> None:
        root = workspace_temp_path()

        with patch("uedev.ue.subprocess.run", side_effect=FileNotFoundError):
            result = _discover_perforce(root)

        self.assertFalse(result.available)
        self.assertFalse(result.in_workspace)
        self.assertIn("p4 executable not found", result.notes[0])

    def test_discover_perforce_parses_workspace_project_and_opened_files(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")

        def fake_run(args, **kwargs):
            self.assertEqual(kwargs["cwd"], str(root))
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
            if command == ["fstat", str(project)]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="... depotFile //depot/Demo/Demo.uproject\n",
                    stderr="",
                )
            if command == ["opened"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            "//depot/Demo/Source/A.cpp#1 - edit default change (text)",
                            "//depot/Demo/Content/Map.umap#1 - edit default change (binary+l)",
                        ]
                    ),
                    stderr="",
                )
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.ue.subprocess.run", side_effect=fake_run):
            result = _discover_perforce(root, project)

        self.assertTrue(result.available)
        self.assertTrue(result.in_workspace)
        self.assertTrue(result.project_tracked)
        self.assertEqual(result.client_name, "demo-client")
        self.assertEqual(result.user_name, "alice")
        self.assertEqual(result.server_address, "perforce:1666")
        self.assertEqual(result.client_root, root.resolve())
        self.assertEqual(result.project_depot_path, "//depot/Demo/Demo.uproject")
        self.assertEqual(result.opened_count, 2)
        self.assertEqual(len(result.opened_preview), 2)

    def test_discover_preforce_keeps_legacy_typo_alias(self) -> None:
        root = workspace_temp_path()

        with patch("uedev.ue.subprocess.run", side_effect=FileNotFoundError):
            result = _discover_preforce(root)

        self.assertFalse(result.available)

    def test_p4_status_renders_workspace_summary(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")

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

        with patch("uedev.ue.perforce.subprocess.run", side_effect=fake_run):
            payload = json.loads(p4_status(root))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["client_name"], "demo-client")
        self.assertTrue(payload["project_tracked"])
        self.assertEqual(payload["opened_count"], 0)

    def test_p4_file_state_reports_tracked_and_untracked_files(self) -> None:
        root = workspace_temp_path()
        tracked = root / "Source" / "A.cpp"
        tracked.parent.mkdir()
        tracked.write_text("int x;\n", encoding="utf-8")
        missing = root / "Source" / "Missing.cpp"

        def fake_run(args, **kwargs):
            command = args[1:]
            if command == ["fstat", str(tracked.resolve())]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            "... depotFile //depot/Demo/Source/A.cpp",
                            f"... clientFile {tracked.resolve()}",
                            "... headRev 3",
                            "... haveRev 3",
                            "... action edit",
                            "... type text",
                        ]
                    ),
                    stderr="",
                )
            if command == ["fstat", str(missing.resolve())]:
                return subprocess.CompletedProcess(args, 1, stdout=f"{missing} - no such file(s).\n", stderr="")
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.ue.perforce.subprocess.run", side_effect=fake_run):
            payload = json.loads(p4_file_state(root, ["Source/A.cpp", "Source/Missing.cpp"]))

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["files"][0]["tracked"])
        self.assertTrue(payload["files"][0]["opened"])
        self.assertFalse(payload["files"][1]["tracked"])

    def test_p4_checkout_stops_on_binary_asset_conflict(self) -> None:
        root = workspace_temp_path()
        asset = root / "Content" / "Map.umap"
        asset.parent.mkdir()
        asset.write_text("", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args[1:])
            command = args[1:]
            if command == ["fstat", str(asset.resolve())]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            "... depotFile //depot/Demo/Content/Map.umap",
                            f"... clientFile {asset.resolve()}",
                            "... headRev 9",
                            "... type binary+l",
                            "... otherOpen 1",
                            "... otherOpen0 bob@other-client",
                        ]
                    ),
                    stderr="",
                )
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.ue.perforce.subprocess.run", side_effect=fake_run):
            payload = json.loads(p4_checkout(root, ["Content/Map.umap"]))

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "conflict")
        self.assertEqual(calls, [["fstat", str(asset.resolve())]])

    def test_p4_checkout_add_delete_reconcile_and_opened_run_expected_commands(self) -> None:
        root = workspace_temp_path()
        source = root / "Source" / "A.cpp"
        source.parent.mkdir()
        source.write_text("int x;\n", encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run(args, **kwargs):
            command = args[1:]
            commands.append(command)
            if command == ["fstat", str(source.resolve())]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            "... depotFile //depot/Demo/Source/A.cpp",
                            f"... clientFile {source.resolve()}",
                            "... headRev 1",
                            "... type text",
                        ]
                    ),
                    stderr="",
                )
            if command[0] in {"edit", "add", "delete", "reconcile"}:
                return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")
            if command == ["opened"]:
                return subprocess.CompletedProcess(args, 0, stdout="//depot/Demo/Source/A.cpp#1 - edit default change (text)\n", stderr="")
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.ue.perforce.subprocess.run", side_effect=fake_run):
            self.assertTrue(json.loads(p4_checkout(root, ["Source/A.cpp"]))["ok"])
            self.assertTrue(json.loads(p4_add(root, ["Source/A.cpp"]))["ok"])
            self.assertTrue(json.loads(p4_delete(root, ["Source/A.cpp"]))["ok"])
            self.assertTrue(json.loads(p4_reconcile(root, ["Source/A.cpp"]))["ok"])
            self.assertEqual(json.loads(p4_opened(root))["opened_count"], 1)

        self.assertIn(["edit", str(source.resolve())], commands)
        self.assertIn(["add", str(source.resolve())], commands)
        self.assertIn(["delete", str(source.resolve())], commands)
        self.assertIn(["reconcile", str(source.resolve())], commands)

    def test_p4_diff_skips_binary_assets(self) -> None:
        root = workspace_temp_path()
        source = root / "Source" / "A.cpp"
        asset = root / "Content" / "Map.umap"
        source.parent.mkdir()
        asset.parent.mkdir()
        source.write_text("int x;\n", encoding="utf-8")
        asset.write_text("", encoding="utf-8")

        def fake_run(args, **kwargs):
            command = args[1:]
            if command == ["fstat", str(source.resolve())]:
                return subprocess.CompletedProcess(args, 0, stdout="... depotFile //depot/Demo/Source/A.cpp\n... type text\n", stderr="")
            if command == ["fstat", str(asset.resolve())]:
                return subprocess.CompletedProcess(args, 0, stdout="... depotFile //depot/Demo/Content/Map.umap\n... type binary+l\n", stderr="")
            if command == ["diff", str(source.resolve())]:
                return subprocess.CompletedProcess(args, 0, stdout="==== //depot/Demo/Source/A.cpp#1 - text ====\n", stderr="")
            raise AssertionError(f"unexpected p4 command: {command}")

        with patch("uedev.ue.perforce.subprocess.run", side_effect=fake_run):
            payload = json.loads(p4_diff(root, ["Source/A.cpp", "Content/Map.umap"]))

        self.assertTrue(payload["ok"])
        self.assertIn("Source/A.cpp", payload["diff"])
        self.assertEqual(payload["skipped_binary"][0]["path"], str(asset.resolve()))

