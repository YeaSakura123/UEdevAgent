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


class UeTests(unittest.TestCase):
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

    def test_discover_preforce_keeps_legacy_typo_alias(self) -> None:
        root = workspace_temp_path()

        with patch("uedev.ue.subprocess.run", side_effect=FileNotFoundError):
            result = _discover_preforce(root)

        self.assertFalse(result.available)

    def test_generate_run_id_is_sortable_and_file_safe(self) -> None:
        first = generate_run_id()
        second = generate_run_id()

        self.assertTrue(first.startswith("ue_"))
        self.assertLess(first, second)
        self.assertNotIn(":", first)
        self.assertNotIn("\\", first)

    def test_build_script_keeps_user_body(self) -> None:
        body = build_python_script("custom", "print('ok')")

        self.assertIn("print('ok')", body)

    def test_wrapper_writes_structured_results_and_errors(self) -> None:
        root = workspace_temp_path()
        user_script = root / "user_script.py"
        result = root / "result.json"
        heartbeat = root / "heartbeat.json"
        events = root / "events.jsonl"

        wrapped = build_wrapper_script(
            run_id="ue_test",
            project_dir=root,
            user_script_path=user_script,
            result_path=result,
            heartbeat_path=heartbeat,
            events_path=events,
            stdout_path=root / "stdout.log",
            stderr_path=root / "stderr.log",
        )

        self.assertIn("traceback.format_exc", wrapped)
        self.assertIn("_uedev_result", wrapped)
        self.assertIn("_uedev_emit", wrapped)
        self.assertIn("_patched_run_path", wrapped)
        self.assertIn("os.replace", wrapped)
        self.assertIn("os.chdir(PROJECT_DIR)", wrapped)
        self.assertIn("_uedev_project_dir", wrapped)
        self.assertIn(json.dumps(str(result)), wrapped)

    def test_wrapper_captures_unreal_logs_emit_and_child_results(self) -> None:
        root = workspace_temp_path()
        user_script = root / "user_script.py"
        child_script = root / "child.py"
        result = root / "result.json"
        heartbeat = root / "heartbeat.json"
        events = root / "events.jsonl"
        stdout = root / "stdout.log"
        stderr = root / "stderr.log"
        child_script.write_text("_uedev_result = {'child': 'ok'}\n", encoding="utf-8")
        user_script.write_text(
            "\n".join(
                [
                    "import runpy",
                    "import unreal",
                    "print('hello stdout')",
                    "unreal.log('HELLOUE: level name = TestMap')",
                    "_uedev_emit('level_name', 'TestMap')",
                    "runpy.run_path('child.py', run_name='__main__')",
                    "_uedev_result = {'main': True}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        fake_unreal = types.SimpleNamespace(
            log=lambda message: None,
            log_warning=lambda message: None,
            log_error=lambda message: None,
        )
        old_unreal = sys.modules.get("unreal")
        sys.modules["unreal"] = fake_unreal
        try:
            wrapped = build_wrapper_script(
                run_id="ue_test",
                project_dir=root,
                user_script_path=user_script,
                result_path=result,
                heartbeat_path=heartbeat,
                events_path=events,
                stdout_path=stdout,
                stderr_path=stderr,
            )
            exec(compile(wrapped, str(root / "wrapper.py"), "exec"), {})
        finally:
            if old_unreal is None:
                sys.modules.pop("unreal", None)
            else:
                sys.modules["unreal"] = old_unreal

        payload = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"], {"main": True})
        self.assertEqual(payload["emitted"], {"level_name": "TestMap"})
        self.assertEqual(payload["child_results"][0]["result"], {"child": "ok"})
        self.assertEqual(payload["logs"][0]["message"], "begin")
        self.assertIn("HELLOUE: level name = TestMap", [log["message"] for log in payload["logs"]])
        self.assertEqual(stdout.read_text(encoding="utf-8"), "hello stdout\n")
        self.assertEqual(stderr.read_text(encoding="utf-8"), "")

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

    def test_prepare_full_editor_script_uses_gui_without_per_run_keep_alive(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root, commandlet=False, gui=True)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            prepared = prepare_ue_python(root, root / ".agent", "print('ok')", mode="full_editor")

        meta = json.loads((prepared.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(prepared.mode, "full_editor")
        self.assertIn("UnrealEditor.exe", prepared.command)
        self.assertIn("-ExecutePythonScript=", prepared.command)
        self.assertNotIn("set_keep_python_script_alive(True)", prepared.user_script_path.read_text(encoding="utf-8"))
        self.assertEqual(prepared.user_script_path.read_text(encoding="utf-8"), "print('ok')")
        self.assertEqual(meta["script_origin"], "inline")
        self.assertEqual(meta["engine_association"], "5.4")
        self.assertEqual(meta["engine_name"], "5.4")
        self.assertIsNone(meta["source_script"])
        self.assertTrue(prepared.run_dir.exists())
        self.assertTrue((prepared.run_dir / "meta.json").exists())
        self.assertTrue(prepared.wrapper_path.exists())
        self.assertIsNotNone(prepared.task_path)
        self.assertIn("ue_queue", str(prepared.task_path))

    def test_prepare_snapshots_source_script_content_and_meta(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        source = root / "hello_source.py"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root, commandlet=True, gui=False)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)
        source.write_text("_uedev_result = {'from': 'source'}\n", encoding="utf-8")

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            prepared = prepare_ue_python(
                root,
                root / ".agent",
                source.read_text(encoding="utf-8"),
                mode="commandlet",
                source_script_path=source,
            )

        meta = json.loads((prepared.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(prepared.user_script_path.read_text(encoding="utf-8"), "_uedev_result = {'from': 'source'}\n")
        self.assertEqual(prepared.script_origin, "script_path")
        self.assertEqual(meta["script_origin"], "script_path")
        self.assertEqual(meta["source_script"]["path"], str(source.resolve()))
        self.assertEqual(meta["source_script"]["available"], True)
        self.assertIn("sha256", meta["source_script"])

    def test_render_running_full_editor_result(self) -> None:
        result = UeRunResult(
            command="UnrealEditor.exe Demo.uproject",
            script_path=Path("wrapper.py"),
            user_script_path=Path("user_script.py"),
            wrapper_path=Path("wrapper.py"),
            executed=True,
            process_id=123,
            status="running",
            run_id="ue_test",
        )
        rendered = render_run_result(result)

        self.assertIn("run_id: ue_test", rendered)
        self.assertIn("status: running", rendered)
        self.assertIn("user_script_path: user_script.py", rendered)
        self.assertIn("wrapper_path: wrapper.py", rendered)
        self.assertNotIn("script: wrapper.py", rendered)
        self.assertIn("process_id: 123", rendered)

    def test_run_python_dry_run_requires_editor_path(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root, commandlet=True, gui=False)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            result = run_ue_python(root, root / ".agent", "print('ok')", execute=False)

        meta = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertFalse(result.executed)
        self.assertIn("UnrealEditor-Cmd.exe", result.command)
        self.assertEqual(result.script_origin, "inline")
        self.assertEqual(meta["script_origin"], "inline")
        self.assertIsNone(meta["source_script"])
        self.assertTrue(result.run_dir.exists())
        self.assertTrue((result.run_dir / "user_script.py").exists())
        self.assertTrue((result.run_dir / "wrapper.py").exists())

    def test_commandlet_execution_saves_stdout_stderr_and_result(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root, commandlet=True, gui=False)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.config.default_system_config_path", return_value=config_path):
            prepared = prepare_ue_python(root, root / ".agent", "_uedev_result = {'ok': 1}", mode="commandlet")

        class Completed:
            returncode = 0
            stdout = "stdout text"
            stderr = "stderr text"

        def fake_run(*args, **kwargs):
            prepared.result_path.write_text(
                json.dumps({"run_id": prepared.run_id, "ok": True, "status": "completed", "result": {"ok": 1}}),
                encoding="utf-8",
            )
            prepared.heartbeat_path.write_text(
                json.dumps({"run_id": prepared.run_id, "status": "completed"}),
                encoding="utf-8",
            )
            return Completed()

        with patch("uedev.ue.subprocess.run", side_effect=fake_run):
            result = execute_prepared_ue_python(prepared, cwd=root)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stdout, "stdout text")
        self.assertEqual(prepared.stdout_path.read_text(encoding="utf-8"), "stdout text")
        self.assertEqual(prepared.stderr_path.read_text(encoding="utf-8"), "stderr text")
        self.assertEqual(result.result_json["result"], {"ok": 1})


if __name__ == "__main__":
    unittest.main()
