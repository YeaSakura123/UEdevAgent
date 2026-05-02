from __future__ import annotations

import unittest
import uuid
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

from uedev.tasks import TodoManager
from uedev.ue import (
    UeRunResult,
    build_editor_executor_script,
    build_python_script,
    build_wrapper_script,
    discover_ue,
    enqueue_editor_stop,
    execute_prepared_ue_python,
    generate_run_id,
    prepare_ue_python,
    render_run_result,
    run_ue_python,
)


def workspace_temp_path() -> Path:
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


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
    def test_discover_project_from_environment(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text("{}", encoding="utf-8")

        os_module = __import__("os")
        old = os_module.environ.get("UE_PROJECT_PATH")
        os_module.environ["UE_PROJECT_PATH"] = str(project)
        try:
            result = discover_ue(root)
            self.assertEqual(result.project_path, project.resolve())
        finally:
            if old is None:
                os_module.environ.pop("UE_PROJECT_PATH", None)
            else:
                os_module.environ["UE_PROJECT_PATH"] = old

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
        editor = root / "UnrealEditor.exe"
        project.write_text("{}", encoding="utf-8")
        editor.write_text("", encoding="utf-8")

        os_module = __import__("os")
        old_project = os_module.environ.get("UE_PROJECT_PATH")
        old_editor = os_module.environ.get("UE_EDITOR_PATH")
        os_module.environ["UE_PROJECT_PATH"] = str(project)
        os_module.environ["UE_EDITOR_PATH"] = str(editor)
        try:
            prepared = prepare_ue_python(root, root / ".agent", "print('ok')", mode="full_editor")
            self.assertEqual(prepared.mode, "full_editor")
            self.assertIn("UnrealEditor.exe", prepared.command)
            self.assertIn("-ExecutePythonScript=", prepared.command)
            self.assertNotIn("set_keep_python_script_alive(True)", prepared.user_script_path.read_text(encoding="utf-8"))
            self.assertTrue(prepared.run_dir.exists())
            self.assertTrue((prepared.run_dir / "meta.json").exists())
            self.assertTrue(prepared.wrapper_path.exists())
            self.assertIsNotNone(prepared.task_path)
            self.assertIn("ue_queue", str(prepared.task_path))
        finally:
            if old_project is None:
                os_module.environ.pop("UE_PROJECT_PATH", None)
            else:
                os_module.environ["UE_PROJECT_PATH"] = old_project
            if old_editor is None:
                os_module.environ.pop("UE_EDITOR_PATH", None)
            else:
                os_module.environ["UE_EDITOR_PATH"] = old_editor

    def test_prepare_snapshots_source_script_content_and_meta(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        editor = root / "UnrealEditor-Cmd.exe"
        source = root / "hello_source.py"
        project.write_text("{}", encoding="utf-8")
        editor.write_text("", encoding="utf-8")
        source.write_text("_uedev_result = {'from': 'source'}\n", encoding="utf-8")

        os_module = __import__("os")
        old_project = os_module.environ.get("UE_PROJECT_PATH")
        old_editor = os_module.environ.get("UE_EDITOR_CMD_PATH")
        os_module.environ["UE_PROJECT_PATH"] = str(project)
        os_module.environ["UE_EDITOR_CMD_PATH"] = str(editor)
        try:
            prepared = prepare_ue_python(
                root,
                root / ".agent",
                source.read_text(encoding="utf-8"),
                mode="commandlet",
                source_script_path=source,
            )
            meta = json.loads((prepared.run_dir / "meta.json").read_text(encoding="utf-8"))

            self.assertEqual(prepared.user_script_path.read_text(encoding="utf-8"), "_uedev_result = {'from': 'source'}\n")
            self.assertEqual(meta["source_script"]["path"], str(source.resolve()))
            self.assertEqual(meta["source_script"]["available"], True)
            self.assertIn("sha256", meta["source_script"])
        finally:
            if old_project is None:
                os_module.environ.pop("UE_PROJECT_PATH", None)
            else:
                os_module.environ["UE_PROJECT_PATH"] = old_project
            if old_editor is None:
                os_module.environ.pop("UE_EDITOR_CMD_PATH", None)
            else:
                os_module.environ["UE_EDITOR_CMD_PATH"] = old_editor

    def test_render_running_full_editor_result(self) -> None:
        result = UeRunResult(
            command="UnrealEditor.exe Demo.uproject",
            script_path=Path("script.py"),
            executed=True,
            process_id=123,
            status="running",
            run_id="ue_test",
        )
        rendered = render_run_result(result)

        self.assertIn("run_id: ue_test", rendered)
        self.assertIn("status: running", rendered)
        self.assertIn("process_id: 123", rendered)

    def test_run_python_dry_run_requires_editor_path(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        editor = root / "UnrealEditor-Cmd.exe"
        project.write_text("{}", encoding="utf-8")
        editor.write_text("", encoding="utf-8")

        os_module = __import__("os")
        old_project = os_module.environ.get("UE_PROJECT_PATH")
        old_editor = os_module.environ.get("UE_EDITOR_CMD_PATH")
        os_module.environ["UE_PROJECT_PATH"] = str(project)
        os_module.environ["UE_EDITOR_CMD_PATH"] = str(editor)
        try:
            result = run_ue_python(root, root / ".agent", "print('ok')", execute=False)
            self.assertFalse(result.executed)
            self.assertIn("UnrealEditor-Cmd.exe", result.command)
            self.assertTrue(result.run_dir.exists())
            self.assertTrue((result.run_dir / "user_script.py").exists())
            self.assertTrue((result.run_dir / "wrapper.py").exists())
        finally:
            if old_project is None:
                os_module.environ.pop("UE_PROJECT_PATH", None)
            else:
                os_module.environ["UE_PROJECT_PATH"] = old_project
            if old_editor is None:
                os_module.environ.pop("UE_EDITOR_CMD_PATH", None)
            else:
                os_module.environ["UE_EDITOR_CMD_PATH"] = old_editor

    def test_commandlet_execution_saves_stdout_stderr_and_result(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        editor = root / "UnrealEditor-Cmd.exe"
        project.write_text("{}", encoding="utf-8")
        editor.write_text("", encoding="utf-8")

        os_module = __import__("os")
        old_project = os_module.environ.get("UE_PROJECT_PATH")
        old_editor = os_module.environ.get("UE_EDITOR_CMD_PATH")
        os_module.environ["UE_PROJECT_PATH"] = str(project)
        os_module.environ["UE_EDITOR_CMD_PATH"] = str(editor)
        try:
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
        finally:
            if old_project is None:
                os_module.environ.pop("UE_PROJECT_PATH", None)
            else:
                os_module.environ["UE_PROJECT_PATH"] = old_project
            if old_editor is None:
                os_module.environ.pop("UE_EDITOR_CMD_PATH", None)
            else:
                os_module.environ["UE_EDITOR_CMD_PATH"] = old_editor


if __name__ == "__main__":
    unittest.main()
