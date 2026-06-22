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
    parse_build_diagnostics,
    prepare_ue_python,
    render_build_result,
    render_doctor,
    render_run_result,
    run_ue_build,
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


def create_build_bat(root: Path) -> Path:
    build_bat = root / "Engine" / "Build" / "BatchFiles" / "Build.bat"
    build_bat.parent.mkdir(parents=True, exist_ok=True)
    build_bat.write_text("", encoding="utf-8")
    return build_bat


class UePythonRunnerTests(unittest.TestCase):
    def test_generate_run_id_is_sortable_and_file_safe(self) -> None:
        first = generate_run_id()
        second = generate_run_id()

        self.assertTrue(first.startswith("ue_"))
        self.assertLess(first, second)
        self.assertNotIn(":", first)
        self.assertNotIn("\\", first)

    def test_run_ue_build_uses_editor_target_and_writes_logs(self) -> None:
        root = workspace_temp_path()
        project = root / "UEAgentDemo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        build_bat = create_build_bat(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)
        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_run(args, **kwargs):
            if args[0] == "p4":
                raise FileNotFoundError("p4")
            calls.append((list(args), kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="build ok\n", stderr="")

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            with patch("uedev.ue.core.os.name", "nt"):
                with patch("uedev.ue.subprocess.run", side_effect=fake_run):
                    result = run_ue_build(root, root / ".agent", timeout_seconds=99)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.target, "UEAgentDemoEditor")
        self.assertEqual(result.platform, "Win64")
        self.assertEqual(result.configuration, "Development")
        self.assertEqual(calls[0][0], [str(build_bat), "UEAgentDemoEditor", "Win64", "Development", f"-Project={project.resolve()}", "-WaitMutex", "-FromMsBuild"])
        self.assertEqual(calls[0][1]["cwd"], str(root.resolve()))
        self.assertEqual(calls[0][1]["timeout"], 99)
        self.assertEqual(result.stdout_path.read_text(encoding="utf-8"), "build ok\n")
        meta = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
        payload = json.loads(result.result_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["target"], "UEAgentDemoEditor")
        self.assertEqual(payload["status"], "completed")
        rendered = render_build_result(result)
        self.assertIn("status: completed", rendered)
        self.assertIn("UEAgentDemoEditor", rendered)

    def test_run_ue_build_returns_failed_diagnostics(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        source = root / "Source" / "DemoActor.cpp"
        source.parent.mkdir()
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        create_build_bat(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        def fake_run(args, **kwargs):
            if args[0] == "p4":
                raise FileNotFoundError("p4")
            stdout = f"{source}(12): error C2143: syntax error\nERROR: UnrealHeaderTool failed\n"
            return subprocess.CompletedProcess(args, 6, stdout=stdout, stderr="")

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            with patch("uedev.ue.core.os.name", "nt"):
                with patch("uedev.ue.subprocess.run", side_effect=fake_run):
                    result = run_ue_build(root, root / ".agent")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.exit_code, 6)
        self.assertEqual(result.diagnostics[0].severity, "error")
        self.assertEqual(result.diagnostics[0].file, str(source))
        self.assertEqual(result.diagnostics[0].line, 12)
        self.assertEqual(result.diagnostics[0].code, "C2143")
        self.assertTrue(any("UnrealHeaderTool failed" in item.message for item in result.diagnostics))
        payload = json.loads(result.result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["diagnostics"][0]["code"], "C2143")

    def test_run_ue_build_timeout_preserves_output(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        create_build_bat(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        def fake_run(args, **kwargs):
            if args[0] == "p4":
                raise FileNotFoundError("p4")
            raise subprocess.TimeoutExpired(args, kwargs["timeout"], output="partial stdout", stderr="partial stderr")

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            with patch("uedev.ue.core.os.name", "nt"):
                with patch("uedev.ue.subprocess.run", side_effect=fake_run):
                    result = run_ue_build(root, root / ".agent", timeout_seconds=1)

        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.exit_code)
        self.assertEqual(result.stdout_path.read_text(encoding="utf-8"), "partial stdout")
        self.assertEqual(result.stderr_path.read_text(encoding="utf-8"), "partial stderr")

    def test_run_ue_build_requires_build_bat(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
            with patch("uedev.ue.core.os.name", "nt"):
                with self.assertRaisesRegex(RuntimeError, "Build.bat"):
                    run_ue_build(root, root / ".agent")

    def test_parse_build_diagnostics_handles_msvc_ubt_and_uht(self) -> None:
        diagnostics = parse_build_diagnostics(
            "\n".join(
                [
                    "C:/Project/Source/Demo.h(8): warning C4996: deprecated",
                    "ERROR: Missing module dependency",
                    "UnrealHeaderTool failed for target DemoEditor",
                ]
            ),
            "",
        )

        self.assertEqual(diagnostics[0].severity, "warning")
        self.assertEqual(diagnostics[0].code, "C4996")
        self.assertEqual(diagnostics[1].message, "Missing module dependency")
        self.assertIn("UnrealHeaderTool failed", diagnostics[2].message)

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

    def test_prepare_full_editor_script_uses_gui_without_per_run_keep_alive(self) -> None:
        root = workspace_temp_path()
        project = root / "Demo.uproject"
        project.write_text(json.dumps({"EngineAssociation": "5.4"}), encoding="utf-8")
        engine_root = root / "UE_5.4"
        create_engine_root(engine_root, commandlet=False, gui=True)
        config_path = root / "system-config.json"
        write_system_config(config_path, "5.4", engine_root)

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
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

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
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

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
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

        with patch("uedev.state.config.default_system_config_path", return_value=config_path):
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

