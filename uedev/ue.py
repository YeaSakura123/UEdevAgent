from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_LAST_RUN_TIME: datetime | None = None


@dataclass(frozen=True)
class UeDiscovery:
    project_path: Path | None
    editor_cmd_path: Path | None
    editor_gui_path: Path | None
    notes: list[str]


@dataclass(frozen=True)
class UeRunResult:
    command: str
    script_path: Path
    executed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    process_id: int | None = None
    status: str = "prepared"
    run_id: str = ""
    run_dir: Path | None = None
    user_script_path: Path | None = None
    wrapper_path: Path | None = None
    result_path: Path | None = None
    heartbeat_path: Path | None = None
    events_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    task_path: Path | None = None
    source_script_path: Path | None = None
    script_origin: str = "inline"
    result_json: dict[str, Any] | None = None
    heartbeat_status: dict[str, Any] | None = None


@dataclass(frozen=True)
class UePreparedRun:
    command: str
    command_parts: list[str]
    script_path: Path
    mode: str
    run_id: str
    run_dir: Path
    user_script_path: Path
    wrapper_path: Path
    result_path: Path
    heartbeat_path: Path
    events_path: Path
    stdout_path: Path
    stderr_path: Path
    task_path: Path | None
    executor_path: Path | None
    executor_heartbeat_path: Path | None
    source_script_path: Path | None
    script_origin: str


def quote_command(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def generate_run_id() -> str:
    global _LAST_RUN_TIME
    now = datetime.now(timezone.utc)
    if _LAST_RUN_TIME is not None and now <= _LAST_RUN_TIME:
        now = _LAST_RUN_TIME + timedelta(microseconds=1)
    _LAST_RUN_TIME = now
    stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
    return f"ue_{stamp}_{uuid.uuid4().hex[:8]}"


def discover_ue(cwd: Path) -> UeDiscovery:
    notes: list[str] = []
    project_path = _discover_project(cwd, notes)
    editor_cmd_path, editor_gui_path = _discover_editor(notes)
    return UeDiscovery(
        project_path=project_path,
        editor_cmd_path=editor_cmd_path,
        editor_gui_path=editor_gui_path,
        notes=notes,
    )


def render_doctor(discovery: UeDiscovery) -> str:
    lines = [
        "UE doctor",
        f"- project: {discovery.project_path or '(missing, set UE_PROJECT_PATH)'}",
        f"- UnrealEditor-Cmd: {discovery.editor_cmd_path or '(missing, set UE_EDITOR_CMD_PATH or UE_ENGINE_ROOT)'}",
        f"- UnrealEditor: {discovery.editor_gui_path or '(missing, set UE_EDITOR_PATH or UE_ENGINE_ROOT)'}",
    ]
    if discovery.notes:
        lines.append("- notes:")
        lines.extend(f"  - {note}" for note in discovery.notes)
    return "\n".join(lines)


def build_python_script(kind: str, user_script: str, *, keep_editor_open: bool = False) -> str:
    """Return the user-visible script body for a UE run.

    The final executable file is a generated wrapper that imports this body from
    user_script.py and writes structured results into the run directory.
    """

    if kind == "list_assets":
        return """
import unreal

registry = unreal.AssetRegistryHelpers.get_asset_registry()
assets = registry.get_assets_by_path("/Game", recursive=True)
_uedev_result = {
    "assets": [
        {
            "asset_name": str(asset.asset_name),
            "asset_class": str(asset.asset_class_path.asset_name),
            "object_path": str(asset.package_name),
        }
        for asset in assets
    ]
}
"""
    if kind == "validate_assets":
        return """
import unreal

subsystem = unreal.get_editor_subsystem(unreal.EditorValidatorSubsystem)
assets = unreal.EditorAssetLibrary.list_assets("/Game", recursive=True, include_folder=False)
loaded_assets = [unreal.EditorAssetLibrary.load_asset(path) for path in assets]
results = subsystem.validate_assets_with_settings(loaded_assets, unreal.AssetValidationSettings())
_uedev_result = {
    "validated_count": len(loaded_assets),
    "result": str(results),
}
"""

    return user_script


def build_wrapper_script(
    *,
    run_id: str,
    project_dir: Path,
    user_script_path: Path,
    result_path: Path,
    heartbeat_path: Path,
    events_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    values = {
        "run_id": run_id,
        "project_dir": str(project_dir),
        "user_script_path": str(user_script_path),
        "result_path": str(result_path),
        "heartbeat_path": str(heartbeat_path),
        "events_path": str(events_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    return f"""from __future__ import annotations

import contextlib
import io
import json
import os
import runpy as _uedev_runpy
import sys
import time
import traceback
import types

RUN_ID = {json.dumps(values["run_id"])}
PROJECT_DIR = {json.dumps(values["project_dir"])}
USER_SCRIPT_PATH = {json.dumps(values["user_script_path"])}
RESULT_PATH = {json.dumps(values["result_path"])}
HEARTBEAT_PATH = {json.dumps(values["heartbeat_path"])}
EVENTS_PATH = {json.dumps(values["events_path"])}
STDOUT_PATH = {json.dumps(values["stdout_path"])}
STDERR_PATH = {json.dumps(values["stderr_path"])}

_LOGS = []
_EMITTED = {{}}
_CHILD_RESULTS = []
_ORIGINAL_RUN_PATH = _uedev_runpy.run_path
_ORIGINAL_UNREAL_LOG = None
_ORIGINAL_UNREAL_WARNING = None
_ORIGINAL_UNREAL_ERROR = None


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _write_text(path, content):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {{str(key): _json_safe(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _append_event(event):
    event.setdefault("run_id", RUN_ID)
    event.setdefault("time", _now())
    directory = os.path.dirname(EVENTS_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\\n")


def _heartbeat(status):
    _atomic_json(HEARTBEAT_PATH, {{"run_id": RUN_ID, "status": status, "time": _now(), "updated_at": time.time()}})


def _record_log(message, level="info"):
    entry = {{"level": level, "message": str(message), "time": _now()}}
    _LOGS.append(entry)
    _append_event({{"type": "ue_log", "level": level, "message": str(message)}})


def _call_unreal_log(message, level="info"):
    try:
        import unreal
        if level == "error" and _ORIGINAL_UNREAL_ERROR is not None:
            _ORIGINAL_UNREAL_ERROR(message)
        elif level == "warning" and _ORIGINAL_UNREAL_WARNING is not None:
            _ORIGINAL_UNREAL_WARNING(message)
        elif _ORIGINAL_UNREAL_LOG is not None:
            _ORIGINAL_UNREAL_LOG(message)
        elif level == "error":
            unreal.log_error(message)
        elif level == "warning":
            unreal.log_warning(message)
        else:
            unreal.log(message)
    except Exception:
        pass


def _ue_log(message, level="info"):
    rendered = f"[uedev:{{RUN_ID}}:{{level}}] {{message}}"
    _call_unreal_log(rendered, level)
    _record_log(message, level)


def _uedev_emit(key, value):
    _EMITTED[str(key)] = _json_safe(value)
    return value


def _install_runtime_module():
    module = types.ModuleType("uedev_runtime")
    module.run_id = RUN_ID
    module.project_dir = PROJECT_DIR
    module.emit = _uedev_emit
    module.log = _ue_log
    module.heartbeat = lambda status="running": _heartbeat(status)
    sys.modules["uedev_runtime"] = module


def _install_unreal_log_proxy():
    global _ORIGINAL_UNREAL_LOG, _ORIGINAL_UNREAL_WARNING, _ORIGINAL_UNREAL_ERROR
    try:
        import unreal
    except Exception:
        return None

    _ORIGINAL_UNREAL_LOG = getattr(unreal, "log", None)
    _ORIGINAL_UNREAL_WARNING = getattr(unreal, "log_warning", None)
    _ORIGINAL_UNREAL_ERROR = getattr(unreal, "log_error", None)

    def _proxy_log(message):
        if _ORIGINAL_UNREAL_LOG is not None:
            _ORIGINAL_UNREAL_LOG(message)
        _record_log(message, "info")

    def _proxy_warning(message):
        if _ORIGINAL_UNREAL_WARNING is not None:
            _ORIGINAL_UNREAL_WARNING(message)
        elif _ORIGINAL_UNREAL_LOG is not None:
            _ORIGINAL_UNREAL_LOG(message)
        _record_log(message, "warning")

    def _proxy_error(message):
        if _ORIGINAL_UNREAL_ERROR is not None:
            _ORIGINAL_UNREAL_ERROR(message)
        elif _ORIGINAL_UNREAL_LOG is not None:
            _ORIGINAL_UNREAL_LOG(message)
        _record_log(message, "error")

    unreal.log = _proxy_log
    unreal.log_warning = _proxy_warning
    unreal.log_error = _proxy_error
    return unreal


def _restore_unreal_log_proxy(unreal_module):
    if unreal_module is None:
        return
    if _ORIGINAL_UNREAL_LOG is not None:
        unreal_module.log = _ORIGINAL_UNREAL_LOG
    if _ORIGINAL_UNREAL_WARNING is not None:
        unreal_module.log_warning = _ORIGINAL_UNREAL_WARNING
    if _ORIGINAL_UNREAL_ERROR is not None:
        unreal_module.log_error = _ORIGINAL_UNREAL_ERROR


def _patched_run_path(path_name, init_globals=None, run_name=None):
    child_emitted = {{}}

    def _child_emit(key, value):
        child_emitted[str(key)] = _json_safe(value)
        return value

    child_globals = dict(init_globals or {{}})
    child_globals.setdefault("_uedev_project_dir", PROJECT_DIR)
    child_globals.setdefault("_uedev_result", None)
    child_globals.setdefault("_uedev_emit", _child_emit)
    child_globals.setdefault("_uedev_log", _ue_log)
    child_globals.setdefault("_uedev_heartbeat", lambda status="running": _heartbeat(status))
    result_globals = _ORIGINAL_RUN_PATH(path_name, init_globals=child_globals, run_name=run_name)
    child_result = result_globals.get("_uedev_result")
    _CHILD_RESULTS.append(
        {{
            "path": str(path_name),
            "result": _json_safe(child_result),
            "emitted": _json_safe(child_emitted),
        }}
    )
    return result_globals


_uedev_result = None
_stdout = io.StringIO()
_stderr = io.StringIO()
_status = "running"
_error = None
_traceback = None
_previous_cwd = os.getcwd()
_unreal_module = None
_heartbeat("running")
_append_event({{"type": "begin"}})

try:
    _install_runtime_module()
    _unreal_module = _install_unreal_log_proxy()
    _uedev_runpy.run_path = _patched_run_path
    _ue_log("begin")
    os.chdir(PROJECT_DIR)
    with open(USER_SCRIPT_PATH, encoding="utf-8") as _script_file:
        _uedev_code = _script_file.read()
    _globals = {{
        "__name__": "__main__",
        "__file__": USER_SCRIPT_PATH,
        "_uedev_project_dir": PROJECT_DIR,
        "_uedev_result": None,
        "_uedev_emit": _uedev_emit,
        "_uedev_log": _ue_log,
        "_uedev_heartbeat": lambda status="running": _heartbeat(status),
    }}
    with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
        exec(compile(_uedev_code, USER_SCRIPT_PATH, "exec"), _globals)
    _status = "completed"
    _uedev_result = _globals.get("_uedev_result")
except Exception as exc:
    _status = "failed"
    _error = str(exc)
    _traceback = traceback.format_exc()
    _ue_log(str(exc), "error")
finally:
    _uedev_runpy.run_path = _ORIGINAL_RUN_PATH
    _restore_unreal_log_proxy(_unreal_module)
    try:
        os.chdir(_previous_cwd)
    except Exception:
        pass
    _stdout_text = _stdout.getvalue()
    _stderr_text = _stderr.getvalue()
    _write_text(STDOUT_PATH, _stdout_text)
    _write_text(STDERR_PATH, _stderr_text)
    _payload = {{
        "run_id": RUN_ID,
        "ok": _status == "completed",
        "status": _status,
        "result": _json_safe(_uedev_result if _status == "completed" else None),
        "emitted": _json_safe(_EMITTED),
        "child_results": _json_safe(_CHILD_RESULTS),
        "logs": _json_safe(_LOGS),
        "stdout": _stdout_text,
        "stderr": _stderr_text,
    }}
    if _error is not None:
        _payload["error"] = _error
    if _traceback is not None:
        _payload["traceback"] = _traceback
    _atomic_json(RESULT_PATH, _payload)
    _heartbeat(_status)
    _append_event({{"type": "end", "status": _status}})
    _ue_log(f"end {{_status}}")
"""


def build_editor_executor_script(agent_dir: Path) -> str:
    queue_dir = agent_dir / "ue_queue"
    heartbeat_path = agent_dir / "ue_executor" / "heartbeat.json"
    return f"""from __future__ import annotations

import json
import os
import time
import traceback

AGENT_DIR = {json.dumps(str(agent_dir))}
QUEUE_DIR = {json.dumps(str(queue_dir))}
HEARTBEAT_PATH = {json.dumps(str(heartbeat_path))}
_UDEV_STOPPED = False

try:
    import unreal
except Exception:
    unreal = None


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _heartbeat(status="running"):
    _atomic_json(HEARTBEAT_PATH, {{"status": status, "time": _now(), "updated_at": time.time()}})


def _log(message, level="info"):
    if unreal is None:
        return
    rendered = f"[uedev:executor:{{level}}] {{message}}"
    if level == "error":
        unreal.log_error(rendered)
    elif level == "warning":
        unreal.log_warning(rendered)
    else:
        unreal.log(rendered)


def _move(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    os.replace(src, dst)
    return dst


def _execute_task(task_path):
    running_dir = os.path.join(QUEUE_DIR, "running")
    done_dir = os.path.join(QUEUE_DIR, "done")
    failed_dir = os.path.join(QUEUE_DIR, "failed")
    running_path = _move(task_path, running_dir)
    try:
        with open(running_path, encoding="utf-8") as handle:
            task = json.load(handle)
        run_id = task.get("run_id", "")
        wrapper_path = task["wrapper_path"]
        _log(f"begin task {{run_id}}")
        with open(wrapper_path, encoding="utf-8") as wrapper_file:
            code = wrapper_file.read()
        exec(compile(code, wrapper_path, "exec"), {{"__name__": "__main__", "__file__": wrapper_path}})
        _move(running_path, done_dir)
        _log(f"end task {{run_id}}")
    except Exception as exc:
        _log(f"task failed: {{exc}}", "error")
        _log(traceback.format_exc(), "error")
        if os.path.exists(running_path):
            _move(running_path, failed_dir)


def _execute_stop(stop_path):
    global _UDEV_STOPPED
    done_dir = os.path.join(QUEUE_DIR, "done")
    _move(stop_path, done_dir)
    _UDEV_STOPPED = True
    _heartbeat("stopped")
    _log("executor stop requested")
    if unreal is not None:
        try:
            unreal.unregister_slate_pre_tick_callback(_UDEV_TICK_HANDLE)
        except Exception:
            pass
        try:
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        except Exception:
            pass


def _tick(delta_seconds=0.0):
    if _UDEV_STOPPED:
        return
    _heartbeat("running")
    pending_dir = os.path.join(QUEUE_DIR, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    stop_tasks = [
        os.path.join(pending_dir, name)
        for name in os.listdir(pending_dir)
        if name.endswith(".stop.json")
    ]
    stop_tasks.sort(key=lambda path: (os.path.getmtime(path), os.path.basename(path)))
    if stop_tasks:
        _execute_stop(stop_tasks[0])
        return
    tasks = [
        os.path.join(pending_dir, name)
        for name in os.listdir(pending_dir)
        if name.endswith(".task.json")
    ]
    tasks.sort(key=lambda path: (os.path.getmtime(path), os.path.basename(path)))
    if tasks:
        _execute_task(tasks[0])


_heartbeat("running")
_log("executor started")

if unreal is not None:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    _UDEV_TICK_HANDLE = unreal.register_slate_pre_tick_callback(_tick)
else:
    _tick(0.0)
"""


def run_ue_python(
    cwd: Path,
    agent_dir: Path,
    script: str,
    *,
    mode: str = "commandlet",
    kind: str = "custom",
    execute: bool = False,
    timeout_seconds: int = 300,
    source_script_path: Path | None = None,
) -> UeRunResult:
    prepared = prepare_ue_python(cwd, agent_dir, script, mode=mode, kind=kind, source_script_path=source_script_path)
    if not execute:
        return _result_from_prepared(prepared, executed=False, status="prepared")
    return execute_prepared_ue_python(prepared, cwd=cwd, timeout_seconds=timeout_seconds)


def enqueue_editor_stop(agent_dir: Path) -> Path:
    _ensure_queue_dirs(agent_dir)
    run_id = generate_run_id()
    stop_path = agent_dir / "ue_queue" / "pending" / f"{run_id}.stop.json"
    _write_json(stop_path, {"run_id": run_id, "type": "stop", "created_at": _iso_now()})
    return stop_path


def prepare_ue_python(
    cwd: Path,
    agent_dir: Path,
    script: str,
    *,
    mode: str = "commandlet",
    kind: str = "custom",
    source_script_path: Path | None = None,
) -> UePreparedRun:
    discovery = discover_ue(cwd)
    if discovery.project_path is None:
        raise RuntimeError("Cannot find .uproject. Set UE_PROJECT_PATH or run from a UE project directory.")

    editor_path = discovery.editor_cmd_path if mode == "commandlet" else discovery.editor_gui_path
    if editor_path is None:
        missing = "UE_EDITOR_CMD_PATH" if mode == "commandlet" else "UE_EDITOR_PATH"
        raise RuntimeError(f"Cannot find UE editor executable. Set {missing} or UE_ENGINE_ROOT.")

    run_id = generate_run_id()
    runs_dir = agent_dir / "ue_runs"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_queue_dirs(agent_dir)

    user_script_path = run_dir / "user_script.py"
    wrapper_path = run_dir / "wrapper.py"
    result_path = run_dir / "result.json"
    heartbeat_path = run_dir / "heartbeat.json"
    events_path = run_dir / "events.jsonl"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    task_path = agent_dir / "ue_queue" / "pending" / f"{run_id}.task.json" if mode == "full_editor" else None

    user_script = build_python_script(kind, script, keep_editor_open=mode == "full_editor")
    script_origin = "script_path" if source_script_path is not None else "inline"
    user_script_path.write_text(user_script, encoding="utf-8")
    wrapper_path.write_text(
        build_wrapper_script(
            run_id=run_id,
            project_dir=discovery.project_path.parent,
            user_script_path=user_script_path,
            result_path=result_path,
            heartbeat_path=heartbeat_path,
            events_path=events_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        ),
        encoding="utf-8",
    )

    executor_path: Path | None = None
    executor_heartbeat_path: Path | None = None
    if mode == "commandlet":
        parts = [
            str(editor_path),
            str(discovery.project_path),
            "-run=pythonscript",
            f"-script={wrapper_path}",
            "-unattended",
            "-nop4",
            "-nosplash",
        ]
    elif mode == "full_editor":
        executor_path = agent_dir / "ue_executor" / "editor_executor.py"
        executor_heartbeat_path = agent_dir / "ue_executor" / "heartbeat.json"
        _write_executor(agent_dir)
        parts = [
            str(editor_path),
            str(discovery.project_path),
            f"-ExecutePythonScript={executor_path}",
            "-nop4",
            "-nosplash",
        ]
    else:
        raise ValueError("mode must be commandlet or full_editor")

    command = quote_command(parts)
    _write_json(
        run_dir / "meta.json",
        {
            "run_id": run_id,
            "mode": mode,
            "kind": kind,
            "project_path": str(discovery.project_path),
            "command": command,
            "created_at": _iso_now(),
            "user_script_path": str(user_script_path),
            "wrapper_path": str(wrapper_path),
            "result_path": str(result_path),
            "heartbeat_path": str(heartbeat_path),
            "events_path": str(events_path),
            "task_path": str(task_path) if task_path else None,
            "source_script": _source_script_meta(source_script_path),
            "script_origin": script_origin,
        },
    )
    _write_latest(agent_dir, run_id, run_dir, "prepared")

    return UePreparedRun(
        command=command,
        command_parts=parts,
        script_path=wrapper_path,
        mode=mode,
        run_id=run_id,
        run_dir=run_dir,
        user_script_path=user_script_path,
        wrapper_path=wrapper_path,
        result_path=result_path,
        heartbeat_path=heartbeat_path,
        events_path=events_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        task_path=task_path,
        executor_path=executor_path,
        executor_heartbeat_path=executor_heartbeat_path,
        source_script_path=source_script_path,
        script_origin=script_origin,
    )


def execute_prepared_ue_python(
    prepared: UePreparedRun,
    *,
    cwd: Path,
    timeout_seconds: int = 300,
) -> UeRunResult:
    _write_latest(prepared.run_dir.parent.parent, prepared.run_id, prepared.run_dir, "running")
    if prepared.mode == "full_editor":
        process = _ensure_editor_executor(prepared, cwd)
        _enqueue_editor_task(prepared)
        result_json, heartbeat_status, status = _wait_for_result(prepared, timeout_seconds)
        _write_latest(prepared.run_dir.parent.parent, prepared.run_id, prepared.run_dir, status)
        return _result_from_prepared(
            prepared,
            executed=True,
            process_id=process.pid if process else None,
            stdout=_read_text(prepared.stdout_path),
            stderr=_read_text(prepared.stderr_path),
            status=status,
            result_json=result_json,
            heartbeat_status=heartbeat_status,
        )

    try:
        process = subprocess.run(
            prepared.command_parts,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        prepared.stdout_path.write_text(process.stdout, encoding="utf-8")
        prepared.stderr_path.write_text(process.stderr, encoding="utf-8")
        result_json = _read_json(prepared.result_path)
        status = _status_from_result(result_json, default="completed" if process.returncode == 0 else "failed")
        _write_latest(prepared.run_dir.parent.parent, prepared.run_id, prepared.run_dir, status)
        return _result_from_prepared(
            prepared,
            executed=True,
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            status=status,
            result_json=result_json,
            heartbeat_status=_read_json(prepared.heartbeat_path),
        )
    except subprocess.TimeoutExpired as error:
        stdout = _coerce_output(error.stdout)
        stderr = _coerce_output(error.stderr)
        prepared.stdout_path.write_text(stdout, encoding="utf-8")
        prepared.stderr_path.write_text(stderr, encoding="utf-8")
        _write_latest(prepared.run_dir.parent.parent, prepared.run_id, prepared.run_dir, "timeout")
        return _result_from_prepared(
            prepared,
            executed=True,
            stdout=stdout,
            stderr=stderr,
            status="timeout",
            result_json=_read_json(prepared.result_path),
            heartbeat_status=_read_json(prepared.heartbeat_path),
        )


def render_run_result(result: UeRunResult) -> str:
    lines = [
        f"run_id: {result.run_id or '(legacy)'}",
        f"status: {result.status}",
        f"run_dir: {result.run_dir or '(none)'}",
        f"script_origin: {result.script_origin}",
    ]
    if result.user_script_path is not None:
        lines.append(f"user_script_path: {result.user_script_path}")
    if result.wrapper_path is not None:
        lines.append(f"wrapper_path: {result.wrapper_path}")
    elif result.script_path is not None:
        lines.append(f"script_path: {result.script_path}")
    if result.source_script_path is not None:
        lines.append(f"source_script_path: {result.source_script_path}")
    lines.extend(
        [
            f"command: {result.command}",
            f"executed: {result.executed}",
        ]
    )
    if result.result_path is not None:
        lines.append(f"result_path: {result.result_path}")
    if result.heartbeat_path is not None:
        lines.append(f"heartbeat_path: {result.heartbeat_path}")
    if result.events_path is not None:
        lines.append(f"events_path: {result.events_path}")
    if result.stdout_path is not None:
        lines.append(f"stdout_path: {result.stdout_path}")
    if result.stderr_path is not None:
        lines.append(f"stderr_path: {result.stderr_path}")
    if result.task_path is not None:
        lines.append(f"task_path: {result.task_path}")
    if result.exit_code is not None:
        lines.append(f"exit_code: {result.exit_code}")
    if result.process_id is not None:
        lines.append(f"process_id: {result.process_id}")
    if result.result_json is not None:
        lines.append("result:")
        lines.append(json.dumps(result.result_json.get("result"), ensure_ascii=False, indent=2))
        if result.result_json.get("emitted"):
            lines.append("emitted:")
            lines.append(json.dumps(result.result_json.get("emitted"), ensure_ascii=False, indent=2))
        if result.result_json.get("child_results"):
            lines.append("child_results:")
            lines.append(json.dumps(result.result_json.get("child_results"), ensure_ascii=False, indent=2))
        logs = result.result_json.get("logs")
        if isinstance(logs, list) and logs:
            lines.append("logs_recent:")
            lines.append(json.dumps(logs[-20:], ensure_ascii=False, indent=2))
        if result.result_json.get("error"):
            lines.append(f"error: {result.result_json.get('error')}")
        if result.result_json.get("traceback"):
            lines.append("traceback:")
            lines.append(str(result.result_json.get("traceback")))
    if result.heartbeat_status is not None:
        lines.append("heartbeat_status:")
        lines.append(json.dumps(result.heartbeat_status, ensure_ascii=False, indent=2))
    if result.stdout or result.exit_code is not None:
        lines.append("stdout:")
        lines.append(result.stdout)
    if result.stderr or result.exit_code is not None:
        lines.append("stderr:")
        lines.append(result.stderr)
    if not result.executed:
        lines.append("dry_run: true; add --execute for standalone UE CLI commands, or approve the agent confirmation prompt.")
    return "\n".join(lines)


def _result_from_prepared(
    prepared: UePreparedRun,
    *,
    executed: bool,
    status: str,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    process_id: int | None = None,
    result_json: dict[str, Any] | None = None,
    heartbeat_status: dict[str, Any] | None = None,
) -> UeRunResult:
    return UeRunResult(
        command=prepared.command,
        script_path=prepared.script_path,
        executed=executed,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        process_id=process_id,
        status=status,
        run_id=prepared.run_id,
        run_dir=prepared.run_dir,
        user_script_path=prepared.user_script_path,
        wrapper_path=prepared.wrapper_path,
        result_path=prepared.result_path,
        heartbeat_path=prepared.heartbeat_path,
        events_path=prepared.events_path,
        stdout_path=prepared.stdout_path,
        stderr_path=prepared.stderr_path,
        task_path=prepared.task_path,
        source_script_path=prepared.source_script_path,
        script_origin=prepared.script_origin,
        result_json=result_json,
        heartbeat_status=heartbeat_status,
    )


def _source_script_meta(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        resolved = path.resolve()
        data = resolved.read_bytes()
        stat = resolved.stat()
    except OSError:
        return {"path": str(path), "available": False}
    return {
        "path": str(resolved),
        "available": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_editor_executor(prepared: UePreparedRun, cwd: Path) -> subprocess.Popen[str] | None:
    if prepared.executor_heartbeat_path and _heartbeat_is_fresh(prepared.executor_heartbeat_path):
        return None
    return subprocess.Popen(
        prepared.command_parts,
        cwd=str(cwd),
        text=True,
    )


def _enqueue_editor_task(prepared: UePreparedRun) -> None:
    if prepared.task_path is None:
        raise RuntimeError("full_editor prepared run is missing a task path")
    payload = {
        "run_id": prepared.run_id,
        "created_at": _iso_now(),
        "wrapper_path": str(prepared.wrapper_path),
        "result_path": str(prepared.result_path),
        "heartbeat_path": str(prepared.heartbeat_path),
    }
    _write_json(prepared.task_path, payload)


def _wait_for_result(prepared: UePreparedRun, timeout_seconds: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    deadline = time.monotonic() + timeout_seconds
    heartbeat_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        result_json = _read_json(prepared.result_path)
        if result_json and result_json.get("run_id") == prepared.run_id:
            return result_json, _read_json(prepared.heartbeat_path), _status_from_result(result_json)
        heartbeat_status = _read_json(prepared.heartbeat_path) or heartbeat_status
        time.sleep(0.5)
    return _read_json(prepared.result_path), heartbeat_status or _read_json(prepared.heartbeat_path), "timeout"


def _write_executor(agent_dir: Path) -> None:
    executor_dir = agent_dir / "ue_executor"
    executor_dir.mkdir(parents=True, exist_ok=True)
    executor_path = executor_dir / "editor_executor.py"
    executor_path.write_text(build_editor_executor_script(agent_dir), encoding="utf-8")


def _ensure_queue_dirs(agent_dir: Path) -> None:
    for name in ["pending", "running", "done", "failed"]:
        (agent_dir / "ue_queue" / name).mkdir(parents=True, exist_ok=True)


def _write_latest(agent_dir: Path, run_id: str, run_dir: Path, status: str) -> None:
    _write_json(
        agent_dir / "ue_runs" / "latest.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": status,
            "updated_at": _iso_now(),
        },
    )
    index_path = agent_dir / "ue_runs" / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "status": status, "time": _iso_now()}, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _heartbeat_is_fresh(path: Path, *, max_age_seconds: int = 10) -> bool:
    data = _read_json(path)
    if not data:
        return False
    updated_at = data.get("updated_at")
    if not isinstance(updated_at, (int, float)):
        return False
    return (time.time() - float(updated_at)) <= max_age_seconds


def _status_from_result(result_json: dict[str, Any] | None, *, default: str = "completed") -> str:
    if not result_json:
        return default
    status = result_json.get("status")
    if isinstance(status, str) and status:
        return status
    return "completed" if result_json.get("ok") else "failed"


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _discover_project(cwd: Path, notes: list[str]) -> Path | None:
    env_project = os.environ.get("UE_PROJECT_PATH")
    if env_project:
        path = Path(env_project).expanduser().resolve()
        if path.exists() and path.suffix == ".uproject":
            return path
        notes.append(f"UE_PROJECT_PATH is set but invalid: {path}")

    for parent in [cwd.resolve(), *cwd.resolve().parents]:
        projects = sorted(parent.glob("*.uproject"))
        if projects:
            return projects[0]

    projects = sorted(cwd.glob("*.uproject"))
    if projects:
        return projects[0].resolve()
    notes.append("No .uproject found in current directory or parents.")
    return None


def _discover_editor(notes: list[str]) -> tuple[Path | None, Path | None]:
    cmd = _existing_env_path("UE_EDITOR_CMD_PATH")
    gui = _existing_env_path("UE_EDITOR_PATH")
    engine_root = os.environ.get("UE_ENGINE_ROOT")

    if engine_root:
        root = Path(engine_root).expanduser().resolve()
        cmd = cmd or _maybe(root / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe")
        gui = gui or _maybe(root / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe")

    if not cmd:
        notes.append("UE commandlet executable not found from environment.")
    if not gui:
        notes.append("UE full editor executable not found from environment.")
    return cmd, gui


def _existing_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return _maybe(Path(value).expanduser().resolve())


def _maybe(path: Path) -> Path | None:
    return path if path.exists() else None


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())
