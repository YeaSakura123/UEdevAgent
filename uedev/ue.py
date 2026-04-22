from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


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


# 内部函数：按当前平台把命令参数安全拼成可展示的命令行字符串。
def quote_command(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


# 外部函数：发现 UE 项目和编辑器路径，负责 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
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


# 外部函数：渲染 UE doctor 检查结果，负责 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
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


# 外部函数：包装 UE Python 脚本和 JSON 异常输出，负责 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def build_python_script(kind: str, user_script: str) -> str:
    """生成 UE 内执行脚本；所有输出走 JSON，方便 agent 解析观察结果。"""

    if kind == "list_assets":
        body = """
import unreal

registry = unreal.AssetRegistryHelpers.get_asset_registry()
assets = registry.get_assets_by_path("/Game", recursive=True)
payload = [
    {
        "asset_name": str(asset.asset_name),
        "asset_class": str(asset.asset_class_path.asset_name),
        "object_path": str(asset.package_name),
    }
    for asset in assets
]
print(json.dumps({"ok": True, "assets": payload}, ensure_ascii=False))
"""
    elif kind == "validate_assets":
        body = """
import unreal

# 这里采用 EditorValidatorSubsystem。不同 UE 版本 API 名称可能略有变化，
# 因此脚本保留异常输出，便于在真实项目里按版本快速调整。
subsystem = unreal.get_editor_subsystem(unreal.EditorValidatorSubsystem)
assets = unreal.EditorAssetLibrary.list_assets("/Game", recursive=True, include_folder=False)
loaded_assets = [unreal.EditorAssetLibrary.load_asset(path) for path in assets]
results = subsystem.validate_assets_with_settings(loaded_assets, unreal.AssetValidationSettings())
print(json.dumps({"ok": True, "validated_count": len(loaded_assets), "result": str(results)}, ensure_ascii=False))
"""
    else:
        body = user_script

    return "\n".join(
        [
            "import json",
            "import traceback",
            "",
            "try:",
            _indent(body.strip() or "print(json.dumps({'ok': True}))"),
            "except Exception as exc:",
            "    print(json.dumps({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}, ensure_ascii=False))",
            "    raise",
            "",
        ]
    )


# 外部函数：准备或执行 UE Python 脚本，负责 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def run_ue_python(
    cwd: Path,
    agent_dir: Path,
    script: str,
    *,
    mode: str = "commandlet",
    kind: str = "custom",
    execute: bool = False,
    timeout_seconds: int = 300,
) -> UeRunResult:
    discovery = discover_ue(cwd)
    if discovery.project_path is None:
        raise RuntimeError("Cannot find .uproject. Set UE_PROJECT_PATH or run from a UE project directory.")

    editor_path = discovery.editor_cmd_path if mode == "commandlet" else discovery.editor_gui_path
    if editor_path is None:
        missing = "UE_EDITOR_CMD_PATH" if mode == "commandlet" else "UE_EDITOR_PATH"
        raise RuntimeError(f"Cannot find UE editor executable. Set {missing} or UE_ENGINE_ROOT.")

    scripts_dir = agent_dir / "ue_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{kind}_{int(time.time())}.py"
    script_path.write_text(build_python_script(kind, script), encoding="utf-8")

    if mode == "commandlet":
        parts = [
            str(editor_path),
            str(discovery.project_path),
            "-run=pythonscript",
            f"-script={script_path}",
            "-unattended",
            "-nop4",
            "-nosplash",
        ]
    elif mode == "full_editor":
        parts = [
            str(editor_path),
            str(discovery.project_path),
            f"-ExecutePythonScript={script_path}",
            "-nop4",
            "-nosplash",
        ]
    else:
        raise ValueError("mode must be commandlet or full_editor")

    command = quote_command(parts)
    if not execute:
        return UeRunResult(command=command, script_path=script_path, executed=False)

    process = subprocess.run(
        parts,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    return UeRunResult(
        command=command,
        script_path=script_path,
        executed=True,
        exit_code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


# 外部函数：渲染 UE Python dry-run 或执行结果，负责 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def render_run_result(result: UeRunResult) -> str:
    lines = [
        f"script: {result.script_path}",
        f"command: {result.command}",
        f"executed: {result.executed}",
    ]
    if result.exit_code is not None:
        lines.append(f"exit_code: {result.exit_code}")
        lines.append("stdout:")
        lines.append(result.stdout)
        lines.append("stderr:")
        lines.append(result.stderr)
    else:
        lines.append("dry_run: true; add --execute or allow_ue_execute to launch UE.")
    return "\n".join(lines)


# 内部函数：处理 _discover_project 辅助逻辑，支撑 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
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


# 内部函数：处理 _discover_editor 辅助逻辑，支撑 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
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


# 内部函数：处理 _existing_env_path 辅助逻辑，支撑 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def _existing_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return _maybe(Path(value).expanduser().resolve())


# 内部函数：处理 _maybe 辅助逻辑，支撑 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def _maybe(path: Path) -> Path | None:
    return path if path.exists() else None


# 内部函数：处理 _indent 辅助逻辑，支撑 Unreal Engine 发现、脚本包装、dry-run 和执行结果展示。
def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())
