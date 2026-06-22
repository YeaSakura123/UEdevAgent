from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import PerforceDiscovery, _discover_perforce

UE_BINARY_ASSET_SUFFIXES = {".uasset", ".umap", ".ubulk", ".uexp", ".uptnl", ".ushaderbytecode"}
MAX_RENDERED_OUTPUT = 20000


@dataclass(frozen=True)
class P4CommandResult:
    args: list[str]
    cwd: Path
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    status: str = "completed"

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def command(self) -> str:
        return " ".join(["p4", *self.args])


def run_p4(args: list[str], cwd: Path, timeout_seconds: int = 30) -> P4CommandResult:
    try:
        result = subprocess.run(
            ["p4", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return P4CommandResult(args, cwd, None, stderr="p4 executable not found on PATH.", status="missing")
    except subprocess.TimeoutExpired:
        return P4CommandResult(args, cwd, None, stderr=f"p4 {' '.join(args)} timed out after {timeout_seconds}s.", status="timeout")
    except OSError as error:
        return P4CommandResult(args, cwd, None, stderr=f"Cannot run p4 {' '.join(args)}: {error}", status="error")
    return P4CommandResult(args, cwd, result.returncode, result.stdout, result.stderr)


def p4_status(cwd: Path) -> str:
    project_path = _find_uproject(cwd)
    discovery = _discover_perforce(project_path.parent if project_path is not None else cwd, project_path)
    return _render_json(
        {
            "ok": discovery.available and discovery.in_workspace,
            "available": discovery.available,
            "in_workspace": discovery.in_workspace,
            "project_tracked": discovery.project_tracked,
            "client_name": discovery.client_name,
            "client_root": str(discovery.client_root) if discovery.client_root is not None else None,
            "user_name": discovery.user_name,
            "server_address": discovery.server_address,
            "project_depot_path": discovery.project_depot_path,
            "opened_count": discovery.opened_count,
            "opened_preview": discovery.opened_preview,
            "notes": discovery.notes,
        }
    )


def p4_file_state(cwd: Path, paths: list[str]) -> str:
    return _render_json({"ok": True, "files": [_file_state(cwd, path) for path in paths]})


def p4_opened(cwd: Path, changelist: str | None = None) -> str:
    args = ["opened"]
    if changelist:
        args.extend(["-c", changelist])
    result = run_p4(args, cwd)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    no_opened = not result.ok and _looks_like_no_opened_files(_first_nonempty(result.stderr) or _first_nonempty(result.stdout))
    return _render_json(
        {
            **_result_payload(result),
            "ok": result.ok or no_opened,
            "opened_count": 0 if no_opened else len(lines),
            "opened": [] if no_opened else lines,
        }
    )


def p4_checkout(cwd: Path, paths: list[str], changelist: str | None = None) -> str:
    resolved = _resolve_paths(cwd, paths)
    states = [_file_state_for_resolved(cwd, path) for path in resolved]
    conflicts = [
        state
        for state in states
        if state.get("binary_asset") and (state.get("other_open") or state.get("other_lock"))
    ]
    if conflicts:
        return _render_json(
            {
                "ok": False,
                "status": "conflict",
                "message": "Perforce checkout blocked by exclusive binary asset conflict.",
                "conflicts": conflicts,
            }
        )

    args = ["edit", *_changelist_args(changelist), *[str(path) for path in resolved]]
    result = run_p4(args, cwd)
    payload = _result_payload(result)
    if not result.ok and _looks_like_exclusive_lock_error(result.stderr or result.stdout):
        payload["status"] = "conflict"
        payload["message"] = "p4 edit reported an exclusive lock conflict."
    payload["files"] = [str(path) for path in resolved]
    return _render_json(payload)


def p4_add(cwd: Path, paths: list[str], changelist: str | None = None) -> str:
    resolved = _resolve_paths(cwd, paths)
    result = run_p4(["add", *_changelist_args(changelist), *[str(path) for path in resolved]], cwd)
    return _render_json({**_result_payload(result), "files": [str(path) for path in resolved]})


def p4_delete(cwd: Path, paths: list[str], changelist: str | None = None) -> str:
    resolved = _resolve_paths(cwd, paths)
    result = run_p4(["delete", *_changelist_args(changelist), *[str(path) for path in resolved]], cwd)
    return _render_json({**_result_payload(result), "files": [str(path) for path in resolved]})


def p4_reconcile(cwd: Path, paths: list[str] | None = None, changelist: str | None = None) -> str:
    resolved = _resolve_paths(cwd, paths or []) if paths else []
    result = run_p4(["reconcile", *_changelist_args(changelist), *[str(path) for path in resolved]], cwd)
    return _render_json({**_result_payload(result), "files": [str(path) for path in resolved]})


def p4_diff(cwd: Path, paths: list[str] | None = None) -> str:
    resolved = _resolve_paths(cwd, paths or []) if paths else []
    text_paths: list[str] = []
    skipped_binary: list[dict[str, Any]] = []

    if resolved:
        for path in resolved:
            state = _file_state_for_resolved(cwd, path)
            if state.get("binary_asset"):
                skipped_binary.append({"path": str(path), "type": state.get("type") or state.get("head_type")})
            else:
                text_paths.append(str(path))
    else:
        opened = _opened_records(cwd)
        for record in opened:
            path_text = record.get("path")
            if not path_text:
                continue
            if _is_binary_asset_path(path_text) or _is_binary_p4_type(record.get("type")):
                skipped_binary.append({"path": path_text, "type": record.get("type")})
            else:
                text_paths.append(path_text)

    if not text_paths:
        return _render_json({"ok": True, "diff": "", "skipped_binary": skipped_binary, "message": "No text files to diff."})

    result = run_p4(["diff", *text_paths], cwd)
    return _render_json(
        {
            **_result_payload(result, stdout_limit=MAX_RENDERED_OUTPUT),
            "files": text_paths,
            "skipped_binary": skipped_binary,
            "diff": _truncate(result.stdout, MAX_RENDERED_OUTPUT),
        }
    )


def _file_state(cwd: Path, raw_path: str) -> dict[str, Any]:
    return _file_state_for_resolved(cwd, _resolve_workspace_path(cwd, raw_path))


def _file_state_for_resolved(cwd: Path, path: Path) -> dict[str, Any]:
    result = run_p4(["fstat", str(path)], cwd)
    base: dict[str, Any] = {
        "path": str(path),
        "ok": result.ok,
        "tracked": False,
        "opened": False,
        "binary_asset": _is_binary_asset_path(str(path)),
        "messages": [],
    }
    if not result.ok:
        message = _first_nonempty(result.stderr) or _first_nonempty(result.stdout)
        base["error"] = message
        base["tracked"] = not _looks_like_untracked_p4_file(message)
        return base

    fields = _parse_fstat(result.stdout)
    other_open = _field_values(fields, "otherOpen")
    other_lock = _field_values(fields, "otherLock")
    p4_type = _first_field(fields, "type")
    head_type = _first_field(fields, "headType")
    binary_asset = _is_binary_asset_path(str(path)) or _is_binary_p4_type(p4_type) or _is_binary_p4_type(head_type)
    return {
        **base,
        "ok": True,
        "tracked": bool(_first_field(fields, "depotFile")),
        "depot_file": _first_field(fields, "depotFile"),
        "client_file": _first_field(fields, "clientFile"),
        "head_rev": _first_field(fields, "headRev"),
        "have_rev": _first_field(fields, "haveRev"),
        "action": _first_field(fields, "action"),
        "type": p4_type,
        "head_type": head_type,
        "opened": bool(_first_field(fields, "action")),
        "other_open": other_open,
        "other_lock": other_lock,
        "binary_asset": binary_asset,
        "conflict": binary_asset and bool(other_open or other_lock),
    }


def _opened_records(cwd: Path) -> list[dict[str, str]]:
    result = run_p4(["opened"], cwd)
    if not result.ok:
        return []
    records: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        path_part = stripped.split("#", 1)[0].strip()
        type_part = ""
        if "(" in stripped and stripped.endswith(")"):
            type_part = stripped.rsplit("(", 1)[1].removesuffix(")")
        records.append({"path": path_part, "type": type_part})
    return records


def _resolve_paths(cwd: Path, paths: list[str]) -> list[Path]:
    if not paths:
        raise ValueError("paths must contain at least one path")
    return [_resolve_workspace_path(cwd, path) for path in paths]


def _resolve_workspace_path(cwd: Path, raw_path: str) -> Path:
    if not str(raw_path).strip():
        raise ValueError("path cannot be empty")
    root = cwd.resolve()
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes workspace: {raw_path}")
    return resolved


def _find_uproject(cwd: Path) -> Path | None:
    root = cwd.resolve()
    for parent in [root, *root.parents]:
        projects = sorted(parent.glob("*.uproject"))
        if projects:
            return projects[0].resolve()
    return None


def _parse_fstat(stdout: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("... "):
            continue
        key, _, value = stripped[4:].partition(" ")
        fields.setdefault(key.strip(), []).append(value.strip())
    return fields


def _first_field(fields: dict[str, list[str]], key: str) -> str | None:
    values = fields.get(key)
    if not values:
        return None
    return values[0] or None


def _field_values(fields: dict[str, list[str]], prefix: str) -> list[str]:
    values: list[str] = []
    for key, items in fields.items():
        if key == prefix:
            continue
        if key.startswith(prefix):
            values.extend(item for item in items if item)
    return values


def _changelist_args(changelist: str | None) -> list[str]:
    value = (changelist or "").strip()
    return ["-c", value] if value else []


def _result_payload(result: P4CommandResult, stdout_limit: int = 4000) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "status": result.status,
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": _truncate(result.stdout, stdout_limit),
        "stderr": _truncate(result.stderr, stdout_limit),
    }


def _render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}\n...[truncated {len(value) - max_chars} chars]"


def _first_nonempty(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_binary_asset_path(path: str) -> bool:
    return Path(path).suffix.lower() in UE_BINARY_ASSET_SUFFIXES


def _is_binary_p4_type(value: str | None) -> bool:
    return bool(value and "binary" in value.lower())


def _looks_like_untracked_p4_file(message: str) -> bool:
    lowered = message.lower()
    return "no such file" in lowered or "not in client view" in lowered or "file(s) not in client view" in lowered


def _looks_like_no_opened_files(message: str) -> bool:
    return "file(s) not opened" in message.lower()


def _looks_like_exclusive_lock_error(message: str) -> bool:
    lowered = message.lower()
    return "exclusive" in lowered and ("opened" in lowered or "lock" in lowered)


__all__ = [
    "P4CommandResult",
    "PerforceDiscovery",
    "p4_add",
    "p4_checkout",
    "p4_delete",
    "p4_diff",
    "p4_file_state",
    "p4_opened",
    "p4_reconcile",
    "p4_status",
    "run_p4",
    "_discover_perforce",
]
