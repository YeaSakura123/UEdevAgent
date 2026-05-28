from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


CollaborationMode = Literal["default", "plan"]
PermissionMode = Literal["read_only", "default", "auto_review", "full_access"]
PermissionAction = Literal["allow", "ask", "deny"]
CommandRisk = Literal["readonly", "mutating", "network", "dangerous", "unknown"]

VALID_COLLABORATION_MODES: tuple[CollaborationMode, ...] = ("default", "plan")
VALID_PERMISSION_MODES: tuple[PermissionMode, ...] = ("read_only", "default", "auto_review", "full_access")


@dataclass(frozen=True)
class PermissionDecision:
    action: PermissionAction
    reason: str


_READ_TOOLS = {
    "read_file",
    "list_files",
    "todo_list",
    "task_get",
    "task_list",
    "background_check",
    "list_teammates",
    "read_inbox",
    "worktree_list",
    "ue_doctor",
    "p4_status",
    "p4_file_state",
    "p4_opened",
    "p4_diff",
    "load_skill",
}

_WRITE_TOOLS = {"write_file", "edit_file"}

_MUTATING_TOOLS = {
    "todo_update",
    "task_create",
    "task_update",
    "claim_task",
    "spawn_teammate",
    "send_message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_submit",
    "plan_review",
    "idle",
    "worktree_create",
    "worktree_keep",
    "worktree_remove",
    "ue_stop_executor",
    "p4_checkout",
    "p4_add",
    "p4_delete",
    "p4_reconcile",
}

_EXECUTION_TOOLS = {"shell", "background_run", "worktree_run", "ue_run_python", "ue_build", "subagent"}

_PLAN_ALLOWED_TOOLS = {
    *_READ_TOOLS,
    "compact",
}

_DANGEROUS_COMMAND_PATTERNS = (
    "git reset --hard",
    "git clean",
    "git checkout --",
    "remove-item",
    "rm ",
    "rmdir",
    "del ",
    "erase ",
    "format ",
    "p4 submit",
    "p4 delete",
    "p4 revert",
    "shutdown",
)

_NETWORK_COMMAND_PATTERNS = (
    "curl",
    "wget",
    "invoke-webrequest",
    "invoke-restmethod",
    "iwr ",
    "irm ",
    "git clone",
    "git fetch",
    "git pull",
    "git push",
    "npm install",
    "pnpm install",
    "yarn add",
    "pip install",
    "python -m pip install",
)

_MUTATING_COMMAND_PATTERNS = (
    ">",
    ">>",
    "out-file",
    "set-content",
    "add-content",
    "new-item",
    "copy-item",
    "move-item",
    "mkdir",
    "git commit",
    "git merge",
    "git rebase",
    "git apply",
    "p4 edit",
    "p4 add",
    "p4 reconcile",
)

_READONLY_COMMAND_EXACT = {
    "pwd",
    "cd",
    "dir",
    "ls",
    "gci",
    "get-childitem",
    "git status",
    "git diff",
    "git log",
    "git branch",
    "git rev-parse",
    "git ls-files",
    "p4 info",
    "p4 opened",
    "p4 fstat",
    "p4 where",
    "p4 diff",
    "uedev ue doctor",
}

_READONLY_COMMAND_PREFIXES = (
    "rg ",
    "findstr ",
    "where ",
    "where.exe ",
    "type ",
    "cat ",
    "echo ",
    "get-content ",
    "gc ",
    "select-string ",
    "git show ",
    "p4 opened ",
    "p4 fstat ",
    "p4 where ",
    "p4 diff ",
    "python -m compileall ",
    "python -m unittest ",
    "py -m unittest ",
    "pytest",
)


def normalize_permission_mode(raw: str | None) -> PermissionMode | None:
    value = (raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases: dict[str, PermissionMode] = {
        "read-only": "read_only",
        "readonly": "read_only",
        "default": "default",
        "auto-review": "auto_review",
        "autoreview": "auto_review",
        "full-access": "full_access",
        "fullaccess": "full_access",
    }
    return aliases.get(value)


def permission_mode_label(mode: PermissionMode) -> str:
    return mode.replace("_", "-")


def permission_mode_description(mode: PermissionMode) -> str:
    descriptions = {
        "read_only": "Can read files in the current workspace. Approval is required to edit files or access the internet.",
        "default": "Can read and edit files in the current workspace, and run commands. Approval is required to access the internet or edit other files.",
        "auto_review": "Same workspace-write permissions as Default, but eligible on-request approvals are routed through the auto-review policy.",
        "full_access": "Skips approval for command execution and internet access. Existing workspace safe-path checks still apply to file tools.",
    }
    return descriptions[mode]


def format_permission_modes(active: PermissionMode) -> str:
    lines = [f"Permission mode: {permission_mode_label(active)}", "Available modes:"]
    for mode in VALID_PERMISSION_MODES:
        marker = " (active)" if mode == active else ""
        lines.append(f"- {permission_mode_label(mode)}{marker}: {permission_mode_description(mode)}")
    return "\n".join(lines)


def is_proposed_plan(answer: str) -> bool:
    stripped = answer.strip()
    return stripped.startswith("<proposed_plan>") and stripped.endswith("</proposed_plan>")


def classify_tool_permission(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    collaboration_mode: CollaborationMode,
    permission_mode: PermissionMode,
) -> PermissionDecision:
    if collaboration_mode == "plan":
        plan_decision = _classify_plan_mode(tool_name, tool_input)
        if plan_decision is not None:
            return plan_decision

    if tool_name == "p4_delete":
        if permission_mode == "read_only":
            return PermissionDecision("ask", "read-only mode requires approval before scheduling Perforce deletes")
        return PermissionDecision("ask", "p4_delete requires explicit approval because it schedules Perforce deletes")

    if tool_name.startswith("mcp__"):
        if permission_mode == "read_only":
            return PermissionDecision("ask", "read-only mode requires approval before calling external MCP tools")
        return PermissionDecision("allow", f"{permission_mode_label(permission_mode)} mode allows MCP tool execution")

    if permission_mode == "full_access":
        return PermissionDecision("allow", "full-access mode allows tool execution without confirmation")

    if tool_name in _READ_TOOLS or tool_name == "compact":
        return PermissionDecision("allow", "read-only tool")

    if tool_name in _WRITE_TOOLS:
        if permission_mode == "read_only":
            return PermissionDecision("ask", "read-only mode requires approval before editing files")
        return PermissionDecision("allow", "workspace-write mode allows workspace file edits")

    if tool_name in {"shell", "background_run", "worktree_run"}:
        command = str(tool_input.get("command", "")).strip()
        return _classify_command_permission(command, permission_mode)

    if tool_name in {"ue_run_python", "ue_build"}:
        if permission_mode == "read_only":
            return PermissionDecision("ask", "read-only mode requires approval before launching Unreal Engine")
        return PermissionDecision("allow", "workspace-write mode allows local command execution")

    if tool_name in _MUTATING_TOOLS or tool_name == "subagent":
        if permission_mode == "read_only":
            return PermissionDecision("ask", "read-only mode requires approval before changing agent state")
        return PermissionDecision("allow", "workspace-write mode allows agent state changes")

    if permission_mode == "auto_review":
        return PermissionDecision("ask", f"auto-review does not recognize tool {tool_name}")
    return PermissionDecision("allow", f"default mode allows tool {tool_name}")


def classify_shell_command(command: str) -> CommandRisk:
    normalized = _normalize_command(command)
    if not normalized:
        return "unknown"
    if _contains_any(normalized, _DANGEROUS_COMMAND_PATTERNS):
        return "dangerous"
    if _contains_any(normalized, _NETWORK_COMMAND_PATTERNS):
        return "network"
    if _contains_any(normalized, _MUTATING_COMMAND_PATTERNS):
        return "mutating"
    if normalized in _READONLY_COMMAND_EXACT or normalized.startswith(_READONLY_COMMAND_PREFIXES):
        return "readonly"
    return "unknown"


def _classify_plan_mode(tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision | None:
    if tool_name in _PLAN_ALLOWED_TOOLS:
        return None
    if tool_name in {"shell", "background_run", "worktree_run"}:
        command = str(tool_input.get("command", "")).strip()
        risk = classify_shell_command(command)
        if risk == "readonly":
            return None
        return PermissionDecision("deny", f"Plan Mode only allows read-only commands; classified this command as {risk}")
    return PermissionDecision("deny", f"Plan Mode blocks mutating or execution tool {tool_name}")


def _classify_command_permission(command: str, permission_mode: PermissionMode) -> PermissionDecision:
    risk = classify_shell_command(command)
    if permission_mode == "read_only":
        if risk == "readonly":
            return PermissionDecision("allow", "read-only command")
        if risk == "dangerous":
            return PermissionDecision("deny", "dangerous command is blocked in read-only mode")
        return PermissionDecision("ask", f"read-only mode requires approval for {risk} commands")

    if permission_mode == "auto_review":
        if risk in {"readonly", "mutating", "unknown"}:
            return PermissionDecision("allow", f"auto-review allows {risk} local command")
        if risk == "network":
            return PermissionDecision("ask", "auto-review requires approval for internet access")
        return PermissionDecision("deny", "auto-review rejected a dangerous command")

    if risk == "network":
        return PermissionDecision("ask", "default mode requires approval for internet access")
    if risk == "dangerous":
        return PermissionDecision("ask", "default mode requires approval for dangerous commands")
    return PermissionDecision("allow", f"default mode allows {risk} local command")


def _normalize_command(command: str) -> str:
    normalized = command.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _contains_any(command: str, patterns: tuple[str, ...]) -> bool:
    padded = f" {command} "
    return any(pattern in padded for pattern in patterns)
