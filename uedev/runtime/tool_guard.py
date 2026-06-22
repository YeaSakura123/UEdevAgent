from __future__ import annotations

from typing import Callable

from ..mcp.registry import is_mcp_tool_name


ToolHandler = Callable[[dict[str, object]], str]


def _permission_prompt_label(name: str, tool_input: dict[str, object]) -> str:
    if name in {"shell", "background_run", "worktree_run"}:
        return str(tool_input.get("command") or name)
    if name in {"write_file", "edit_file", "read_file", "list_files", "grep"}:
        path = str(tool_input.get("path") or "").strip()
        return f"{name} {path}".strip()
    if name == "ue_build":
        cwd = str(tool_input.get("cwd") or "").strip()
        return f"ue_build {cwd}".strip()
    if name == "worktree_remove":
        worktree = str(tool_input.get("name") or "").strip()
        return f"worktree_remove {worktree}".strip()
    if name.startswith("p4_"):
        paths = tool_input.get("paths")
        if isinstance(paths, list):
            rendered = " ".join(str(path) for path in paths[:3])
            if len(paths) > 3:
                rendered += f" ... ({len(paths)} paths)"
            return f"{name} {rendered}".strip()
        path = str(paths or tool_input.get("cwd") or "").strip()
        return f"{name} {path}".strip()
    if is_mcp_tool_name(name):
        return name
    return name

__all__ = ["ToolHandler", "_permission_prompt_label"]
