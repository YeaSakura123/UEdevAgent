from __future__ import annotations

from pathlib import Path


def safe_path(cwd: Path, raw_path: str) -> Path:
    """限制文件工具只能访问当前工作区，避免 agent 越界读写。"""

    path = (cwd / raw_path).resolve()
    root = cwd.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes workspace: {raw_path}")
    return path


def read_file(cwd: Path, raw_path: str, limit: int | None = None) -> str:
    path = safe_path(cwd, raw_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit is not None and limit >= 0 and len(lines) > limit:
        lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
    return "\n".join(lines)


def write_file(cwd: Path, raw_path: str, content: str) -> str:
    path = safe_path(cwd, raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


def edit_file(cwd: Path, raw_path: str, old_text: str, new_text: str) -> str:
    path = safe_path(cwd, raw_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    if old_text not in content:
        raise ValueError(f"Text not found in {raw_path}")
    path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def list_files(cwd: Path, raw_path: str = ".", pattern: str = "*", limit: int = 200) -> str:
    path = safe_path(cwd, raw_path)
    files = sorted(item for item in path.rglob(pattern) if item.is_file())
    rendered = [str(item.relative_to(cwd.resolve())) for item in files[:limit]]
    if len(files) > limit:
        rendered.append(f"... ({len(files) - limit} more files)")
    return "\n".join(rendered) if rendered else "(no files)"
