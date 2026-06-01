from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from ..state.config import DEFAULT_WORKSPACE_EXCLUDED_DIRS

_UE_ASSET_SUFFIXES = {".uasset", ".umap", ".uexp", ".ubulk"}
_VALID_GREP_OUTPUT_MODES = {"content", "files", "count"}


@dataclass(frozen=True)
class _GrepMatch:
    path: Path
    line: int | None = None
    column: int | None = None
    text: str = ""
    kind: str = "text"


def safe_path(cwd: Path, raw_path: str, *, excluded_dirs: Iterable[str] | None = None) -> Path:
    """Return a lexical workspace path without resolving directory links."""

    root = _lexical_abs(cwd)
    path = _lexical_abs(root / raw_path)
    if not _is_within(root, path):
        raise ValueError(f"Path escapes workspace: {raw_path}")
    if _has_excluded_part(root, path, excluded_dirs):
        raise ValueError(f"Path is excluded from workspace tools: {raw_path}")
    return path


def read_file(cwd: Path, raw_path: str, limit: int | None = None, *, excluded_dirs: Iterable[str] | None = None) -> str:
    path = safe_path(cwd, raw_path, excluded_dirs=excluded_dirs)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit is not None and limit >= 0 and len(lines) > limit:
        lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
    return "\n".join(lines)


def write_file(cwd: Path, raw_path: str, content: str, *, excluded_dirs: Iterable[str] | None = None) -> str:
    path = safe_path(cwd, raw_path, excluded_dirs=excluded_dirs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


def edit_file(
    cwd: Path,
    raw_path: str,
    old_text: str,
    new_text: str,
    *,
    excluded_dirs: Iterable[str] | None = None,
) -> str:
    path = safe_path(cwd, raw_path, excluded_dirs=excluded_dirs)
    content = path.read_text(encoding="utf-8", errors="replace")
    if old_text not in content:
        raise ValueError(f"Text not found in {raw_path}")
    path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def list_files(
    cwd: Path,
    raw_path: str = ".",
    pattern: str = "*",
    limit: int = 200,
    *,
    excluded_dirs: Iterable[str] | None = None,
) -> str:
    root = _lexical_abs(cwd)
    path = safe_path(cwd, raw_path, excluded_dirs=excluded_dirs)
    excluded = _excluded_set(excluded_dirs)
    files: list[Path] = []

    if path.is_file():
        relative_to_search = Path(path.name)
        if relative_to_search.match(pattern):
            files.append(path)
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(name for name in dirnames if name.casefold() not in excluded)
            for filename in sorted(filenames):
                item = Path(dirpath) / filename
                if _has_excluded_part(root, _lexical_abs(item), excluded):
                    continue
                try:
                    relative_to_search = item.relative_to(path)
                except ValueError:
                    relative_to_search = Path(filename)
                if relative_to_search.match(pattern):
                    files.append(item)

    rendered = [_relative_text(root, item) for item in files[:limit]]
    if len(files) > limit:
        rendered.append(f"... ({len(files) - limit} more files)")
    return "\n".join(rendered) if rendered else "(no files)"


def grep(
    cwd: Path,
    pattern: str,
    raw_path: str = ".",
    glob: str | None = None,
    limit: int = 100,
    *,
    case_sensitive: bool = True,
    output_mode: str = "content",
    include_asset_paths: bool = True,
    excluded_dirs: Iterable[str] | None = None,
) -> str:
    if not str(pattern):
        raise ValueError("grep pattern cannot be empty")
    output_mode = output_mode.strip().lower()
    if output_mode not in _VALID_GREP_OUTPUT_MODES:
        raise ValueError("grep output_mode must be content, files, or count")

    regex = _compile_grep_pattern(pattern, case_sensitive)
    root = _lexical_abs(cwd)
    path = safe_path(cwd, raw_path or ".", excluded_dirs=excluded_dirs)
    result_limit = max(0, int(limit))

    try:
        matches = _grep_with_rg(root, path, pattern, glob, case_sensitive, excluded_dirs)
    except FileNotFoundError:
        matches = _grep_with_python(root, path, regex, glob, excluded_dirs)

    if include_asset_paths:
        matches.extend(_grep_asset_paths(root, path, regex, glob, excluded_dirs))

    return _render_grep_matches(root, matches, output_mode, result_limit)


def _lexical_abs(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(root: Path, path: Path) -> bool:
    root_text = os.path.normcase(os.fspath(root))
    path_text = os.path.normcase(os.fspath(path))
    try:
        return os.path.commonpath([root_text, path_text]) == root_text
    except ValueError:
        return False


def _has_excluded_part(root: Path, path: Path, excluded_dirs: Iterable[str] | None = None) -> bool:
    excluded = _excluded_set(excluded_dirs)
    if not excluded:
        return False
    rel = os.path.relpath(os.fspath(path), os.fspath(root))
    if rel == ".":
        return False
    return any(part.casefold() in excluded for part in Path(rel).parts)


def _excluded_set(excluded_dirs: Iterable[str] | None = None) -> set[str]:
    values = DEFAULT_WORKSPACE_EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs
    return {str(value).casefold() for value in values if str(value)}


def _excluded_values(excluded_dirs: Iterable[str] | None = None) -> list[str]:
    values = DEFAULT_WORKSPACE_EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs
    names: set[str] = set()
    for value in values:
        text = str(value)
        if text:
            names.add(text)
            names.add(text.casefold())
    return sorted(names)


def _relative_text(root: Path, path: Path) -> str:
    return os.path.relpath(os.fspath(path), os.fspath(root))


def _compile_grep_pattern(pattern: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(pattern, flags)
    except re.error as error:
        raise ValueError(f"Invalid grep pattern: {error}") from error


def _grep_with_rg(
    root: Path,
    path: Path,
    pattern: str,
    glob: str | None,
    case_sensitive: bool,
    excluded_dirs: Iterable[str] | None,
) -> list[_GrepMatch]:
    target = _relative_text(root, path)
    args = [
        "rg",
        "--json",
        "--line-number",
        "--column",
        "--color",
        "never",
    ]
    if not case_sensitive:
        args.append("-i")
    if glob:
        args.extend(["-g", glob])
    for excluded in _excluded_values(excluded_dirs):
        args.extend(["-g", f"!{excluded}/**"])
        args.extend(["-g", f"!**/{excluded}/**"])
    args.extend(["--", pattern, target])

    process = subprocess.run(
        args,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode not in {0, 1}:
        detail = process.stderr.strip() or f"rg exited with {process.returncode}"
        raise ValueError(f"grep failed: {detail}")

    matches: list[_GrepMatch] = []
    for raw_line in process.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        path_data = data.get("path")
        lines_data = data.get("lines")
        if not isinstance(path_data, dict) or not isinstance(lines_data, dict):
            continue
        raw_path = str(path_data.get("text") or "")
        if not raw_path:
            continue
        match_path = Path(raw_path)
        if not match_path.is_absolute():
            match_path = root / match_path
        match_path = _lexical_abs(match_path)
        if not _is_within(root, match_path) or _has_excluded_part(root, match_path, excluded_dirs):
            continue
        text = str(lines_data.get("text") or "").rstrip("\r\n")
        submatches = data.get("submatches")
        column = 1
        if isinstance(submatches, list) and submatches:
            first = submatches[0]
            if isinstance(first, dict):
                start = first.get("start")
                if isinstance(start, int):
                    column = start + 1
        line_number = data.get("line_number")
        matches.append(
            _GrepMatch(
                path=match_path,
                line=int(line_number) if isinstance(line_number, int) else None,
                column=column,
                text=text,
            )
        )
    return matches


def _grep_with_python(
    root: Path,
    path: Path,
    regex: re.Pattern[str],
    glob: str | None,
    excluded_dirs: Iterable[str] | None,
) -> list[_GrepMatch]:
    matches: list[_GrepMatch] = []
    for item in _iter_search_files(root, path, glob, excluded_dirs):
        if _is_ue_asset_path(item) or _looks_binary(item):
            continue
        try:
            with item.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    match = regex.search(line)
                    if match is None:
                        continue
                    matches.append(
                        _GrepMatch(
                            path=item,
                            line=line_number,
                            column=match.start() + 1,
                            text=line.rstrip("\r\n"),
                        )
                    )
        except OSError:
            continue
    return matches


def _grep_asset_paths(
    root: Path,
    path: Path,
    regex: re.Pattern[str],
    glob: str | None,
    excluded_dirs: Iterable[str] | None,
) -> list[_GrepMatch]:
    matches: list[_GrepMatch] = []
    for item in _iter_search_files(root, path, glob, excluded_dirs):
        if not _is_ue_asset_path(item):
            continue
        rel = _relative_text(root, item).replace("\\", "/")
        if regex.search(rel) or regex.search(item.name):
            matches.append(_GrepMatch(path=item, kind="asset"))
    return matches


def _iter_search_files(root: Path, path: Path, glob: str | None, excluded_dirs: Iterable[str] | None) -> Iterable[Path]:
    excluded = _excluded_set(excluded_dirs)
    if path.is_file():
        if _matches_glob(root, path, glob):
            yield path
        return

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(name for name in dirnames if name.casefold() not in excluded)
        for filename in sorted(filenames):
            item = _lexical_abs(Path(dirpath) / filename)
            if _has_excluded_part(root, item, excluded):
                continue
            if _matches_glob(root, item, glob):
                yield item


def _matches_glob(root: Path, path: Path, glob: str | None) -> bool:
    if not glob:
        return True
    rel = _relative_text(root, path).replace("\\", "/")
    rel_path = PurePosixPath(rel)
    if rel_path.match(glob):
        return True
    if glob.startswith("**/") and rel_path.match(glob[3:]):
        return True
    return False


def _is_ue_asset_path(path: Path) -> bool:
    return path.suffix.casefold() in _UE_ASSET_SUFFIXES


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(8192)
    except OSError:
        return True


def _render_grep_matches(root: Path, matches: list[_GrepMatch], output_mode: str, limit: int) -> str:
    if output_mode == "files":
        seen: set[str] = set()
        lines: list[str] = []
        for match in matches:
            rel = _relative_text(root, match.path)
            if rel in seen:
                continue
            seen.add(rel)
            lines.append(rel)
    elif output_mode == "count":
        counts: dict[str, int] = {}
        for match in matches:
            rel = _relative_text(root, match.path)
            counts[rel] = counts.get(rel, 0) + 1
        lines = [f"{path}: {count}" for path, count in counts.items()]
    else:
        lines = []
        for match in matches:
            rel = _relative_text(root, match.path)
            if match.kind == "asset":
                lines.append(f"{rel}: asset path match")
                continue
            lines.append(f"{rel}:{match.line or 0}:{match.column or 1}: {match.text}")

    if not lines:
        return "(no matches)"
    rendered = lines[:limit]
    if len(lines) > limit:
        rendered.append(f"... ({len(lines) - limit} more matches)")
    return "\n".join(rendered)
