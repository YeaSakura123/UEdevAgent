from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..llm.client import ChatMessage
from ..runtime.history import load_history_file, write_history_messages
from ..runtime.prompts import build_system_prompt
from ..runtime.skills import SkillLoader
from ..state.tasks import TaskManager
from .shell import shell_name


UE_LINKED_WORKTREE_KIND = "ue-linked-worktree"
UE_GIT_WORKTREE_MODE = "git-worktree-p4-content"
UE_LEGACY_TEXT_MODE = "git-text-p4-content"
SHARED_CONTENT_WARNING = "Content is shared; editing assets in this worktree edits the original project assets."

BRANCH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
UE_TEXT_PATHS = ("Source", "Config", "Plugins", "Build", "Scripts")
UE_LOCAL_EXCLUDES = (
    ".agent/",
    "Content/",
    "Binaries/",
    "Intermediate/",
    "Saved/",
    ".vs/",
    "DerivedDataCache/",
)


class WorktreeManager:
    def __init__(self, cwd: Path, worktrees_dir: Path, task_manager: TaskManager):
        self.cwd = cwd
        self.worktrees_dir = worktrees_dir
        self.index_path = worktrees_dir / "index.json"
        self.events_path = worktrees_dir / "events.jsonl"
        self.task_manager = task_manager
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        path = self.worktrees_dir / name
        branch = f"wt/{name}"
        self._emit("worktree.create.before", {"name": name, "task_id": task_id, "path": str(path)})
        if not path.exists():
            result = _run_git(self.cwd, ["worktree", "add", "-b", branch, str(path), base_ref])
            if result.returncode != 0:
                self._emit("worktree.create.failed", {"name": name, "stderr": result.stderr})
                raise RuntimeError(result.stderr.strip() or "git worktree add failed")

        index = self._load_index()
        index[name] = {"name": name, "path": str(path), "branch": branch, "task_id": task_id, "status": "active"}
        self._save_index(index)
        if task_id is not None:
            self.task_manager.bind_worktree(task_id, name)
        self._emit("worktree.create.after", index[name])
        return json.dumps(index[name], ensure_ascii=False, indent=2)

    def create_ue_linked(
        self,
        name: str,
        default_root: Path | None = None,
        mode: str = UE_GIT_WORKTREE_MODE,
        session_dir: Path | None = None,
    ) -> str:
        return self.create_ue_git_linked(name, default_root=default_root, mode=mode, session_dir=session_dir)

    def create_ue_git_linked(
        self,
        name: str,
        default_root: Path | None = None,
        mode: str = UE_GIT_WORKTREE_MODE,
        session_dir: Path | None = None,
    ) -> str:
        self._validate_ue_git_worktree_name(name)
        if mode == "p4-full":
            raise RuntimeError("Perforce full-control UE worktree mode is not implemented yet.")
        if mode != UE_GIT_WORKTREE_MODE:
            raise ValueError(f"Unsupported UE linked worktree mode: {mode}")
        if os.name != "nt":
            raise RuntimeError("UE linked worktrees currently support Windows junctions only.")

        project_path = _find_uproject(self.cwd)
        if project_path is None:
            raise RuntimeError("Cannot find .uproject from the current workspace.")
        project_dir = project_path.parent
        content_source = project_dir / "Content"
        if not content_source.is_dir():
            raise RuntimeError(f"UE linked worktree requires an existing Content directory: {content_source}")

        repo_root = _git_repo_root(self.cwd)
        project_relative = _relative_to(project_dir, repo_root, "UE project must be inside the Git repository.")
        project_file_relative = _relative_to(project_path, repo_root, "UE project file must be inside the Git repository.")
        content_relative = _relative_to(content_source, repo_root, "UE Content directory must be inside the Git repository.")

        _validate_git_branch(repo_root, name)
        _ensure_branch_available(repo_root, name)
        _ensure_clean_project_text(repo_root, project_file_relative, project_relative)
        _ensure_path_untracked_by_git(repo_root, content_relative, "Content")

        project_name = project_dir.name
        root = default_root or (project_dir.parent / ".uedev-worktrees")
        worktree_repo_path = (root / project_name / name).resolve()
        if worktree_repo_path.exists():
            raise RuntimeError(f"UE linked worktree target already exists: {worktree_repo_path}")
        worktree_project_path = _join_relative(worktree_repo_path, project_relative)
        content_link = worktree_project_path / "Content"
        target_agent_dir = worktree_project_path / ".agent"

        self._emit(
            "worktree.ue_linked.create.before",
            {
                "name": name,
                "branch": name,
                "source_repo_path": str(repo_root),
                "source_project_path": str(project_dir),
                "worktree_repo_path": str(worktree_repo_path),
                "worktree_project_path": str(worktree_project_path),
            },
        )

        created_git_worktree = False
        created_content_link = False
        target_session: Path | None = None
        try:
            result = _run_git(repo_root, ["worktree", "add", "-b", name, str(worktree_repo_path), "HEAD"])
            if result.returncode != 0:
                self._emit("worktree.ue_linked.create.failed", {"name": name, "stderr": result.stderr})
                raise RuntimeError(result.stderr.strip() or "git worktree add failed")
            created_git_worktree = True

            target_project_file = worktree_project_path / project_path.name
            if not target_project_file.is_file():
                raise RuntimeError(f"Git worktree did not check out the UE project file: {target_project_file}")
            if content_link.exists():
                raise RuntimeError(f"Git checkout produced Content path; Content must not be tracked by Git: {content_link}")

            _write_local_git_excludes(worktree_repo_path, project_relative)
            _create_junction(content_link, content_source)
            created_content_link = True
            target_session = _copy_agent_state(
                self.worktrees_dir.parent,
                target_agent_dir,
                session_dir,
                worktree_project_path,
            )

            item = {
                "kind": UE_LINKED_WORKTREE_KIND,
                "mode": mode,
                "name": name,
                "branch": name,
                "path": str(worktree_project_path),
                "source_repo_path": str(repo_root),
                "source_project_path": str(project_dir),
                "worktree_repo_path": str(worktree_repo_path),
                "worktree_project_path": str(worktree_project_path),
                "uproject_path": str(target_project_file),
                "content_source": str(content_source),
                "content_link": str(content_link),
                "agent_session_source": str(session_dir) if session_dir is not None else "",
                "agent_session_target": str(target_session) if target_session is not None else "",
                "created_at": time.time(),
                "status": "active",
                "warnings": [SHARED_CONTENT_WARNING],
            }
            index = self._load_index()
            index[name] = item
            self._save_index(index)
        except Exception:
            if created_content_link:
                _remove_junction(content_link, missing_ok=True)
            if created_git_worktree:
                _remove_git_worktree(repo_root, worktree_repo_path, force=True, fail_silently=True)
            raise

        self._emit("worktree.ue_linked.create.after", item)
        return "\n".join(
            [
                f"Created UE linked worktree: {name}",
                f"Branch: {item['branch']}",
                f"Worktree: {item['worktree_repo_path']}",
                f"Project: {item['worktree_project_path']}",
                f"uproject: {item['uproject_path']}",
                f"Content: linked to {item['content_source']}",
                f"Copied session: {item['agent_session_target'] or '(none)'}",
                "",
                f"Warning: {SHARED_CONTENT_WARNING}",
            ]
        )

    def list_all(self) -> str:
        index = self._load_index()
        if not index:
            return "No managed worktrees."
        lines: list[str] = []
        for name, item in sorted(index.items()):
            parts = [
                f"{name}: {item.get('status', 'unknown')}",
                f"kind={item.get('kind', 'git-worktree')}",
            ]
            if item.get("mode"):
                parts.append(f"mode={item['mode']}")
            if item.get("branch"):
                parts.append(f"branch={item['branch']}")
            if item.get("task_id") is not None:
                parts.append(f"task={item.get('task_id')}")
            parts.append(f"path={item.get('path', '')}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def run(self, name: str, command: str, timeout_seconds: int = 300) -> str:
        item = self._get(name)
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(item["path"]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return "\n".join(
            [
                f"worktree: {name}",
                f"exitCode: {result.returncode}",
                "stdout:",
                result.stdout,
                "stderr:",
                result.stderr,
            ]
        )

    def keep(self, name: str) -> str:
        index = self._load_index()
        item = index.get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")
        item["status"] = "kept"
        self._save_index(index)
        self._emit("worktree.keep", item)
        return f"Kept worktree {name}: {item['path']}"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        index = self._load_index()
        item = index.get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")

        if item.get("kind") == UE_LINKED_WORKTREE_KIND:
            if item.get("mode") != UE_GIT_WORKTREE_MODE:
                index.pop(name, None)
                self._save_index(index)
                item["status"] = "removed"
                self._emit("worktree.ue_linked.remove.after", item)
                return f"Removed UE linked worktree index entry {name}. Files remain at {item['path']}."

            self._emit("worktree.ue_linked.remove.before", item)
            content_link = Path(str(item.get("content_link") or ""))
            worktree_repo_path = Path(str(item.get("worktree_repo_path") or item.get("path") or ""))
            worktree_project_path = Path(str(item.get("worktree_project_path") or item.get("path") or ""))
            source_repo_path = Path(str(item.get("source_repo_path") or self.cwd))
            _remove_junction(content_link, missing_ok=True)
            _remove_agent_copy(worktree_project_path / ".agent")
            _remove_git_worktree(source_repo_path, worktree_repo_path, force=force, fail_silently=False)
            item["status"] = "removed"
            self._emit("worktree.ue_linked.remove.after", item)
            index.pop(name, None)
            self._save_index(index)
            return f"Removed UE linked worktree {name}. Branch {item.get('branch') or name} was not deleted."

        self._emit("worktree.remove.before", item)
        command = ["worktree", "remove"]
        if force:
            command.append("--force")
        command.append(str(item["path"]))
        result = _run_git(self.cwd, command)
        if result.returncode != 0:
            self._emit("worktree.remove.failed", {"name": name, "stderr": result.stderr})
            raise RuntimeError(result.stderr.strip() or "git worktree remove failed")

        task_id = item.get("task_id")
        if complete_task and task_id is not None:
            self.task_manager.update(int(task_id), status="completed", worktree="")
        elif task_id is not None:
            self.task_manager.unbind_worktree(int(task_id))

        item["status"] = "removed"
        self._emit("worktree.remove.after", item)
        index.pop(name, None)
        self._save_index(index)
        return f"Removed worktree {name}."

    def _get(self, name: str) -> dict[str, object]:
        item = self._load_index().get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")
        return item

    def _load_index(self) -> dict[str, dict[str, object]]:
        if not self.index_path.exists():
            return {}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _save_index(self, index: dict[str, dict[str, object]]) -> None:
        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        record = {"event": event, "ts": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validate_name(self, name: str) -> None:
        if not name or any(char in name for char in "\\/:*?\"<>| "):
            raise ValueError("worktree name must be non-empty and path-safe")

    def _validate_ue_git_worktree_name(self, name: str) -> None:
        self._validate_name(name)
        if not BRANCH_SAFE_RE.match(name):
            raise ValueError("worktree name must contain only letters, numbers, dot, underscore, and hyphen")
        if name.startswith("-") or name.upper() == "HEAD":
            raise ValueError("worktree name must also be a valid Git branch name")


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_repo_root(cwd: Path) -> Path:
    result = _run_git(cwd, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Current workspace is not inside a Git repository.")
    return Path(result.stdout.strip()).resolve()


def _validate_git_branch(repo_root: Path, name: str) -> None:
    result = _run_git(repo_root, ["check-ref-format", "--branch", name])
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"invalid Git branch name: {name}")


def _ensure_branch_available(repo_root: Path, name: str) -> None:
    result = _run_git(repo_root, ["show-ref", "--verify", "--quiet", f"refs/heads/{name}"])
    if result.returncode == 0:
        raise RuntimeError(f"Git branch already exists: {name}")


def _ensure_clean_project_text(repo_root: Path, project_file_relative: Path, project_relative: Path) -> None:
    pathspecs = [_git_pathspec(project_file_relative)]
    for name in UE_TEXT_PATHS:
        pathspecs.append(_git_pathspec(_join_relative(project_relative, Path(name))))
    result = _run_git(repo_root, ["status", "--porcelain", "--", *pathspecs])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    dirty = result.stdout.strip()
    if dirty:
        raise RuntimeError(
            "Git text project paths have uncommitted changes; commit or stash them before /worktree:\n"
            f"{dirty}"
        )


def _ensure_path_untracked_by_git(repo_root: Path, relative_path: Path, label: str) -> None:
    result = _run_git(repo_root, ["ls-files", "--", _git_pathspec(relative_path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    if result.stdout.strip():
        raise RuntimeError(f"{label} is tracked by Git; UE linked worktrees require it to be provided outside Git.")


def _remove_git_worktree(repo_root: Path, worktree_path: Path, force: bool, fail_silently: bool) -> None:
    command = ["worktree", "remove"]
    if force:
        command.append("--force")
    command.append(str(worktree_path))
    result = _run_git(repo_root, command)
    if result.returncode != 0 and not fail_silently:
        raise RuntimeError(result.stderr.strip() or "git worktree remove failed")


def _write_local_git_excludes(worktree_repo_path: Path, project_relative: Path) -> None:
    result = _run_git(worktree_repo_path, ["rev-parse", "--git-path", "info/exclude"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse --git-path info/exclude failed")
    raw_path = result.stdout.strip()
    if not raw_path:
        raise RuntimeError("git rev-parse --git-path info/exclude returned an empty path")
    exclude_path = Path(raw_path)
    if not exclude_path.is_absolute():
        exclude_path = worktree_repo_path / exclude_path
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    existing_lines = {line.strip() for line in existing.splitlines()}
    lines = [_git_exclude_pattern(project_relative, pattern) for pattern in UE_LOCAL_EXCLUDES]
    missing = [line for line in lines if line not in existing_lines]
    if not missing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    exclude_path.write_text(existing + prefix + "\n".join(missing) + "\n", encoding="utf-8")


def _copy_agent_state(
    source_agent_dir: Path,
    target_agent_dir: Path,
    session_dir: Path | None,
    target_project_path: Path,
) -> Path | None:
    if target_agent_dir.exists() and any(target_agent_dir.iterdir()):
        raise RuntimeError(f"Target .agent directory already exists and is not empty: {target_agent_dir}")
    target_agent_dir.mkdir(parents=True, exist_ok=True)

    source_config = source_agent_dir / "config.json"
    if source_config.is_file():
        shutil.copy2(source_config, target_agent_dir / "config.json")

    if session_dir is None:
        return None

    source_session = session_dir.resolve()
    source_agent_root = source_agent_dir.resolve()
    try:
        session_relative = source_session.relative_to(source_agent_root)
    except ValueError as error:
        raise RuntimeError(f"Current session is outside .agent state: {source_session}") from error
    if not session_relative.parts or session_relative.parts[0] != "sessions":
        raise RuntimeError(f"Current session is not under .agent/sessions: {source_session}")

    target_session = target_agent_dir / session_relative
    if target_session.exists():
        raise RuntimeError(f"Target session already exists: {target_session}")
    shutil.copytree(source_session, target_session)
    _rewrite_session_context(target_session, target_project_path)
    return target_session


def _rewrite_session_context(session_dir: Path, target_project_path: Path) -> None:
    system_prompt = build_system_prompt(
        target_project_path,
        shell_name(),
        SkillLoader(target_project_path / "skills").descriptions(),
    )
    for filename in ("messages.jsonl", "transcript.jsonl"):
        path = session_dir / filename
        if path.exists() and path.read_text(encoding="utf-8").strip():
            _rewrite_history_file(path, target_project_path, system_prompt)


def _rewrite_history_file(path: Path, target_project_path: Path, system_prompt: str) -> None:
    messages = load_history_file(path)
    updated: list[ChatMessage] = []
    replaced_system = False
    replaced_cwd = False
    for message in messages:
        if message.role == "system" and not replaced_system:
            updated.append(_with_content(message, system_prompt))
            replaced_system = True
        elif message.role == "user" and message.content.startswith("Working directory:") and not replaced_cwd:
            updated.append(_with_content(message, f"Working directory: {target_project_path}\nShell: {shell_name()}"))
            replaced_cwd = True
        else:
            updated.append(message)
    if not replaced_system:
        updated.insert(0, ChatMessage(role="system", content=system_prompt))
    if not replaced_cwd:
        insert_at = 1 if updated and updated[0].role == "system" else 0
        updated.insert(insert_at, ChatMessage(role="user", content=f"Working directory: {target_project_path}\nShell: {shell_name()}"))
    write_history_messages(path, updated)


def _with_content(message: ChatMessage, content: str) -> ChatMessage:
    return ChatMessage(
        role=message.role,
        content=content,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        name=message.name,
        reasoning_content=message.reasoning_content,
    )


def _remove_agent_copy(agent_dir: Path) -> None:
    if agent_dir.exists():
        shutil.rmtree(agent_dir)


def _find_uproject(cwd: Path) -> Path | None:
    root = cwd.resolve()
    for parent in [root, *root.parents]:
        projects = sorted(parent.glob("*.uproject"))
        if projects:
            return projects[0].resolve()
    return None


def _create_junction(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        raise RuntimeError(f"Content link path already exists: {link_path}")
    command = ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "mklink /J failed"
        raise RuntimeError(detail)


def _remove_junction(link_path: Path, missing_ok: bool = False) -> None:
    if not link_path.exists():
        if missing_ok:
            return
        raise RuntimeError(f"Content link path does not exist: {link_path}")
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "rmdir", str(link_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "rmdir failed"
            raise RuntimeError(detail)
        return
    if link_path.is_symlink():
        link_path.unlink()
        return
    raise RuntimeError("Content link removal is supported only for Windows junctions.")


def _relative_to(path: Path, root: Path, message: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(message) from error


def _join_relative(root: Path, relative: Path) -> Path:
    return root if not relative.parts else root / relative


def _git_pathspec(path: Path) -> str:
    return "." if not path.parts else path.as_posix()


def _git_exclude_pattern(project_relative: Path, pattern: str) -> str:
    prefix = "" if not project_relative.parts else project_relative.as_posix().rstrip("/") + "/"
    return prefix + pattern
