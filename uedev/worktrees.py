from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .tasks import TaskManager


class WorktreeManager:
    """s12：任务图负责目标，git worktree 负责隔离执行目录。"""

    # 内部函数：初始化当前类实例，准备 git worktree 创建、运行、保留、删除和索引维护 所需状态。
    def __init__(self, cwd: Path, worktrees_dir: Path, task_manager: TaskManager):
        self.cwd = cwd
        self.worktrees_dir = worktrees_dir
        self.index_path = worktrees_dir / "index.json"
        self.events_path = worktrees_dir / "events.jsonl"
        self.task_manager = task_manager
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    # 外部函数：实现 worktree_create 工具能力，创建隔离 worktree 并可绑定任务。
    def create(self, name: str, task_id: int | None = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        path = self.worktrees_dir / name
        branch = f"wt/{name}"
        self._emit("worktree.create.before", {"name": name, "task_id": task_id, "path": str(path)})
        if not path.exists():
            command = ["git", "worktree", "add", "-b", branch, str(path), base_ref]
            result = subprocess.run(command, cwd=str(self.cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
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

    # 外部函数：实现 worktree_list 和 CLI worktrees 展示，列出托管 worktree。
    def list_all(self) -> str:
        index = self._load_index()
        if not index:
            return "No managed worktrees."
        return "\n".join(
            f"{name}: {item['status']} task={item.get('task_id')} path={item['path']}"
            for name, item in sorted(index.items())
        )

    # 外部函数：实现 worktree_run 工具能力，在指定 worktree 中执行命令。
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

    # 外部函数：标记 worktree 为保留，负责 git worktree 创建、运行、保留、删除和索引维护。
    def keep(self, name: str) -> str:
        index = self._load_index()
        item = index.get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")
        item["status"] = "kept"
        self._save_index(index)
        self._emit("worktree.keep", item)
        return f"Kept worktree {name}: {item['path']}"

    # 外部函数：删除托管 worktree 并维护任务绑定，负责 git worktree 创建、运行、保留、删除和索引维护。
    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        index = self._load_index()
        item = index.get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")

        self._emit("worktree.remove.before", item)
        command = ["git", "worktree", "remove"]
        if force:
            command.append("--force")
        command.append(str(item["path"]))
        result = subprocess.run(command, cwd=str(self.cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
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

    # 内部函数：处理 _get 辅助逻辑，支撑 git worktree 创建、运行、保留、删除和索引维护。
    def _get(self, name: str) -> dict[str, object]:
        item = self._load_index().get(name)
        if item is None:
            raise ValueError(f"unknown worktree: {name}")
        return item

    # 内部函数：处理 _load_index 辅助逻辑，支撑 git worktree 创建、运行、保留、删除和索引维护。
    def _load_index(self) -> dict[str, dict[str, object]]:
        if not self.index_path.exists():
            return {}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    # 内部函数：处理 _save_index 辅助逻辑，支撑 git worktree 创建、运行、保留、删除和索引维护。
    def _save_index(self, index: dict[str, dict[str, object]]) -> None:
        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 内部函数：处理 _emit 辅助逻辑，支撑 git worktree 创建、运行、保留、删除和索引维护。
    def _emit(self, event: str, payload: dict[str, object]) -> None:
        record = {"event": event, "ts": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 内部函数：处理 _validate_name 辅助逻辑，支撑 git worktree 创建、运行、保留、删除和索引维护。
    def _validate_name(self, name: str) -> None:
        if not name or any(char in name for char in "\\/:*?\"<>| "):
            raise ValueError("worktree name must be non-empty and path-safe")
