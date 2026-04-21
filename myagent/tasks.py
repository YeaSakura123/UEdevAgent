from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TodoStatus = Literal["pending", "in_progress", "completed"]
TaskStatus = Literal["pending", "in_progress", "completed", "blocked", "deleted"]


@dataclass(frozen=True)
class TodoItem:
    id: str
    text: str
    status: TodoStatus
    active_form: str = ""


class TodoManager:
    """s03 TodoWrite：短期计划在磁盘落地，便于 chat/restart 后继续查看。"""

    def __init__(self, agent_dir: Path):
        self.agent_dir = agent_dir
        self.path = agent_dir / "todos.json"
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    def update(self, raw_items: list[dict[str, object]]) -> str:
        items: list[TodoItem] = []
        in_progress_count = 0

        for index, raw in enumerate(raw_items, start=1):
            item_id = str(raw.get("id") or index).strip()
            text = str(raw.get("text") or raw.get("content") or "").strip()
            status = str(raw.get("status") or "pending").strip()
            active_form = str(raw.get("activeForm") or raw.get("active_form") or "").strip()

            if not item_id:
                raise ValueError("todo item id cannot be empty")
            if not text:
                raise ValueError(f"todo item {item_id} text cannot be empty")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"todo item {item_id} has invalid status: {status}")
            if status == "in_progress":
                in_progress_count += 1

            items.append(TodoItem(id=item_id, text=text, status=status, active_form=active_form))  # type: ignore[arg-type]

        if in_progress_count > 1:
            raise ValueError("only one todo item can be in_progress")

        payload = [
            {"id": item.id, "text": item.text, "status": item.status, "activeForm": item.active_form}
            for item in items
        ]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.render(items)

    def load(self) -> list[TodoItem]:
        if not self.path.exists():
            return []

        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []

        items: list[TodoItem] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("id", "")).strip()
            text = str(raw.get("text") or raw.get("content") or "").strip()
            status = str(raw.get("status", "pending")).strip()
            active_form = str(raw.get("activeForm") or raw.get("active_form") or "").strip()
            if item_id and text and status in {"pending", "in_progress", "completed"}:
                items.append(TodoItem(id=item_id, text=text, status=status, active_form=active_form))  # type: ignore[arg-type]
        return items

    def has_open_items(self) -> bool:
        return any(item.status != "completed" for item in self.load())

    def render_current(self) -> str:
        return self.render(self.load())

    def render(self, items: list[TodoItem]) -> str:
        if not items:
            return "No todos."

        icons = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        lines = []
        for item in items:
            suffix = f" <- {item.active_form}" if item.status == "in_progress" and item.active_form else ""
            lines.append(f"{icons[item.status]} {item.id}. {item.text}{suffix}")
        done = sum(1 for item in items if item.status == "completed")
        lines.append(f"\n({done}/{len(items)} completed)")
        return "\n".join(lines)


class TaskManager:
    """s07 持久化任务图：一个 task 一个 JSON 文件，支持依赖和 owner。"""

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[int] | None = None,
        owner: str | None = None,
    ) -> str:
        task = {
            "id": self._next_id(),
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": owner,
            "blockedBy": sorted(set(blocked_by or [])),
            "worktree": "",
        }
        self._save(task)
        return json.dumps(task, ensure_ascii=False, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), ensure_ascii=False, indent=2)

    def update(
        self,
        task_id: int,
        status: str | None = None,
        owner: str | None = None,
        add_blocked_by: list[int] | None = None,
        remove_blocked_by: list[int] | None = None,
        worktree: str | None = None,
    ) -> str:
        task = self._load(task_id)
        if status:
            if status == "deleted":
                self._path(task_id).unlink(missing_ok=True)
                return f"Task {task_id} deleted."
            if status not in {"pending", "in_progress", "completed", "blocked"}:
                raise ValueError(f"invalid task status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if owner is not None:
            task["owner"] = owner or None
        if worktree is not None:
            task["worktree"] = worktree
        if add_blocked_by:
            task["blockedBy"] = sorted(set(task.get("blockedBy", []) + add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [item for item in task.get("blockedBy", []) if item not in remove_blocked_by]

        self._save(task)
        return json.dumps(task, ensure_ascii=False, indent=2)

    def list_all(self) -> str:
        tasks = [self._read_file(path) for path in sorted(self.tasks_dir.glob("task_*.json"))]
        if not tasks:
            return "No tasks."

        lines = []
        for task in tasks:
            icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "blocked": "[-]"}.get(
                task.get("status"), "[?]"
            )
            owner = f" @{task['owner']}" if task.get("owner") else ""
            blocked = f" blockedBy={task['blockedBy']}" if task.get("blockedBy") else ""
            worktree = f" wt={task['worktree']}" if task.get("worktree") else ""
            lines.append(f"{icon} #{task['id']}: {task['subject']}{owner}{blocked}{worktree}")
        return "\n".join(lines)

    def claim(self, task_id: int, owner: str) -> str:
        return self.update(task_id, status="in_progress", owner=owner)

    def ready_tasks(self) -> list[dict[str, object]]:
        tasks = [self._read_file(path) for path in sorted(self.tasks_dir.glob("task_*.json"))]
        return [
            task
            for task in tasks
            if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy")
        ]

    def bind_worktree(self, task_id: int, worktree: str) -> None:
        self.update(task_id, status="in_progress", worktree=worktree)

    def unbind_worktree(self, task_id: int) -> None:
        self.update(task_id, worktree="")

    def _next_id(self) -> int:
        ids = []
        for path in self.tasks_dir.glob("task_*.json"):
            try:
                ids.append(int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(ids, default=0) + 1

    def _path(self, task_id: int) -> Path:
        return self.tasks_dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict[str, object]:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return self._read_file(path)

    def _read_file(self, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid task file: {path}")
        return data

    def _save(self, task: dict[str, object]) -> None:
        task_id = int(task["id"])
        self._path(task_id).write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _clear_dependency(self, completed_id: int) -> None:
        for path in self.tasks_dir.glob("task_*.json"):
            task = self._read_file(path)
            blocked_by = task.get("blockedBy", [])
            if isinstance(blocked_by, list) and completed_id in blocked_by:
                task["blockedBy"] = [item for item in blocked_by if item != completed_id]
                self._save(task)
