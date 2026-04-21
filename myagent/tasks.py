from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass(frozen=True)
class TodoItem:
    id: str
    text: str
    status: TodoStatus


class TodoManager:
    """把 Claude Code 风格的 TodoWrite 做成磁盘状态，方便断点恢复。"""

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

            if not item_id:
                raise ValueError("todo item id cannot be empty")
            if not text:
                raise ValueError(f"todo item {item_id} text cannot be empty")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"todo item {item_id} has invalid status: {status}")

            if status == "in_progress":
                in_progress_count += 1
            items.append(TodoItem(id=item_id, text=text, status=status))  # type: ignore[arg-type]

        if in_progress_count > 1:
            raise ValueError("only one todo item can be in_progress")

        payload = [{"id": item.id, "text": item.text, "status": item.status} for item in items]
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
            text = str(raw.get("text", "")).strip()
            status = str(raw.get("status", "pending")).strip()
            if item_id and text and status in {"pending", "in_progress", "completed"}:
                items.append(TodoItem(id=item_id, text=text, status=status))  # type: ignore[arg-type]
        return items

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
        return "\n".join(f"{icons[item.status]} {item.id}. {item.text}" for item in items)
