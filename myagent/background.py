from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue


@dataclass
class BackgroundTask:
    id: str
    command: str
    status: str
    result: str = ""


class BackgroundManager:
    """把慢命令丢到后台，主 agent loop 在下一轮收到完成通知。"""

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.tasks: dict[str, BackgroundTask] = {}
        self.notifications: Queue[BackgroundTask] = Queue()

    def run(self, command: str, timeout_seconds: int = 300) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(id=task_id, command=command, status="running")
        self.tasks[task_id] = task
        thread = threading.Thread(target=self._execute, args=(task, timeout_seconds), daemon=True)
        thread.start()
        return f"Background task {task_id} started: {command}"

    def _execute(self, task: BackgroundTask, timeout_seconds: int) -> None:
        try:
            process = subprocess.run(
                task.command,
                shell=True,
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            task.status = "completed" if process.returncode == 0 else "failed"
            task.result = (process.stdout + process.stderr).strip() or "(no output)"
        except subprocess.TimeoutExpired:
            task.status = "timeout"
            task.result = f"Command timed out after {timeout_seconds}s"
        except Exception as error:
            task.status = "error"
            task.result = str(error)
        self.notifications.put(task)

    def check(self, task_id: str | None = None) -> str:
        if task_id:
            task = self.tasks.get(task_id)
            if task is None:
                return f"Unknown background task: {task_id}"
            return f"{task.id}: {task.status}\n{task.result or '(still running)'}"

        if not self.tasks:
            return "No background tasks."
        return "\n".join(f"{task.id}: {task.status} - {task.command}" for task in self.tasks.values())

    def drain(self) -> list[BackgroundTask]:
        drained: list[BackgroundTask] = []
        while not self.notifications.empty():
            drained.append(self.notifications.get_nowait())
        return drained
