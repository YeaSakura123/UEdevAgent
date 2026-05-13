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

    # 内部函数：初始化当前类实例，准备 后台任务启动、状态查询和完成通知 所需状态。
    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.tasks: dict[str, BackgroundTask] = {}
        self.notifications: Queue[BackgroundTask] = Queue()

    # 外部函数：提供 background_run 工具能力，启动后台 shell 命令并返回任务 ID。
    def run(self, command: str, timeout_seconds: int = 300) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(id=task_id, command=command, status="running")
        self.tasks[task_id] = task
        thread = threading.Thread(target=self._execute, args=(task, timeout_seconds), daemon=True)
        thread.start()
        return f"Background task {task_id} started: {command}"

    # 内部函数：处理 _execute 辅助逻辑，支撑 后台任务启动、状态查询和完成通知。
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

    # 外部函数：查询后台任务状态，负责 后台任务启动、状态查询和完成通知。
    def check(self, task_id: str | None = None) -> str:
        if task_id:
            task = self.tasks.get(task_id)
            if task is None:
                return f"Unknown background task: {task_id}"
            return f"{task.id}: {task.status}\n{task.result or '(still running)'}"

        if not self.tasks:
            return "No background tasks."
        return "\n".join(f"{task.id}: {task.status} - {task.command}" for task in self.tasks.values())

    # 内部函数：取出已完成后台任务通知，供 agent loop 注入运行时观察。
    def drain(self) -> list[BackgroundTask]:
        drained: list[BackgroundTask] = []
        while not self.notifications.empty():
            drained.append(self.notifications.get_nowait())
        return drained
