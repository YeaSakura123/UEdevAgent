from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .tasks import TaskManager


VALID_MESSAGE_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_request",
    "plan_response",
}


class MessageBus:
    """s09 JSONL inbox：append-only 写入，read 时 drain。"""

    # 内部函数：初始化当前类实例，准备 队友状态、消息 inbox 和协作协议 所需状态。
    def __init__(self, team_dir: Path):
        self.team_dir = team_dir
        self.inbox_dir = team_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    # 外部函数：写入团队消息到目标 inbox，负责 队友状态、消息 inbox 和协作协议。
    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict[str, object] | None = None) -> str:
        if msg_type not in VALID_MESSAGE_TYPES:
            raise ValueError(f"invalid message type: {msg_type}")
        message: dict[str, object] = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            message.update(extra)
        with (self.inbox_dir / f"{to}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} from {sender} to {to}"

    # 外部函数：读取并清空指定成员 inbox，负责 队友状态、消息 inbox 和协作协议。
    def read_inbox(self, name: str) -> list[dict[str, object]]:
        path = self.inbox_dir / f"{name}.jsonl"
        if not path.exists():
            return []

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        path.write_text("", encoding="utf-8")
        return [json.loads(line) for line in lines]


class TeamManager:
    """s09-s11 团队状态：成员、消息、协议请求、自主任务认领。"""

    # 内部函数：初始化当前类实例，准备 队友状态、消息 inbox 和协作协议 所需状态。
    def __init__(self, team_dir: Path, task_manager: TaskManager, bus: MessageBus):
        self.team_dir = team_dir
        self.config_path = team_dir / "config.json"
        self.requests_path = team_dir / "requests.json"
        self.task_manager = task_manager
        self.bus = bus
        self.team_dir.mkdir(parents=True, exist_ok=True)

    # 外部函数：注册或更新队友信息，负责 队友状态、消息 inbox 和协作协议。
    def spawn(self, name: str, role: str, prompt: str = "") -> str:
        config = self._load_config()
        members = config["members"]
        existing = next((item for item in members if item["name"] == name), None)
        if existing:
            existing.update({"role": role, "status": "idle", "prompt": prompt})
        else:
            members.append({"name": name, "role": role, "status": "idle", "prompt": prompt})
        self._save_config(config)
        return f"Registered teammate {name} ({role})."

    # 外部函数：实现 list_teammates 和 CLI team 展示，列出队友状态。
    def list_all(self) -> str:
        config = self._load_config()
        if not config["members"]:
            return "No teammates."
        lines = [f"Team: {config['team_name']}"]
        for member in config["members"]:
            lines.append(f"- {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    # 内部函数：读取当前团队成员名称列表，供广播和协议操作使用。
    def names(self) -> list[str]:
        return [member["name"] for member in self._load_config()["members"]]

    # 外部函数：向团队成员广播消息，负责 队友状态、消息 inbox 和协作协议。
    def broadcast(self, sender: str, content: str) -> str:
        count = 0
        for name in self.names():
            if name != sender:
                self.bus.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates."

    # 外部函数：创建并发送关闭请求，负责 队友状态、消息 inbox 和协作协议。
    def shutdown_request(self, teammate: str) -> str:
        request_id = str(uuid.uuid4())[:8]
        requests = self._load_requests()
        requests[request_id] = {"type": "shutdown", "target": teammate, "status": "pending"}
        self._save_requests(requests)
        self.bus.send("lead", teammate, "Please shut down gracefully.", "shutdown_request", {"request_id": request_id})
        return f"Shutdown request {request_id} sent to {teammate}."

    # 外部函数：记录关闭请求审批结果，负责 队友状态、消息 inbox 和协作协议。
    def shutdown_response(self, request_id: str, approve: bool, reason: str = "") -> str:
        requests = self._load_requests()
        request = requests.get(request_id)
        if request is None:
            raise ValueError(f"unknown shutdown request: {request_id}")
        request["status"] = "approved" if approve else "rejected"
        request["reason"] = reason
        self._save_requests(requests)
        return f"Shutdown request {request_id}: {request['status']}"

    # 外部函数：提交队友计划给 lead 审核，负责 队友状态、消息 inbox 和协作协议。
    def plan_submit(self, teammate: str, plan: str) -> str:
        request_id = str(uuid.uuid4())[:8]
        requests = self._load_requests()
        requests[request_id] = {"type": "plan", "from": teammate, "plan": plan, "status": "pending"}
        self._save_requests(requests)
        self.bus.send(teammate, "lead", plan, "plan_request", {"request_id": request_id})
        return f"Plan request {request_id} submitted by {teammate}."

    # 外部函数：审批计划并反馈提交者，负责 队友状态、消息 inbox 和协作协议。
    def plan_review(self, request_id: str, approve: bool, feedback: str = "") -> str:
        requests = self._load_requests()
        request = requests.get(request_id)
        if request is None:
            raise ValueError(f"unknown plan request: {request_id}")
        request["status"] = "approved" if approve else "rejected"
        request["feedback"] = feedback
        self._save_requests(requests)
        target = str(request.get("from") or "lead")
        self.bus.send("lead", target, feedback, "plan_response", {"request_id": request_id, "approve": approve})
        return f"Plan request {request_id}: {request['status']}"

    # 外部函数：让队友认领第一个 ready task，负责 队友状态、消息 inbox 和协作协议。
    def claim_ready_task(self, teammate: str) -> str:
        ready = self.task_manager.ready_tasks()
        if not ready:
            return "No ready tasks to claim."
        task_id = int(ready[0]["id"])
        self.task_manager.claim(task_id, teammate)
        self._set_member_status(teammate, "working")
        return f"{teammate} claimed task #{task_id}: {ready[0]['subject']}"

    # 外部函数：把队友状态设置为空闲，负责 队友状态、消息 inbox 和协作协议。
    def idle(self, teammate: str) -> str:
        self._set_member_status(teammate, "idle")
        return f"{teammate} is idle."

    # 内部函数：处理 _set_member_status 辅助逻辑，支撑 队友状态、消息 inbox 和协作协议。
    def _set_member_status(self, teammate: str, status: str) -> None:
        config = self._load_config()
        for member in config["members"]:
            if member["name"] == teammate:
                member["status"] = status
                break
        self._save_config(config)

    # 内部函数：处理 _load_config 辅助逻辑，支撑 队友状态、消息 inbox 和协作协议。
    def _load_config(self) -> dict[str, object]:
        if self.config_path.exists():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "members" in data:
                return data
        return {"team_name": "ue-agent-team", "members": []}

    # 内部函数：处理 _save_config 辅助逻辑，支撑 队友状态、消息 inbox 和协作协议。
    def _save_config(self, config: dict[str, object]) -> None:
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 内部函数：处理 _load_requests 辅助逻辑，支撑 队友状态、消息 inbox 和协作协议。
    def _load_requests(self) -> dict[str, object]:
        if not self.requests_path.exists():
            return {}
        data = json.loads(self.requests_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    # 内部函数：处理 _save_requests 辅助逻辑，支撑 队友状态、消息 inbox 和协作协议。
    def _save_requests(self, requests: dict[str, object]) -> None:
        self.requests_path.write_text(json.dumps(requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
