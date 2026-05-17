from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm import ChatMessage, ToolCall


HISTORY_DIR = "history"


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryEntry:
    path: Path
    kind: str
    modified_at: float
    message_count: int
    preview: str

    @property
    def label(self) -> str:
        timestamp = datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{self.kind}] {self.message_count} messages - {self.preview}"


class HistoryRecorder:
    def __init__(self, agent_dir: Path, initial_messages: list[ChatMessage]):
        self.agent_dir = agent_dir
        self.initial_messages = list(initial_messages)
        self.path: Path | None = None

    def reset(self, initial_messages: list[ChatMessage]) -> None:
        self.initial_messages = list(initial_messages)
        self.path = None

    def append(self, message: ChatMessage) -> None:
        path = self._ensure_path()
        append_history_message(path, message)

    def _ensure_path(self) -> Path:
        if self.path is None:
            self.path = create_session_history_path(self.agent_dir)
            write_history_messages(self.path, self.initial_messages)
        return self.path


def create_session_history_path(agent_dir: Path) -> Path:
    history_dir = agent_dir / HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"session_{time.time_ns()}.jsonl"


def append_history_message(path: Path, message: ChatMessage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_message_to_json(message) + "\n")


def write_history_messages(path: Path, messages: list[ChatMessage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(_message_to_json(message) + "\n")


def load_history_file(path: Path) -> list[ChatMessage]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HistoryError(f"Failed to read history file: {path}: {error}") from error

    messages: list[ChatMessage] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise HistoryError(f"Invalid JSON in {path} at line {line_number}: {error}") from error
        messages.append(_message_from_dict(raw, path, line_number))

    if not messages:
        raise HistoryError(f"History file is empty: {path}")
    return messages


def list_history_entries(agent_dir: Path) -> list[HistoryEntry]:
    candidates: list[tuple[Path, str]] = []
    candidates.extend((path, "session") for path in (agent_dir / HISTORY_DIR).glob("session_*.jsonl"))

    entries: list[HistoryEntry] = []
    for path, kind in candidates:
        try:
            messages = load_history_file(path)
            stat = path.stat()
        except (HistoryError, OSError):
            continue
        entries.append(
            HistoryEntry(
                path=path,
                kind=kind,
                modified_at=stat.st_mtime,
                message_count=len(messages),
                preview=_history_preview(messages),
            )
        )
    return sorted(entries, key=lambda entry: entry.modified_at, reverse=True)


def ensure_system_prompt(messages: list[ChatMessage], system_prompt: str) -> list[ChatMessage]:
    if any(message.role == "system" for message in messages):
        return list(messages)
    return [ChatMessage(role="system", content=system_prompt), *messages]


def _message_to_json(message: ChatMessage) -> str:
    return json.dumps(asdict(message), ensure_ascii=False)


def _message_from_dict(raw: Any, path: Path, line_number: int) -> ChatMessage:
    if not isinstance(raw, dict):
        raise HistoryError(f"History line must be an object in {path} at line {line_number}")
    role = str(raw.get("role") or "")
    content = str(raw.get("content") or "")
    if not role:
        raise HistoryError(f"History message is missing role in {path} at line {line_number}")
    tool_calls = [_tool_call_from_dict(item, path, line_number) for item in raw.get("tool_calls") or []]
    return ChatMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=_optional_str(raw.get("tool_call_id")),
        name=_optional_str(raw.get("name")),
    )


def _tool_call_from_dict(raw: Any, path: Path, line_number: int) -> ToolCall:
    if not isinstance(raw, dict):
        raise HistoryError(f"tool_calls entries must be objects in {path} at line {line_number}")
    return ToolCall(
        id=str(raw.get("id") or ""),
        name=str(raw.get("name") or ""),
        arguments=raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _history_preview(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role not in {"user", "assistant"}:
            continue
        content = " ".join(message.content.split())
        if not content or content.startswith("Working directory:"):
            continue
        return content[:96] + ("..." if len(content) > 96 else "")
    return "(no preview)"
