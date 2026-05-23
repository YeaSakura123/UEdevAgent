from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..llm.client import ChatMessage, ToolCall
from ..ui.events import AgentEvent


SESSIONS_DIR = "sessions"
MESSAGES_FILE = "messages.jsonl"
DISPLAY_FILE = "display.jsonl"
METADATA_FILE = "metadata.json"
TRANSCRIPT_FILE = "transcript.jsonl"
SCHEMA_VERSION = 1


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryEntry:
    path: Path
    kind: str
    modified_at: float
    message_count: int
    preview: str
    display_path: Path | None = None
    session_dir: Path | None = None
    transcript_path: Path | None = None

    @property
    def label(self) -> str:
        timestamp = datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{self.kind}] {self.message_count} messages - {self.preview}"


class HistoryRecorder:
    def __init__(
        self,
        agent_dir: Path,
        initial_messages: list[ChatMessage],
        initial_display_records: list[dict[str, Any]] | None = None,
    ):
        self.agent_dir = agent_dir
        self.initial_messages = list(initial_messages)
        self.initial_display_records = list(initial_display_records or [])
        self.session_dir: Path | None = None
        self.path: Path | None = None
        self.display_path: Path | None = None
        self.transcript_path: Path | None = None
        self._display_seeded = False

    def reset(
        self,
        initial_messages: list[ChatMessage],
        initial_display_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.initial_messages = list(initial_messages)
        self.initial_display_records = list(initial_display_records or [])
        self.session_dir = None
        self.path = None
        self.display_path = None
        self.transcript_path = None
        self._display_seeded = False

    def resume(self, entry: HistoryEntry, initial_messages: list[ChatMessage]) -> None:
        self.initial_messages = list(initial_messages)
        self.initial_display_records = []
        self.session_dir = entry.session_dir
        self.path = entry.path
        self.display_path = entry.display_path or (entry.session_dir / DISPLAY_FILE if entry.session_dir is not None else None)
        self.transcript_path = entry.transcript_path or (entry.session_dir / TRANSCRIPT_FILE if entry.session_dir is not None else None)
        self._display_seeded = True

    def append(self, message: ChatMessage) -> None:
        path = self._ensure_path()
        append_history_message(path, message)
        self._write_metadata()

    def record_turn_start(self, turn_id: str, message: str) -> None:
        path = self._ensure_display_path()
        append_display_turn_start(path, turn_id, message)
        self._write_metadata()

    def record_event(self, event: AgentEvent) -> None:
        path = self._ensure_display_path()
        append_display_event(path, event)
        self._write_metadata()

    def _ensure_path(self) -> Path:
        if self.path is None:
            self.session_dir = create_session_dir(self.agent_dir)
            self.path = self.session_dir / MESSAGES_FILE
            self.display_path = self.session_dir / DISPLAY_FILE
            self.transcript_path = self.session_dir / TRANSCRIPT_FILE
            write_history_messages(self.path, self.initial_messages)
            self._write_metadata()
        return self.path

    def _ensure_display_path(self) -> Path:
        if self.display_path is None:
            self._ensure_path()
        if self.display_path is None:
            raise HistoryError("Display history path was not initialized.")
        if not self._display_seeded:
            if self.initial_display_records:
                write_display_records(self.display_path, self.initial_display_records)
            self._display_seeded = True
        return self.display_path

    def ensure_session(self) -> Path:
        self._ensure_path()
        if self.session_dir is None:
            raise HistoryError("Session directory was not initialized.")
        return self.session_dir

    def ensure_transcript_path(self) -> Path:
        self.ensure_session()
        if self.transcript_path is None:
            raise HistoryError("Transcript path was not initialized.")
        self._write_metadata()
        return self.transcript_path

    def _write_metadata(self) -> None:
        if self.session_dir is None or self.path is None or self.display_path is None or self.transcript_path is None:
            return
        write_session_metadata(self.session_dir, self.path, self.display_path, self.transcript_path)


def create_session_dir(agent_dir: Path) -> Path:
    while True:
        created_at = time.time()
        local_time = time.localtime(created_at)
        session_id = f"session_{time.time_ns()}_{uuid4().hex[:8]}"
        session_dir = (
            agent_dir
            / SESSIONS_DIR
            / f"{local_time.tm_year:04d}"
            / f"{local_time.tm_mon:02d}"
            / f"{local_time.tm_mday:02d}"
            / session_id
        )
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return session_dir


def write_session_metadata(session_dir: Path, messages_path: Path, display_path: Path, transcript_path: Path) -> None:
    metadata_path = session_dir / METADATA_FILE
    created_at = time.time()
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                created_at = float(raw.get("created_at") or created_at)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_dir.name,
        "created_at": created_at,
        "updated_at": time.time(),
        "messages_path": messages_path.name,
        "display_path": display_path.name,
        "transcript_path": transcript_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_standalone_session_transcript_path(agent_dir: Path, messages: list[ChatMessage]) -> Path:
    session_dir = create_session_dir(agent_dir)
    messages_path = session_dir / MESSAGES_FILE
    display_path = session_dir / DISPLAY_FILE
    transcript_path = session_dir / TRANSCRIPT_FILE
    write_history_messages(messages_path, messages)
    write_session_metadata(session_dir, messages_path, display_path, transcript_path)
    return transcript_path


def append_history_message(path: Path, message: ChatMessage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_message_to_json(message) + "\n")


def write_history_messages(path: Path, messages: list[ChatMessage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(_message_to_json(message) + "\n")


def append_display_turn_start(path: Path, turn_id: str, message: str) -> None:
    append_display_record(path, {"type": "turn_start", "turn_id": turn_id, "message": message})


def append_display_event(path: Path, event: AgentEvent) -> None:
    append_display_record(path, {"type": "event", "event": asdict(event)})


def append_display_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_display_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def load_display_history(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HistoryError(f"Failed to read display history file: {path}: {error}") from error

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise HistoryError(f"Invalid JSON in {path} at line {line_number}: {error}") from error
        if not isinstance(raw, dict):
            raise HistoryError(f"Display history line must be an object in {path} at line {line_number}")
        record_type = raw.get("type")
        if record_type == "turn_start":
            _require_display_string(raw, "turn_id", path, line_number)
            _require_display_string(raw, "message", path, line_number)
        elif record_type == "event":
            event = raw.get("event")
            if not isinstance(event, dict):
                raise HistoryError(f"Display event must be an object in {path} at line {line_number}")
            _require_display_string(event, "type", path, line_number)
        else:
            raise HistoryError(f"Unknown display history record type in {path} at line {line_number}: {record_type}")
        records.append(raw)
    return records


def list_history_entries(agent_dir: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for metadata_path in (agent_dir / SESSIONS_DIR).glob("*/*/*/*/" + METADATA_FILE):
        try:
            metadata = _load_session_metadata(metadata_path)
            session_dir = metadata_path.parent
            path = session_dir / str(metadata.get("messages_path") or MESSAGES_FILE)
            display_path = session_dir / str(metadata.get("display_path") or DISPLAY_FILE)
            transcript_path = session_dir / str(metadata.get("transcript_path") or TRANSCRIPT_FILE)
            messages = load_history_file(path)
            stat = path.stat()
        except (HistoryError, OSError):
            continue
        entries.append(
            HistoryEntry(
                path=path,
                kind="session",
                modified_at=stat.st_mtime,
                message_count=len(messages),
                preview=_history_preview(messages),
                display_path=display_path if display_path.exists() else None,
                session_dir=session_dir,
                transcript_path=transcript_path,
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


def _load_session_metadata(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryError(f"Failed to read session metadata: {path}: {error}") from error
    if not isinstance(raw, dict):
        raise HistoryError(f"Session metadata must be an object: {path}")
    if int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
        raise HistoryError(f"Unsupported session metadata schema in {path}: {raw.get('schema_version')}")
    return raw


def _require_display_string(raw: dict[str, Any], key: str, path: Path, line_number: int) -> None:
    if not isinstance(raw.get(key), str):
        raise HistoryError(f"Display history record missing string {key!r} in {path} at line {line_number}")


def _history_preview(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role not in {"user", "assistant"}:
            continue
        content = " ".join(message.content.split())
        if not content or content.startswith("Working directory:"):
            continue
        return content[:96] + ("..." if len(content) > 96 else "")
    return "(no preview)"
