from __future__ import annotations

import json
import threading
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
SESSION_STATE_RECORD = "session_state"
TOKEN_USAGE_RECORD = "token_usage"
_TRANSCRIPT_LOCK = threading.RLock()


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionModelState:
    profile_name: str
    model: str
    effort: str | None


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
    model_state: SessionModelState | None = None

    @property
    def label(self) -> str:
        timestamp = datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{self.kind}] {self.message_count} messages - {self.preview}"


@dataclass(frozen=True)
class HistorySnapshot:
    session_dir: Path | None
    path: Path | None
    display_path: Path | None
    transcript_path: Path | None
    display_seeded: bool
    initial_messages: list[ChatMessage]
    initial_display_records: list[dict[str, Any]]
    model_state: SessionModelState | None
    file_sizes: dict[Path, int | None]


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
        self.model_state: SessionModelState | None = None
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
        self.model_state = None
        self._display_seeded = False

    def resume(self, entry: HistoryEntry, initial_messages: list[ChatMessage]) -> None:
        self.initial_messages = list(initial_messages)
        self.initial_display_records = []
        self.session_dir = entry.session_dir
        self.path = entry.path
        self.display_path = entry.display_path or (entry.session_dir / DISPLAY_FILE if entry.session_dir is not None else None)
        self.transcript_path = entry.transcript_path or (entry.session_dir / TRANSCRIPT_FILE if entry.session_dir is not None else None)
        self.model_state = entry.model_state
        self._display_seeded = True

    def update_model_state(self, profile_name: str, model: str, effort: str | None) -> None:
        state = SessionModelState(profile_name=profile_name, model=model, effort=effort)
        if self.model_state == state and self.transcript_path is not None and self.transcript_path.exists():
            return
        self.ensure_session()
        self.model_state = state
        self._write_metadata()
        if self.transcript_path is not None:
            write_transcript_model_state(self.transcript_path, state)

    def record_token_usage(self, usage: dict[str, Any]) -> None:
        path = self.ensure_transcript_path()
        append_transcript_token_usage(path, usage)

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

    def snapshot(self) -> HistorySnapshot:
        paths = [self.path, self.display_path, self.transcript_path]
        if self.session_dir is not None:
            paths.append(self.session_dir / METADATA_FILE)
        file_sizes: dict[Path, int | None] = {}
        for path in paths:
            if path is None or path in file_sizes:
                continue
            try:
                file_sizes[path] = path.stat().st_size
            except FileNotFoundError:
                file_sizes[path] = None
        return HistorySnapshot(
            session_dir=self.session_dir,
            path=self.path,
            display_path=self.display_path,
            transcript_path=self.transcript_path,
            display_seeded=self._display_seeded,
            initial_messages=list(self.initial_messages),
            initial_display_records=list(self.initial_display_records),
            model_state=self.model_state,
            file_sizes=file_sizes,
        )

    def restore(self, snapshot: HistorySnapshot) -> None:
        current_paths = [self.path, self.display_path, self.transcript_path]
        if self.session_dir is not None:
            current_paths.append(self.session_dir / METADATA_FILE)
        for path in current_paths:
            if path is None or path in snapshot.file_sizes:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path, size in snapshot.file_sizes.items():
            if size is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                with path.open("r+b") as handle:
                    handle.truncate(size)
            except FileNotFoundError:
                pass
        self.session_dir = snapshot.session_dir
        self.path = snapshot.path
        self.display_path = snapshot.display_path
        self.transcript_path = snapshot.transcript_path
        self._display_seeded = snapshot.display_seeded
        self.initial_messages = list(snapshot.initial_messages)
        self.initial_display_records = list(snapshot.initial_display_records)
        self.model_state = snapshot.model_state
        if self.session_dir is not None and self.transcript_path is not None and self.transcript_path.exists():
            _update_session_token_usage_metadata(self.session_dir, self.transcript_path)

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
        write_session_metadata(
            self.session_dir,
            self.path,
            self.display_path,
            self.transcript_path,
            model_state=self.model_state,
        )


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


def write_session_metadata(
    session_dir: Path,
    messages_path: Path,
    display_path: Path,
    transcript_path: Path,
    model_state: SessionModelState | None = None,
) -> None:
    metadata_path = session_dir / METADATA_FILE
    created_at = time.time()
    active_plan: Any = None
    existing_model_state: Any = None
    existing_token_usage: Any = None
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                created_at = float(raw.get("created_at") or created_at)
                active_plan = raw.get("active_plan")
                existing_model_state = raw.get("model_state")
                existing_token_usage = raw.get("token_usage")
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
    if isinstance(active_plan, dict):
        metadata["active_plan"] = active_plan
    if model_state is not None:
        metadata["model_state"] = _model_state_to_dict(model_state)
    elif isinstance(existing_model_state, dict):
        metadata["model_state"] = existing_model_state
    if isinstance(existing_token_usage, dict):
        metadata["token_usage"] = existing_token_usage
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_session_metadata(session_dir: Path) -> dict[str, Any]:
    return _load_session_metadata(session_dir / METADATA_FILE)


def update_session_active_plan(session_dir: Path, active_plan: dict[str, Any]) -> None:
    metadata_path = session_dir / METADATA_FILE
    try:
        metadata = _load_session_metadata(metadata_path)
    except HistoryError:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_dir.name,
            "created_at": time.time(),
            "messages_path": MESSAGES_FILE,
            "display_path": DISPLAY_FILE,
            "transcript_path": TRANSCRIPT_FILE,
        }
    metadata["updated_at"] = time.time()
    metadata["active_plan"] = active_plan
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


def write_transcript_model_state(path: Path, state: SessionModelState) -> None:
    with _TRANSCRIPT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining_lines: list[str] = []
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise HistoryError(f"Failed to read transcript file: {path}: {error}") from error
            for line in lines:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    remaining_lines.append(line)
                    continue
                if not (isinstance(raw, dict) and raw.get("type") == SESSION_STATE_RECORD):
                    remaining_lines.append(line)
        record = json.dumps(_model_state_to_record(state), ensure_ascii=False)
        path.write_text("\n".join([record, *remaining_lines]) + "\n", encoding="utf-8")


def append_transcript_token_usage(path: Path, usage: dict[str, Any]) -> None:
    record = {"type": TOKEN_USAGE_RECORD, **usage}
    with _TRANSCRIPT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _update_session_token_usage_metadata(path.parent, path)


def load_transcript_token_usage(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and raw.get("type") == TOKEN_USAGE_RECORD:
            records.append(raw)
    return records


def write_transcript_messages_preserving_records(path: Path, messages: list[ChatMessage]) -> None:
    with _TRANSCRIPT_LOCK:
        model_state = load_transcript_model_state(path)
        token_usage = load_transcript_token_usage(path)
        lines: list[str] = []
        if model_state is not None:
            lines.append(json.dumps(_model_state_to_record(model_state), ensure_ascii=False))
        lines.extend(_message_to_json(message) for message in messages)
        lines.extend(json.dumps(record, ensure_ascii=False) for record in token_usage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _update_session_token_usage_metadata(path.parent, path)


def summarize_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "requests": len(records),
        "estimated_requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    for record in records:
        if record.get("source") == "estimated":
            summary["estimated_requests"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
            try:
                summary[key] += max(0, int(record.get(key) or 0))
            except (TypeError, ValueError):
                continue
    return summary


def _update_session_token_usage_metadata(session_dir: Path, transcript_path: Path) -> None:
    metadata_path = session_dir / METADATA_FILE
    try:
        metadata = _load_session_metadata(metadata_path)
    except HistoryError:
        return
    metadata["updated_at"] = time.time()
    metadata["token_usage"] = summarize_token_usage(load_transcript_token_usage(transcript_path))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_transcript_model_state(path: Path) -> SessionModelState | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and raw.get("type") == SESSION_STATE_RECORD:
            return _model_state_from_dict(raw)
    return None


def append_display_turn_start(path: Path, turn_id: str, message: str) -> None:
    append_display_record(path, {"type": "turn_start", "turn_id": turn_id, "message": message})


def append_display_event(path: Path, event: AgentEvent) -> None:
    if event.type in {"assistant_delta", "budget"}:
        return
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
        if isinstance(raw, dict) and raw.get("type") in {SESSION_STATE_RECORD, TOKEN_USAGE_RECORD}:
            continue
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
            model_state = _model_state_from_dict(metadata.get("model_state"))
            if model_state is None:
                model_state = load_transcript_model_state(transcript_path)
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
                model_state=model_state,
            )
        )
    return sorted(entries, key=lambda entry: entry.modified_at, reverse=True)


def ensure_system_prompt(messages: list[ChatMessage], system_prompt: str) -> list[ChatMessage]:
    if any(message.role == "system" for message in messages):
        return list(messages)
    return [ChatMessage(role="system", content=system_prompt), *messages]


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    if message.reasoning_content:
        payload["reasoning_content"] = message.reasoning_content
    return payload


def _message_to_json(message: ChatMessage) -> str:
    return json.dumps(message_to_dict(message), ensure_ascii=False)


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
        reasoning_content=_optional_str(raw.get("reasoning_content")),
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


def _model_state_to_dict(state: SessionModelState) -> dict[str, Any]:
    return {
        "profile_name": state.profile_name,
        "model": state.model,
        "effort": state.effort,
    }


def _model_state_to_record(state: SessionModelState) -> dict[str, Any]:
    return {
        "type": SESSION_STATE_RECORD,
        **_model_state_to_dict(state),
        "updated_at": time.time(),
    }


def _model_state_from_dict(raw: Any) -> SessionModelState | None:
    if not isinstance(raw, dict):
        return None
    profile_name = str(raw.get("profile_name") or "").strip()
    model = str(raw.get("model") or "").strip()
    effort_value = raw.get("effort")
    effort = str(effort_value).strip().lower() if effort_value is not None else None
    if not profile_name and not model:
        return None
    return SessionModelState(profile_name=profile_name, model=model, effort=effort or None)


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
