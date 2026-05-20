from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from contextlib import contextmanager

from uedev.runtime.history import (
    HistoryError,
    HistoryRecorder,
    create_session_history_path,
    list_history_entries,
    load_history_file,
    write_history_messages,
)
from uedev.llm.client import ChatMessage, ToolCall


@contextmanager
def workspace_temp_dir():
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    yield path


class HistoryTests(unittest.TestCase):
    def test_history_round_trips_tool_calls(self) -> None:
        with workspace_temp_dir() as root:
            path = create_session_history_path(root / ".agent")
            messages = [
                ChatMessage(role="system", content="system"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
                ),
                ChatMessage(role="tool", content="ok", tool_call_id="call_1", name="read_file"),
            ]

            write_history_messages(path, messages)
            loaded = load_history_file(path)

            self.assertEqual(loaded[1].tool_calls[0].name, "read_file")
            self.assertEqual(loaded[2].tool_call_id, "call_1")

    def test_history_entries_include_only_sessions(self) -> None:
        with workspace_temp_dir() as root:
            agent_dir = root / ".agent"
            session_path = agent_dir / "history" / "session_1.jsonl"
            transcript_path = agent_dir / "transcripts" / "transcript_1.jsonl"

            write_history_messages(session_path, [ChatMessage(role="user", content="session request")])
            write_history_messages(transcript_path, [ChatMessage(role="assistant", content="compact answer")])

            entries = list_history_entries(agent_dir)
            kinds = {entry.kind for entry in entries}
            previews = "\n".join(entry.preview for entry in entries)

            self.assertEqual(kinds, {"session"})
            self.assertIn("session request", previews)
            self.assertNotIn("compact answer", previews)

    def test_history_recorder_is_lazy_and_persists_initial_context(self) -> None:
        with workspace_temp_dir() as root:
            agent_dir = root / ".agent"
            recorder = HistoryRecorder(agent_dir, [ChatMessage(role="system", content="system")])

            self.assertFalse((agent_dir / "history").exists())

            recorder.append(ChatMessage(role="user", content="hello"))
            loaded = load_history_file(recorder.path or Path())

            self.assertEqual([message.role for message in loaded], ["system", "user"])
            self.assertEqual(loaded[-1].content, "hello")

    def test_invalid_history_json_raises(self) -> None:
        with workspace_temp_dir() as root:
            path = root / ".agent" / "history" / "session_bad.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("{bad\n", encoding="utf-8")

            with self.assertRaises(HistoryError):
                load_history_file(path)
