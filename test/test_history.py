from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from contextlib import contextmanager

from uedev.runtime.history import (
    HistoryError,
    HistoryRecorder,
    append_display_event,
    append_display_turn_start,
    load_display_history,
    list_history_entries,
    load_history_file,
    write_history_messages,
)
from uedev.ui.events import final_event, thinking_event, tool_result_event, tool_start_event
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
            path = root / "messages.jsonl"
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
            legacy_session_path = agent_dir / "history" / "session_1.jsonl"
            transcript_path = agent_dir / "transcripts" / "transcript_1.jsonl"
            recorder = HistoryRecorder(agent_dir, [ChatMessage(role="system", content="system")])

            write_history_messages(legacy_session_path, [ChatMessage(role="user", content="legacy request")])
            write_history_messages(transcript_path, [ChatMessage(role="assistant", content="compact answer")])
            recorder.append(ChatMessage(role="user", content="session request"))

            entries = list_history_entries(agent_dir)
            kinds = {entry.kind for entry in entries}
            previews = "\n".join(entry.preview for entry in entries)

            self.assertEqual(kinds, {"session"})
            self.assertIn("session request", previews)
            self.assertNotIn("legacy request", previews)
            self.assertNotIn("compact answer", previews)

    def test_history_recorder_is_lazy_and_persists_initial_context(self) -> None:
        with workspace_temp_dir() as root:
            agent_dir = root / ".agent"
            recorder = HistoryRecorder(agent_dir, [ChatMessage(role="system", content="system")])

            self.assertFalse((agent_dir / "sessions").exists())

            recorder.append(ChatMessage(role="user", content="hello"))
            loaded = load_history_file(recorder.path or Path())

            self.assertEqual([message.role for message in loaded], ["system", "user"])
            self.assertEqual(loaded[-1].content, "hello")
            self.assertEqual((recorder.path or Path()).name, "messages.jsonl")
            self.assertEqual((recorder.display_path or Path()).name, "display.jsonl")
            self.assertEqual((recorder.transcript_path or Path()).name, "transcript.jsonl")
            self.assertTrue(((recorder.session_dir or Path()) / "metadata.json").exists())
            self.assertEqual((recorder.session_dir or Path()).relative_to(agent_dir).parts[0], "sessions")
            self.assertEqual(len((recorder.session_dir or Path()).relative_to(agent_dir).parts), 5)

    def test_display_history_round_trips_turn_and_events(self) -> None:
        with workspace_temp_dir() as root:
            path = root / ".agent" / "history" / "session_1.display.jsonl"

            append_display_turn_start(path, "turn-1", "run shell")
            append_display_event(path, thinking_event(1, 3, "turn-1"))
            append_display_event(path, tool_start_event("shell", {"command": "Write-Output hi"}, "turn-1"))
            append_display_event(path, tool_result_event("shell", "command: Write-Output hi\nexitCode: 0", "turn-1"))
            append_display_event(path, final_event("done", "turn-1", duration_ms=1234))

            loaded = load_display_history(path)

            self.assertEqual(loaded[0]["type"], "turn_start")
            self.assertEqual(loaded[0]["message"], "run shell")
            self.assertEqual(loaded[1]["event"]["type"], "thinking")
            self.assertEqual(loaded[2]["event"]["name"], "shell")
            self.assertEqual(loaded[-1]["event"]["duration_ms"], 1234)

    def test_history_recorder_seeds_loaded_display_records(self) -> None:
        with workspace_temp_dir() as root:
            agent_dir = root / ".agent"
            seed = [{"type": "turn_start", "turn_id": "old-turn", "message": "old request"}]
            recorder = HistoryRecorder(agent_dir, [ChatMessage(role="system", content="system")], seed)

            recorder.record_turn_start("new-turn", "new request")
            loaded = load_display_history(recorder.display_path or Path())

            self.assertEqual([record["turn_id"] for record in loaded if record["type"] == "turn_start"], ["old-turn", "new-turn"])

    def test_invalid_history_json_raises(self) -> None:
        with workspace_temp_dir() as root:
            path = root / "bad.jsonl"
            path.write_text("{bad\n", encoding="utf-8")

            with self.assertRaises(HistoryError):
                load_history_file(path)
