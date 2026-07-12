from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from contextlib import contextmanager

from uedev.runtime.history import (
    HistoryError,
    HistoryRecorder,
    SessionModelState,
    append_display_event,
    append_display_turn_start,
    load_display_history,
    list_history_entries,
    load_history_file,
    load_session_metadata,
    load_transcript_token_usage,
    summarize_token_usage,
    write_history_messages,
)
from uedev.runtime.context import save_transcript
from uedev.ui.events import final_event, thinking_event, tool_result_event, tool_start_event
from uedev.ui.events import plan_event
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

    def test_history_round_trips_reasoning_content_when_present(self) -> None:
        with workspace_temp_dir() as root:
            path = root / "messages.jsonl"
            messages = [
                ChatMessage(role="assistant", content="done", reasoning_content="thinking"),
                ChatMessage(role="assistant", content="plain"),
            ]

            write_history_messages(path, messages)
            raw = path.read_text(encoding="utf-8")
            loaded = load_history_file(path)

            self.assertEqual(loaded[0].reasoning_content, "thinking")
            self.assertIsNone(loaded[1].reasoning_content)
            self.assertEqual(raw.count("reasoning_content"), 1)

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

    def test_model_state_is_saved_in_metadata_and_transcript(self) -> None:
        with workspace_temp_dir() as root:
            recorder = HistoryRecorder(root / ".agent", [ChatMessage(role="system", content="system")])
            recorder.append(ChatMessage(role="user", content="hello"))
            recorder.update_model_state("Renamed GPT", "provider/gpt-5.5", "xhigh")

            metadata = load_session_metadata(recorder.session_dir or Path())
            transcript_path = recorder.transcript_path or Path()
            transcript_record = json.loads(transcript_path.read_text(encoding="utf-8").splitlines()[0])
            entries = list_history_entries(root / ".agent")

            self.assertEqual(metadata["model_state"]["profile_name"], "Renamed GPT")
            self.assertEqual(metadata["model_state"]["model"], "provider/gpt-5.5")
            self.assertEqual(metadata["model_state"]["effort"], "xhigh")
            self.assertEqual(transcript_record["type"], "session_state")
            self.assertEqual(entries[0].model_state, SessionModelState("Renamed GPT", "provider/gpt-5.5", "xhigh"))

            save_transcript([ChatMessage(role="user", content="before compact")], transcript_path)
            compacted_lines = transcript_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(compacted_lines[0])["type"], "session_state")
            self.assertEqual(load_history_file(transcript_path)[0].content, "before compact")

    def test_token_usage_is_saved_and_survives_transcript_compaction(self) -> None:
        with workspace_temp_dir() as root:
            recorder = HistoryRecorder(root / ".agent", [ChatMessage(role="system", content="system")])
            recorder.append(ChatMessage(role="user", content="hello"))
            recorder.record_token_usage(
                {
                    "request_id": "req_1",
                    "turn_id": "turn-1",
                    "purpose": "main",
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                    "cached_input_tokens": 60,
                    "reasoning_tokens": 10,
                    "source": "provider",
                }
            )
            transcript_path = recorder.transcript_path or Path()

            save_transcript([ChatMessage(role="user", content="compacted source")], transcript_path)
            recorder.append(ChatMessage(role="assistant", content="done"))

            records = load_transcript_token_usage(transcript_path)
            summary = summarize_token_usage(records)
            metadata = load_session_metadata(recorder.session_dir or Path())
            self.assertEqual(len(records), 1)
            self.assertEqual(summary["total_tokens"], 125)
            self.assertEqual(metadata["token_usage"]["cached_input_tokens"], 60)
            self.assertEqual(load_history_file(transcript_path)[0].content, "compacted source")

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

    def test_display_history_round_trips_plan_event(self) -> None:
        with workspace_temp_dir() as root:
            path = root / ".agent" / "history" / "session_1.display.jsonl"

            append_display_event(path, plan_event("# Plan\n\n- Step", str(root / "plan.md"), "Plan", "pending", "turn-1"))

            loaded = load_display_history(path)

            self.assertEqual(loaded[0]["type"], "event")
            self.assertEqual(loaded[0]["event"]["type"], "plan")
            self.assertEqual(loaded[0]["event"]["summary"], "Plan")
            self.assertEqual(loaded[0]["event"]["status"], "pending")

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
