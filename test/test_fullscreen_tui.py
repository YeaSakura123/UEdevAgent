from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from contextlib import contextmanager

from prompt_toolkit.document import Document
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.widgets import TextArea

from uedev.llm.client import ChatMessage, ModelResponse, ModelStreamEvent, ToolCall
from uedev.runtime.agent import AgentOptions, AgentRuntime, SlashCommandCompleter
from uedev.runtime.history import append_display_event, load_display_history, load_history_file
from uedev.state.config import agent_dir
from uedev.tools.workspace import write_file
from uedev.ui.events import assistant_delta_event, budget_event, final_event, thinking_event, usage_event
from uedev.ui.tui import ApprovalModal, ChatScreenState, ChatTuiApplication, FullscreenRenderer, SelectionModal


@contextmanager
def workspace_temp_dir():
    root = Path.cwd() / ".tmp" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    yield path


def write_system_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "first-model": {
                        "model": "gpt-test",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "test-key",
                    }
                },
                "ue": {"engines": {}},
            }
        ),
        encoding="utf-8",
    )


class FullscreenStateTests(unittest.TestCase):
    def test_assistant_delta_is_transient_until_final(self) -> None:
        state = ChatScreenState("banner")
        state.start_turn("turn-1", "hello")

        state.render(thinking_event(1, 3, "turn-1"))
        state.render(assistant_delta_event("hel", "turn-1"))
        state.render(assistant_delta_event("lo", "turn-1"))

        self.assertIn("hello", state.render_text())

        state.render(final_event("hello", "turn-1"))
        transcript = state.render_text()

        self.assertEqual(transcript.count("hello"), 2)
        self.assertIn("summary", transcript)
        self.assertIn("assistant", transcript)

    def test_turn_summary_includes_token_usage(self) -> None:
        state = ChatScreenState("banner")
        state.start_turn("turn-1", "hello")
        state.render(
            usage_event(
                {
                    "input_tokens": 1200,
                    "output_tokens": 300,
                    "total_tokens": 1500,
                    "cached_input_tokens": 800,
                    "reasoning_tokens": 200,
                    "source": "provider",
                },
                "turn-1",
            )
        )
        state.render(final_event("done", "turn-1"))

        self.assertIn("1,500 tokens (1,200 in / 300 out)", state.render_text())

    def test_thinking_block_is_removed_after_final(self) -> None:
        state = ChatScreenState("banner")
        state.start_turn("turn-1", "hello")

        state.render(thinking_event(1, 3, "turn-1"))
        self.assertIn("thinking\nThinking...", state.render_text())

        state.render(final_event("done", "turn-1"))
        transcript = state.render_text()

        self.assertNotIn("thinking\nThinking...", transcript)
        self.assertIn("assistant\ndone", transcript)

    def test_turn_status_appears_under_user_until_assistant_streams(self) -> None:
        state = ChatScreenState("banner")
        state.start_turn("turn-1", "hello")
        state.render(budget_event("model 1/3 · requesting model", "turn-1", summary="model 1/3 · requesting model"))

        pending = state.render_text()

        self.assertLess(pending.index("user\nhello"), pending.index("status\n"))
        self.assertIn("model 1/3", pending)

        state.render(assistant_delta_event("hi", "turn-1"))
        streaming = state.render_text()

        self.assertNotIn("status\n", streaming)
        self.assertIn("assistant\nhi", streaming)

    def test_selection_modal_submit_invokes_callback(self) -> None:
        selected: list[object] = []
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                app.screen.modal = SelectionModal("Pick", ["one", "two"], [1, 2], selected.append, selected_index=1)

                self.assertTrue(app._handle_modal_submit())

        self.assertEqual(selected, [2])

    def test_fullscreen_renderer_queues_state_until_drain(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                renderer = FullscreenRenderer(app.screen, app._enqueue_ui_action)

                renderer.print_system("queued")
                self.assertNotIn("queued", app.screen.render_text())

                app._drain_ui_events()

        self.assertIn("queued", app.screen.render_text())

    def test_exit_slash_command_exits_fullscreen_app(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.exited = False

            def exit(self) -> None:
                self.exited = True

        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                fake_app = FakeApp()
                app._fullscreen_app = fake_app  # type: ignore[assignment]

                app._handle_fullscreen_query("/exit")

        self.assertTrue(fake_app.exited)

    def test_throttled_refresh_coalesces_invalidates(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.invalidates = 0

            def invalidate(self) -> None:
                self.invalidates += 1

        class FakeTimer:
            def __init__(self, delay, callback) -> None:
                self.delay = delay
                self.callback = callback
                self.daemon = False
                self.started = False

            def start(self) -> None:
                self.started = True

        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                fake_app = FakeApp()
                app._fullscreen_app = fake_app  # type: ignore[assignment]

                with patch("uedev.ui.tui.time.monotonic", side_effect=[1.0, 1.01]), patch("uedev.ui.tui.threading.Timer", FakeTimer):
                    app._request_refresh()
                    app._request_refresh()

        self.assertEqual(fake_app.invalidates, 1)
        self.assertTrue(app._invalidate_pending)

    def test_sticky_scroll_preserves_user_scroll_until_bottom(self) -> None:
        state = ChatScreenState("banner")
        for index in range(30):
            state.print_system(f"line {index}")

        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = state
                app._transcript_area = TextArea(text="", read_only=True)
                app._sync_fullscreen_controls()
                bottom = app._transcript_cursor_row()

                app._scroll_transcript(-10)
                scrolled = app._transcript_cursor_row()
                state.print_system("late line")
                app._sync_fullscreen_controls()

        self.assertLess(scrolled, bottom)
        self.assertEqual(app._transcript_cursor_row(), scrolled)
        self.assertFalse(state.sticky_scroll)

    def test_mouse_wheel_scrolls_transcript_before_page_keys(self) -> None:
        state = ChatScreenState("banner")
        for index in range(30):
            state.print_system(f"line {index}")

        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = state
                app._transcript_area = TextArea(text="", read_only=True, focusable=False)
                app._install_transcript_mouse_handler()
                app._sync_fullscreen_controls()
                app._transcript_area.window.render_info = SimpleNamespace(content_height=100, window_height=20, vertical_scroll=80)
                scroll_up = MouseEvent(Point(x=0, y=0), MouseEventType.SCROLL_UP, MouseButton.NONE, frozenset())
                scroll_down = MouseEvent(Point(x=0, y=0), MouseEventType.SCROLL_DOWN, MouseButton.NONE, frozenset())

                app._transcript_area.control.mouse_handler(scroll_up)
                first_scroll_offset = app._transcript_area.window.vertical_scroll
                scrolled = app._transcript_cursor_row()
                app._transcript_area.window.render_info = SimpleNamespace(content_height=100, window_height=20, vertical_scroll=first_scroll_offset)
                for _ in range(20):
                    app._transcript_area.control.mouse_handler(scroll_down)
                    app._transcript_area.window.render_info = SimpleNamespace(
                        content_height=100,
                        window_height=20,
                        vertical_scroll=app._transcript_area.window.vertical_scroll,
                    )

        self.assertEqual(first_scroll_offset, 77)
        self.assertLess(scrolled, 80)
        self.assertTrue(state.sticky_scroll)
        self.assertEqual(app._transcript_area.window.vertical_scroll, 80)

    def test_slash_completion_panel_accepts_completion(self) -> None:
        class FakePromptApp:
            def create_background_task(self, coroutine) -> None:
                coroutine.close()

        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                app._input_area = TextArea(multiline=False, completer=SlashCommandCompleter(), complete_while_typing=False)
                app._input_area.buffer.document = Document("/per", cursor_position=4)
                completions = list(SlashCommandCompleter().get_completions(app._input_area.buffer.document, None))
                app._input_area.buffer.complete_state = CompletionState(app._input_area.buffer.document, completions, None)

                self.assertTrue(app._completion_visible())
                with patch("prompt_toolkit.buffer.get_app", return_value=FakePromptApp()):
                    self.assertTrue(app._accept_completion())

        self.assertEqual(app._input_area.text, "/permissions")

    def test_cancelled_turn_rolls_back_state_and_restores_input(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 2, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 2, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                app._input_area = TextArea(multiline=False)
                app.renderer = FullscreenRenderer(app.screen, app._enqueue_ui_action)
                snapshot = app._create_turn_snapshot("remember this")

                def fake_stream(*args, **kwargs):
                    yield ModelStreamEvent(type="delta", delta="partial")
                    app._cancel_requested = True
                    yield ModelStreamEvent(type="final", response=ModelResponse("done"))

                with patch("uedev.runtime.agent.call_model_stream", side_effect=fake_stream):
                    app._run_turn("remember this", rollback_snapshot=snapshot)
                app._drain_ui_events()

        self.assertEqual(len(app.messages), snapshot.messages_len)
        self.assertIsNone(app.history.path)
        self.assertEqual(app._input_area.text, "remember this")
        transcript = app.screen.render_text()
        self.assertNotIn("remember this", transcript)
        self.assertNotIn("partial", transcript)
        self.assertNotIn("Cancelled by user", transcript)

    def test_cancelled_turn_truncates_existing_history_files(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 2, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 2, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                app._input_area = TextArea(multiline=False)
                app.renderer = FullscreenRenderer(app.screen, app._enqueue_ui_action)
                old_message = ChatMessage(role="user", content="old message")
                app.messages.append(old_message)
                app.history.append(old_message)
                history_path = app.history.path or Path()
                original_size = history_path.stat().st_size
                snapshot = app._create_turn_snapshot("new message")

                def fake_stream(*args, **kwargs):
                    yield ModelStreamEvent(type="delta", delta="partial")
                    app._cancel_requested = True
                    yield ModelStreamEvent(type="final", response=ModelResponse("done"))

                with patch("uedev.runtime.agent.call_model_stream", side_effect=fake_stream):
                    app._run_turn("new message", rollback_snapshot=snapshot)
                app._drain_ui_events()

                restored_messages = load_history_file(history_path)

        self.assertEqual(history_path.stat().st_size, original_size)
        self.assertEqual(restored_messages[-1].content, "old message")
        self.assertFalse((app.history.display_path or Path()).exists())
        self.assertEqual(app._input_area.text, "new message")

    def test_request_cancel_rejects_approval_modal_and_restores_input(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 1, True, root, 120, False))
                app = ChatTuiApplication(AgentOptions("", 1, True, root, 120, False), runtime, "banner", SlashCommandCompleter())
                app.screen = ChatScreenState("banner")
                app._input_area = TextArea(multiline=False)
                app._active_turn_snapshot = app._create_turn_snapshot("needs approval")
                modal = ApprovalModal(command="run", reason="test")
                app.screen.running = True
                app.screen.modal = modal

                app._request_turn_cancel()

        self.assertTrue(app._cancel_requested)
        self.assertTrue(modal.cancelled)
        self.assertTrue(modal.done.is_set())
        self.assertEqual(app._input_area.text, "needs approval")
        self.assertIsNone(app.screen.modal)


class FullscreenHistoryTests(unittest.TestCase):
    def test_assistant_delta_is_not_recorded_to_display_history(self) -> None:
        with workspace_temp_dir() as root:
            path = root / "display.jsonl"

            append_display_event(path, assistant_delta_event("partial", "turn-1"))
            append_display_event(path, budget_event("model 1/3", "turn-1"))
            append_display_event(path, final_event("done", "turn-1"))

            records = load_display_history(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"]["type"], "final")


class RuntimeStreamingTests(unittest.TestCase):
    def test_run_turn_events_emits_assistant_delta_before_final(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 3, True, root, 120, False))
                messages = [ChatMessage(role="system", content=runtime.system_prompt)]

                def fake_stream(*args, **kwargs):
                    yield ModelStreamEvent(type="delta", delta="hel")
                    yield ModelStreamEvent(type="delta", delta="lo")
                    yield ModelStreamEvent(type="final", response=ModelResponse("hello"))

                with patch("uedev.runtime.agent.call_model_stream", side_effect=fake_stream):
                    events = list(runtime.run_turn_events(messages, "say hello", turn_id="turn-stream"))

        self.assertEqual([event.type for event in events], ["thinking", "assistant_delta", "assistant_delta", "final"])
        self.assertEqual(events[-1].message, "hello")

    def test_streaming_final_tool_call_executes_tool_flow(self) -> None:
        with workspace_temp_dir() as root:
            config_path = root / "system-config.json"
            write_system_config(config_path)
            write_file(root, "a.txt", "hello")
            with patch("uedev.state.config.default_system_config_path", return_value=config_path):
                runtime = AgentRuntime(AgentOptions("", 3, True, root, 120, False))
                messages = [ChatMessage(role="system", content=runtime.system_prompt)]
                responses = iter(
                    [
                        ModelResponse("", [ToolCall(id="call-1", name="read_file", arguments={"path": "a.txt"})]),
                        ModelResponse("done"),
                    ]
                )

                def fake_stream(*args, **kwargs):
                    yield ModelStreamEvent(type="final", response=next(responses))

                with patch("uedev.runtime.agent.call_model_stream", side_effect=fake_stream):
                    events = list(runtime.run_turn_events(messages, "read a.txt", turn_id="turn-tool"))

        self.assertEqual([event.type for event in events], ["thinking", "tool_start", "tool_result", "thinking", "final"])
        self.assertEqual(events[1].name, "read_file")
        self.assertEqual(events[-1].message, "done")


if __name__ == "__main__":
    unittest.main()
