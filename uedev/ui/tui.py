from __future__ import annotations

import shutil
import uuid
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output.base import Output

from ..state.config import ConfigError
from ..runtime.history import HistoryEntry, HistoryError, HistoryRecorder, ensure_system_prompt, list_history_entries, load_history_file
from uedev.ui.events import stopped_event
from ..llm.client import ChatMessage
from uedev.ui.renderer import TuiRenderer
from ..tools.shell import shell_name

if TYPE_CHECKING:
    from ..runtime.agent import AgentOptions, AgentRuntime
    from ..runtime.subagents import SubagentRecord


class ChatTuiApplication:
    def __init__(
        self,
        options: "AgentOptions",
        runtime: "AgentRuntime",
        banner: str,
        completer: Completer,
        input: Input | None = None,
        output: Output | None = None,
    ):
        self.options = options
        self.runtime = runtime
        self.banner = banner
        self.completer = completer
        self.input = input
        self.output = output
        self.renderer = TuiRenderer(banner=banner, verbose=options.verbose)
        self.runtime.approval_provider = self.confirm_command
        self.messages = self._initial_messages()
        self.history = HistoryRecorder(self.runtime.agent_dir, self.messages)
        self.current_subagent: "SubagentRecord | None" = None

    def run(self) -> None:
        from ..runtime.agent import create_chat_prompt_options, create_chat_session

        session = create_chat_session(
            completer=self.completer,
            input=self.input,
            output=self.output,
            key_bindings=self.create_key_bindings(),
        )
        self.renderer.print_banner()

        while True:
            try:
                prompt_options = create_chat_prompt_options()
                prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
                query = session.prompt([("class:prompt", "\n> ")], **prompt_options).strip()
            except (EOFError, KeyboardInterrupt):
                return

            if query.lower() in {"", "quit", "exit"}:
                return

            if query.lower() == "/clear":
                self.messages = self._initial_messages()
                self.history.reset(self.messages)
                self.renderer.clear()
                self.renderer.print_system("Conversation context cleared.")
                continue

            if query.lower() == "/history":
                selected = self.prompt_history_selection(session)
                if selected is not None:
                    self.load_history(selected)
                continue

            if query.lower() == "/subagents":
                selected = self.prompt_subagent_selection(session)
                if selected == "main":
                    self.load_main_conversation()
                elif selected is not None:
                    self.load_subagent(selected)
                continue

            if query.lower() == "/permissions":
                selected = self.prompt_permission_mode(session)
                if selected is None:
                    continue
                query = selected

            if self.current_subagent is not None and not query.startswith("/"):
                self.renderer.print_system(
                    f"Subagent {self.current_subagent.id} is {self.current_subagent.status} and closed. "
                    "Use /subagents to switch back to the main conversation."
                )
                continue

            if self.runtime.handle_slash_command(query, emit=self.renderer.print_system, messages=self.messages):
                continue

            self._run_turn(query)

    def create_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _exit_plan_mode(event) -> None:
            if self.exit_plan_mode():
                event.app.invalidate()

        return bindings

    def exit_plan_mode(self) -> bool:
        if self.runtime.collaboration_mode != "plan":
            return False
        self.runtime.collaboration_mode = "default"
        return True

    def status_fragments(self):
        model = self._status_model_name()
        directory = str(self.options.cwd)
        right = ""
        if self.current_subagent is not None:
            right = f"Viewing {self.current_subagent.id} "
        elif self.runtime.collaboration_mode == "plan":
            right = "Plan mode "
        left_length = len(model) + 3 + len(directory)
        right_length = len(right)
        width = self._terminal_width()
        fragments = [
            ("", model),
            ("", "   "),
            ("", directory),
        ]
        if right:
            fragments.append(("", " " * max(1, width - left_length - right_length)))
            fragments.append(("", right))
        return fragments

    def status_bottom_toolbar(self):
        return self.status_fragments()

    def prompt_permission_mode(self, session: PromptSession) -> str | None:
        from ..runtime.agent import create_chat_prompt_options

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\n> ")],
                default="/permissions ",
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return selected or None

    def prompt_history_selection(self, session: PromptSession) -> HistoryEntry | None:
        from ..runtime.agent import create_chat_prompt_options

        entries = list_history_entries(self.runtime.agent_dir)
        if not entries:
            self.renderer.print_system("No history found for this project.")
            return None

        labels = [f"{index}. {entry.label}" for index, entry in enumerate(entries, start=1)]
        by_label = {label: entry for label, entry in zip(labels, entries)}

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\nHistory> ")],
                completer=WordCompleter(labels, ignore_case=True, sentence=True),
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not selected:
            return None
        entry = by_label.get(selected)
        if entry is None:
            self.renderer.print_system(f"Unknown history selection: {selected}")
        return entry

    def prompt_subagent_selection(self, session: PromptSession):
        from ..runtime.agent import create_chat_prompt_options

        records = self.runtime.subagents.list_records()
        labels = ["Main conversation", *[f"{index}. {record.label}" for index, record in enumerate(records, start=1)]]
        by_label = {"Main conversation": "main"}
        by_label.update({label: record for label, record in zip(labels[1:], records)})

        def start_completion() -> None:
            session.app.current_buffer.start_completion(select_first=True)

        try:
            prompt_options = create_chat_prompt_options()
            prompt_options["bottom_toolbar"] = self.status_bottom_toolbar
            selected = session.prompt(
                [("class:prompt", "\nSubagents> ")],
                completer=WordCompleter(labels, ignore_case=True, sentence=True),
                pre_run=start_completion,
                **prompt_options,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not selected:
            return None
        choice = by_label.get(selected)
        if choice is None:
            self.renderer.print_system(f"Unknown subagent selection: {selected}")
        return choice

    def load_history(self, entry: HistoryEntry) -> None:
        try:
            messages = ensure_system_prompt(load_history_file(entry.path), self.runtime.system_prompt)
        except HistoryError as error:
            self.renderer.print_system(f"Failed to load history: {error}")
            return
        self.messages = messages
        self.history.reset(self.messages)
        self.current_subagent = None
        self.renderer.render_history(self.messages, str(entry.path))

    def load_subagent(self, record: "SubagentRecord") -> None:
        try:
            messages = self.runtime.subagents.load_messages(record)
        except HistoryError as error:
            self.renderer.print_system(f"Failed to load subagent history: {error}")
            return
        self.current_subagent = record
        self.renderer.render_history(messages, f"subagent {record.id}: {record.history_path}")

    def load_main_conversation(self) -> None:
        self.current_subagent = None
        messages = self.messages
        source = "main conversation"
        if self.history.path is not None:
            try:
                messages = ensure_system_prompt(load_history_file(self.history.path), self.runtime.system_prompt)
                source = str(self.history.path)
            except HistoryError as error:
                self.renderer.print_system(f"Failed to load main conversation history: {error}")
        self.renderer.render_history(messages, source)

    def _status_model_name(self) -> str:
        try:
            profile = self.runtime.current_model_profile()
        except ConfigError:
            return "(missing config)"
        return profile.model or profile.name or "(missing model)"

    def plan_mode_bottom_toolbar(self):
        return self.status_fragments()

    def _terminal_width(self) -> int:
        if self.output is not None:
            try:
                return int(self.output.get_size().columns)
            except Exception:
                pass
        return shutil.get_terminal_size((80, 20)).columns

    def confirm_command(self, command: str, reason: str) -> bool:
        self.renderer.print_approval(command, reason)
        session_kwargs = {}
        if self.input is not None:
            session_kwargs["input"] = self.input
        if self.output is not None:
            session_kwargs["output"] = self.output
        session = PromptSession(**session_kwargs)
        answer = session.prompt("Approve? [y/N] ").strip().lower()
        approved = answer == "y"
        self.renderer.print_system("Approved." if approved else "Rejected.")
        return approved

    def _initial_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.runtime.system_prompt),
            ChatMessage(role="user", content=f"Working directory: {self.options.cwd}\nShell: {shell_name()}"),
        ]

    def _run_turn(self, goal: str) -> None:
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        self.renderer.start_turn(turn_id, goal)
        try:
            for event in self.runtime.run_turn_events(self.messages, goal=goal, turn_id=turn_id, history=self.history):
                self.renderer.render(event)
        except Exception as error:
            self.renderer.render(stopped_event(f"Error: {error}", turn_id=turn_id))
