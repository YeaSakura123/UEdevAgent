from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
from prompt_toolkit.input.base import Input
from prompt_toolkit.output.base import Output

from .events import stopped_event
from .llm import ChatMessage
from .renderer import TuiRenderer
from .shell import shell_name

if TYPE_CHECKING:
    from .loop import AgentOptions, AgentRuntime


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

    def run(self) -> None:
        from .loop import create_chat_prompt_options, create_chat_session

        session = create_chat_session(completer=self.completer, input=self.input, output=self.output)
        self.renderer.print_banner()

        while True:
            try:
                query = session.prompt([("class:prompt", "\n> ")], **create_chat_prompt_options()).strip()
            except (EOFError, KeyboardInterrupt):
                return

            if query.lower() in {"", "quit", "exit"}:
                return

            if query.lower() == "/clear":
                self.messages = self._initial_messages()
                self.renderer.clear()
                self.renderer.print_system("Conversation context cleared.")
                continue

            if self.runtime.handle_slash_command(query, emit=self.renderer.print_system):
                continue

            self._run_turn(query)

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
        self.messages.append(ChatMessage(role="user", content=goal))
        self.renderer.start_turn(turn_id, goal)
        try:
            for event in self.runtime.run_turn_events(self.messages, goal=goal, turn_id=turn_id):
                self.renderer.render(event)
        except Exception as error:
            self.renderer.render(stopped_event(f"Error: {error}", turn_id=turn_id))
