from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

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
        self.messages = self._initial_messages()

    # 外部函数：启动 transcript-flow chat，负责输入、slash command 和 agent 事件顺序输出。
    def run(self) -> None:
        from .loop import create_chat_prompt_options, create_chat_session

        session = create_chat_session(completer=self.completer, input=self.input, output=self.output)
        self.renderer.print_banner()

        while True:
            try:
                query = session.prompt([("class:prompt", "\n› ")], **create_chat_prompt_options()).strip()
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

    # 内部函数：创建一轮会话的初始上下文，负责保留系统提示和当前工作目录信息。
    def _initial_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=self.runtime.system_prompt),
            ChatMessage(role="user", content=f"Working directory: {self.options.cwd}\nShell: {shell_name()}"),
        ]

    # 内部函数：同步运行一轮 agent，负责按时间顺序把事件追加到 transcript。
    def _run_turn(self, goal: str) -> None:
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        self.messages.append(ChatMessage(role="user", content=goal))
        self.renderer.start_turn(turn_id, goal)
        try:
            for event in self.runtime.run_turn_events(self.messages, goal=goal, turn_id=turn_id):
                self.renderer.render(event)
        except Exception as error:
            self.renderer.render(stopped_event(f"Error: {error}", turn_id=turn_id))
