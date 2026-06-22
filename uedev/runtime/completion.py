from __future__ import annotations

from collections.abc import Iterator

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts.prompt import CompleteStyle
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - exercised only in minimal test environments.
    PromptSession = None  # type: ignore[assignment]

    class Completer:  # type: ignore[no-redef]
        pass

    class Completion:  # type: ignore[no-redef]
        def __init__(self, text: str, **kwargs: object) -> None:
            self.text = text
            self.display = kwargs.get("display", text)
            self.display_meta = kwargs.get("display_meta", "")

    class CompleteStyle:  # type: ignore[no-redef]
        COLUMN = "COLUMN"

    class CursorShape:  # type: ignore[no-redef]
        BLINKING_BLOCK = "BLINKING_BLOCK"

    class SimpleCursorShapeConfig:  # type: ignore[no-redef]
        def __init__(self, cursor_shape: object) -> None:
            self.cursor_shape = cursor_shape

    class InMemoryHistory:  # type: ignore[no-redef]
        pass

    class Style:  # type: ignore[no-redef]
        @staticmethod
        def from_dict(style: dict[str, str]) -> dict[str, str]:
            return style

from .slash import (
    SLASH_COMMANDS,
    _is_subsequence,
    _match_permission_mode_commands,
    _match_slash_commands,
    _normalize_slash_query,
    render_slash_help,
)


class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        permission_matches = _match_permission_mode_commands(text)
        if permission_matches:
            for command, description in permission_matches:
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )
            return

        for command, description in _match_slash_commands(text):
            yield Completion(
                command,
                start_position=-len(text),
                display=command,
                display_meta=description,
            )


def create_chat_style() -> Style:
    return Style.from_dict(
        {
            "completion-menu.completion": "fg:#c0c0c0 bg:#202020",
            "completion-menu.completion.current": "fg:#ffffff bg:#005f87",
            "completion-menu.meta.completion": "fg:#808080 bg:#202020",
            "completion-menu.meta.completion.current": "fg:#ffffff bg:#005f87",
            "prompt": "fg:#5fafff bold",
        }
    )


def create_chat_prompt_options() -> dict[str, object]:
    return {
        "complete_while_typing": True,
        "complete_style": CompleteStyle.COLUMN,
        "reserve_space_for_menu": 8,
        "cursor": SimpleCursorShapeConfig(CursorShape.BLINKING_BLOCK),
        "refresh_interval": 0.5,
    }


def create_chat_session(
    completer: Completer | None = None,
    input=None,
    output=None,
    key_bindings=None,
    input_processors=None,
) -> PromptSession:
    if PromptSession is None:
        raise RuntimeError("prompt_toolkit is required for interactive chat sessions")
    kwargs = {
        "completer": completer or SlashCommandCompleter(),
        "complete_while_typing": True,
        "history": InMemoryHistory(),
        "style": create_chat_style(),
    }
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if key_bindings is not None:
        kwargs["key_bindings"] = key_bindings
    if input_processors is not None:
        kwargs["input_processors"] = input_processors
    return PromptSession(**kwargs)

__all__ = [
    "SLASH_COMMANDS",
    "SlashCommandCompleter",
    "_is_subsequence",
    "_match_permission_mode_commands",
    "_match_slash_commands",
    "_normalize_slash_query",
    "create_chat_prompt_options",
    "create_chat_session",
    "create_chat_style",
    "render_slash_help",
]
