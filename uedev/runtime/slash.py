from __future__ import annotations

from ..policy.permissions import (
    VALID_PERMISSION_MODES,
    permission_mode_description,
    permission_mode_label,
)


SLASH_COMMANDS = [
    ("/help", "Show available chat slash commands."),
    ("/context", "Show current conversation context usage."),
    ("/diff", "Show Git and Perforce workspace changes."),
    ("/todos", "Show the current lightweight todo list."),
    ("/tasks", "Show the persistent task graph."),
    ("/history", "Load a previous conversation from this project."),
    ("/subagents", "Choose a subagent conversation to view."),
    ("/worktree", "Create a UE Git linked worktree from the current project."),
    ("/model", "Choose or list model profiles for this project."),
    ("/mcp", "Show configured MCP server status and tools."),
    ("/plan", "Enter, leave, or inspect Plan Mode."),
    ("/permissions", "Show or switch the current permission mode."),
    ("/doctor", "Inspect Unreal Engine project and editor configuration."),
    ("/ue doctor", "Inspect Unreal Engine project and editor configuration."),
    ("/compact", "Compact the current conversation context."),
    ("/clear", "Reset the current chat conversation context."),
    ("/exit", "Exit interactive chat."),
]


def render_slash_help() -> str:
    width = max(len(command) for command, _ in SLASH_COMMANDS)
    lines = ["Chat commands:"]
    lines.extend(f"  {command.ljust(width)}  {description}" for command, description in SLASH_COMMANDS)
    return "\n".join(lines)


def _match_slash_commands(text: str) -> list[tuple[str, str]]:
    query = _normalize_slash_query(text)
    if not query:
        return SLASH_COMMANDS.copy()

    prefix_matches: list[tuple[str, str]] = []
    word_matches: list[tuple[str, str]] = []
    fuzzy_matches: list[tuple[str, str]] = []

    for command, description in SLASH_COMMANDS:
        command_key = command.lower()
        command_compact = _normalize_slash_query(command)
        words = [part for part in command_key.replace("/", " ").split() if part]

        if command_key.startswith(text.lower()):
            prefix_matches.append((command, description))
        elif any(word.startswith(query) for word in words):
            word_matches.append((command, description))
        elif query in command_compact or _is_subsequence(query, command_compact):
            fuzzy_matches.append((command, description))

    if prefix_matches:
        return prefix_matches
    if word_matches:
        return word_matches
    return fuzzy_matches


def _match_permission_mode_commands(text: str) -> list[tuple[str, str]]:
    lower = text.lower()
    if not (lower == "/permissions" or lower.startswith("/permissions ")):
        return []

    query = ""
    if lower.startswith("/permissions "):
        query = lower.split(" ", 1)[1].strip()

    matches: list[tuple[str, str]] = []
    for mode in VALID_PERMISSION_MODES:
        label = permission_mode_label(mode)
        command = f"/permissions {label}"
        if not query or label.startswith(query):
            matches.append((command, permission_mode_description(mode)))
    return matches


def _normalize_slash_query(text: str) -> str:
    return text.lower().lstrip("/").replace(" ", "")


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    position = 0
    for char in haystack:
        if char == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False

__all__ = [
    "SLASH_COMMANDS",
    "_is_subsequence",
    "_match_permission_mode_commands",
    "_match_slash_commands",
    "_normalize_slash_query",
    "render_slash_help",
]
