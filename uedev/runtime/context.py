from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..llm.client import ChatMessage


SUMMARY_PREFIX = "[Conversation summary]"
RUNTIME_STATE_MARKER = "<runtime-state>"
COMPACT_USER_MESSAGE_MAX_TOKENS = 20000
SUMMARIZATION_PROMPT = """You are compacting a long agent conversation so another model turn can continue from a short replacement history.

Write a concise but complete handoff summary. Preserve:
- the user's current goal and relevant prior requests
- completed work and important observations
- files, commands, tools, task ids, and decisions that matter
- outstanding work, blockers, and next steps

Do not include filler, greetings, or speculation. The result will replace older conversation history."""


def estimate_tokens(messages: list[ChatMessage]) -> int:
    raw = json.dumps([asdict(message) for message in messages], ensure_ascii=False)
    return max(1, len(raw) // 4)


def micro_compact(messages: list[ChatMessage], keep_recent: int = 8, max_content: int = 4000) -> None:
    """Compact older tool observations without breaking tool-call identity."""

    tool_indices = [
        index
        for index, message in enumerate(messages)
        if message.role in {"user", "tool"} and message.content.startswith("Tool result for:")
    ]
    compact_indices = tool_indices if keep_recent <= 0 else tool_indices[:-keep_recent]
    for index in compact_indices:
        content = messages[index].content
        if len(content) > max_content:
            first_line = content.splitlines()[0] if content else "Tool result"
            messages[index] = ChatMessage(
                role=messages[index].role,
                content=f"{first_line}\n[older observation compacted]",
                tool_call_id=messages[index].tool_call_id,
                name=messages[index].name,
            )


def repair_tool_call_messages(messages: list[ChatMessage]) -> None:
    """Keep OpenAI tool-call history valid after compaction or older sessions.

    An assistant message with tool_calls must be followed immediately by one
    tool message for every tool_call_id. If older compaction damaged that shape,
    downgrade the assistant tool-call record to plain context before sending.
    """

    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue

        expected = [tool_call.id for tool_call in message.tool_calls]
        cursor = index + 1
        observed: list[str] = []
        while cursor < len(messages) and messages[cursor].role == "tool":
            if messages[cursor].tool_call_id:
                observed.append(messages[cursor].tool_call_id)
            cursor += 1

        if observed[: len(expected)] != expected:
            tool_names = ", ".join(tool_call.name for tool_call in message.tool_calls)
            content = message.content.strip()
            summary = f"{content}\n[tool calls omitted from compacted history: {tool_names}]".strip()
            messages[index] = ChatMessage(role="assistant", content=summary)

        index += 1


def save_transcript(messages: list[ChatMessage], transcript_dir: Path) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"transcript_{time.time_ns()}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
    return path


def is_runtime_state_message(message: ChatMessage) -> bool:
    return message.role == "system" and message.content.startswith(RUNTIME_STATE_MARKER)


def is_summary_message(message: ChatMessage) -> bool:
    return message.role == "user" and message.content.startswith(SUMMARY_PREFIX)


def is_real_user_message(message: ChatMessage) -> bool:
    if message.role != "user":
        return False
    content = message.content.strip()
    if not content:
        return False
    if content.startswith(SUMMARY_PREFIX):
        return False
    if content.startswith("Tool result for:"):
        return False
    if content.startswith("Working directory:"):
        return False
    if content.startswith("<background-results>") or content.startswith("<inbox>"):
        return False
    return True


def latest_real_user_message(messages: list[ChatMessage]) -> ChatMessage | None:
    for message in reversed(messages):
        if is_real_user_message(message):
            return message
    return None


def build_compaction_request(messages: list[ChatMessage], reason: str) -> list[ChatMessage]:
    request = [
        message
        for message in messages
        if not is_runtime_state_message(message)
    ]
    request.append(
        ChatMessage(
            role="user",
            content=f"{SUMMARIZATION_PROMPT}\n\nCompaction reason: {reason}",
        )
    )
    return request


def build_compacted_history(
    messages: list[ChatMessage],
    summary: str,
    max_user_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> list[ChatMessage]:
    system_message = next(
        (message for message in messages if message.role == "system" and not is_runtime_state_message(message)),
        None,
    )
    selected_users: list[ChatMessage] = []
    used_tokens = 0
    for message in reversed(messages):
        if not is_real_user_message(message):
            continue
        message_tokens = estimate_tokens([message])
        if used_tokens + message_tokens > max_user_tokens:
            continue
        selected_users.append(message)
        used_tokens += message_tokens

    selected_users.reverse()
    summary_content = summary.strip()
    if not summary_content.startswith(SUMMARY_PREFIX):
        summary_content = f"{SUMMARY_PREFIX}\n{summary_content}".strip()

    compacted = [*selected_users, ChatMessage(role="user", content=summary_content)]
    if system_message is not None:
        return [system_message, *compacted]
    return compacted


def compact_locally(messages: list[ChatMessage], transcript_dir: Path, reason: str) -> list[ChatMessage]:
    """Legacy local compaction used by older tests and callers."""

    transcript = save_transcript(messages, transcript_dir)
    system_message = messages[0] if messages and messages[0].role == "system" else None
    recent = [message for message in messages if message.role != "system"][-6:]
    summary_lines = [
        f"[Compressed locally: {reason}]",
        f"Full transcript saved at: {transcript}",
        "Recent context:",
    ]
    for message in recent:
        summary_lines.append(f"{message.role}: {message.content[:1000]}")
    compacted = [ChatMessage(role="user", content="\n".join(summary_lines))]
    if system_message is not None:
        return [system_message, *compacted]
    return compacted
