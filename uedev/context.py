from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .llm import ChatMessage


# 内部函数：估算当前消息上下文大小，用于判断是否需要压缩。
def estimate_tokens(messages: list[ChatMessage]) -> int:
    raw = json.dumps([asdict(message) for message in messages], ensure_ascii=False)
    return max(1, len(raw) // 4)


# 内部函数：压缩旧工具结果，降低 agent loop 的上下文占用。
def micro_compact(messages: list[ChatMessage], keep_recent: int = 8, max_content: int = 4000) -> None:
    """轻量压缩旧观察结果，保留最近几轮，降低上下文膨胀。"""

    tool_indices = [
        index
        for index, message in enumerate(messages)
        if message.role in {"user", "tool"} and message.content.startswith("Tool result for:")
    ]
    for index in tool_indices[:-keep_recent]:
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


# 内部函数：保存完整会话 transcript，供压缩后追溯原始上下文。
def save_transcript(messages: list[ChatMessage], transcript_dir: Path) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
    return path


# 内部函数：生成本地压缩后的上下文摘要，替换过长会话历史。
def compact_locally(messages: list[ChatMessage], transcript_dir: Path, reason: str) -> list[ChatMessage]:
    """不额外调用模型的保守压缩：保存原文，只留下任务连续性摘要。"""

    transcript = save_transcript(messages, transcript_dir)
    recent = messages[-6:]
    summary_lines = [
        f"[Compressed locally: {reason}]",
        f"Full transcript saved at: {transcript}",
        "Recent context:",
    ]
    for message in recent:
        summary_lines.append(f"{message.role}: {message.content[:1000]}")
    return [ChatMessage(role="user", content="\n".join(summary_lines))]
