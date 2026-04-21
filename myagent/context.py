from __future__ import annotations

import json
import time
from pathlib import Path

from .llm import ChatMessage


def estimate_tokens(messages: list[ChatMessage]) -> int:
    raw = json.dumps([message.__dict__ for message in messages], ensure_ascii=False)
    return max(1, len(raw) // 4)


def micro_compact(messages: list[ChatMessage], keep_recent: int = 8, max_content: int = 4000) -> None:
    """轻量压缩旧观察结果，保留最近几轮，降低上下文膨胀。"""

    tool_indices = [
        index
        for index, message in enumerate(messages)
        if message.role == "user" and message.content.startswith("Tool result for:")
    ]
    for index in tool_indices[:-keep_recent]:
        content = messages[index].content
        if len(content) > max_content:
            first_line = content.splitlines()[0] if content else "Tool result"
            messages[index] = ChatMessage(role="user", content=f"{first_line}\n[older observation compacted]")


def save_transcript(messages: list[ChatMessage], transcript_dir: Path) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message.__dict__, ensure_ascii=False) + "\n")
    return path


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
