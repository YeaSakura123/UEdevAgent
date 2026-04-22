from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, OpenAIError


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


# 内部函数：把本地消息结构转换成 OpenAI Chat Completions 消息格式。
def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
                for tool_call in message.tool_calls
            ],
        }

    if message.role == "tool":
        if not message.tool_call_id:
            raise RuntimeError("tool messages require tool_call_id")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }

    return {"role": message.role, "content": message.content}


# 内部函数：解析模型返回的 function tool call 参数。
def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    if not raw_arguments.strip():
        return {}
    try:
        data = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Model returned invalid tool arguments: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("Model tool arguments must be a JSON object.")
    return data


# 外部函数：向模型发送消息，返回文本或原生 tool/function calling 结果。
def call_model(messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None) -> ModelResponse:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Create .env from .env.example and set your API key.")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL. Set it in .env.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_serialize_message(message) for message in messages],
            "temperature": 0.1,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    message = response.choices[0].message
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=_parse_tool_arguments(tool_call.function.arguments),
        )
        for tool_call in (message.tool_calls or [])
    ]
    content = message.content or ""
    if not content and not tool_calls:
        raise RuntimeError("Model returned an empty response.")

    return ModelResponse(content=content, tool_calls=tool_calls)
