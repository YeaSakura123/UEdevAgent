from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    from openai import OpenAI, OpenAIError
except ModuleNotFoundError:  # pragma: no cover - only used in minimal test environments.
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[assignment]

from ..state.config import ModelProfile


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
    reasoning_content: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None


def _serialize_message(message: ChatMessage, profile: ModelProfile | None = None) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        payload: dict[str, Any] = {
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
        if _should_send_reasoning_content(message, profile):
            payload["reasoning_content"] = message.reasoning_content
        return payload

    if message.role == "tool":
        if not message.tool_call_id:
            raise RuntimeError("tool messages require tool_call_id")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }

    payload = {"role": message.role, "content": message.content}
    if _should_send_reasoning_content(message, profile):
        payload["reasoning_content"] = message.reasoning_content
    return payload


def _should_send_reasoning_content(message: ChatMessage, profile: ModelProfile | None) -> bool:
    return (
        message.role == "assistant"
        and bool(message.reasoning_content)
        and bool(profile is not None and profile.requires_reasoning_content)
    )


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


def call_model(
    messages: list[ChatMessage],
    profile: ModelProfile,
    tools: list[dict[str, Any]] | None = None,
) -> ModelResponse:
    if OpenAI is None:
        raise RuntimeError("The openai package is required to call a model.")
    if not profile.api_key:
        raise RuntimeError(f"Model profile {profile.name!r} is missing api_key in the system JSON config.")
    if not profile.model:
        raise RuntimeError(f"Model profile {profile.name!r} is missing model in the system JSON config.")

    client = OpenAI(api_key=profile.api_key, base_url=profile.base_url.rstrip("/"), timeout=120)

    try:
        kwargs: dict[str, Any] = {
            "model": profile.model,
            "messages": [_serialize_message(message, profile) for message in messages],
            "temperature": 0.1,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    message = response.choices[0].message
    try:
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=_parse_tool_arguments(tool_call.function.arguments or ""),
            )
            for tool_call in (message.tool_calls or [])
        ]
    except RuntimeError as error:
        raise RuntimeError(f"Failed to parse model tool call arguments: {error}") from error
    content = message.content or ""
    reasoning_content = _extract_reasoning_content(message)
    if not content and not tool_calls:
        raise RuntimeError("Model returned an empty response.")

    return ModelResponse(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


def _extract_reasoning_content(message: Any) -> str | None:
    value = getattr(message, "reasoning_content", None)
    if value is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("reasoning_content")
    if value is None:
        return None
    text = str(value)
    return text if text else None
