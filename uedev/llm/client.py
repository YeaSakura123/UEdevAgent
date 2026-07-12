from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, Literal

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
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    source: Literal["provider", "estimated"] = "provider"


@dataclass(frozen=True)
class ModelStreamEvent:
    type: Literal["delta", "final"]
    delta: str = ""
    response: ModelResponse | None = None


_RESPONSES_API_OPTION_KEYS = (
    "store",
    "reasoning",
    "text",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "truncation",
    "include",
)


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


def _parse_tool_arguments(raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
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
    response: ModelResponse | None = None
    for event in call_model_stream(messages, profile, tools):
        if event.type == "final":
            response = event.response
    if response is None:
        raise RuntimeError("Model stream ended without a final response.")
    return response


def call_model_stream(
    messages: list[ChatMessage],
    profile: ModelProfile,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[ModelStreamEvent]:
    if profile.response:
        yield from _call_responses_stream(messages, profile, tools)
        return
    yield from _call_chat_completions_stream(messages, profile, tools)


def _call_chat_completions_stream(
    messages: list[ChatMessage],
    profile: ModelProfile,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[ModelStreamEvent]:
    if OpenAI is None:
        raise RuntimeError("The openai package is required to call a model.")
    if not profile.api_key:
        raise RuntimeError(f"Model profile {profile.name!r} is missing api_key in the system JSON config.")
    if not profile.model:
        raise RuntimeError(f"Model profile {profile.name!r} is missing model in the system JSON config.")

    client = OpenAI(api_key=profile.api_key, base_url=profile.base_url.rstrip("/"), timeout=profile.timeout_seconds)

    try:
        kwargs: dict[str, Any] = {
            "model": profile.model,
            "messages": [_serialize_message(message, profile) for message in messages],
            "temperature": 0.1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if profile.effort:
            kwargs["reasoning_effort"] = profile.effort
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        stream = client.chat.completions.create(**kwargs)
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    content_parts: list[str] = []
    tool_chunks: dict[int, dict[str, str]] = {}
    reasoning_parts: list[str] = []
    usage: TokenUsage | None = None
    try:
        for chunk in stream:
            chunk_usage = _parse_token_usage(getattr(chunk, "usage", None), response_api=False)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content_parts.append(str(text))
                yield ModelStreamEvent(type="delta", delta=str(text))
            reasoning = _extract_reasoning_content(delta)
            if reasoning:
                reasoning_parts.append(reasoning)
            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_call, "index", 0) or 0)
                record = tool_chunks.setdefault(index, {"id": "", "name": "", "arguments": ""})
                tool_id = getattr(tool_call, "id", None)
                if tool_id:
                    record["id"] += str(tool_id)
                function = getattr(tool_call, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if name:
                        record["name"] += str(name)
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        record["arguments"] += str(arguments)
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    try:
        tool_calls = [
            ToolCall(
                id=record["id"],
                name=record["name"],
                arguments=_parse_tool_arguments(record["arguments"]),
            )
            for _, record in sorted(tool_chunks.items())
            if record["id"] or record["name"] or record["arguments"]
        ]
    except RuntimeError as error:
        raise RuntimeError(f"Failed to parse model tool call arguments: {error}") from error

    content = "".join(content_parts)
    if not content and not tool_calls:
        raise RuntimeError("Model returned an empty response.")
    if usage is None:
        usage = estimate_token_usage(kwargs, content, "".join(reasoning_parts), tool_calls)
    yield ModelStreamEvent(
        type="final",
        response=ModelResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content="".join(reasoning_parts) or None,
            usage=usage,
        ),
    )


def _call_responses_stream(
    messages: list[ChatMessage],
    profile: ModelProfile,
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[ModelStreamEvent]:
    if OpenAI is None:
        raise RuntimeError("The openai package is required to call a model.")
    if not profile.api_key:
        raise RuntimeError(f"Model profile {profile.name!r} is missing api_key in the system JSON config.")
    if not profile.model:
        raise RuntimeError(f"Model profile {profile.name!r} is missing model in the system JSON config.")

    client = OpenAI(api_key=profile.api_key, base_url=profile.base_url.rstrip("/"), timeout=profile.timeout_seconds)

    try:
        kwargs = _build_responses_kwargs(messages, profile, tools)
        stream = client.responses.create(**kwargs)
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    content_parts: list[str] = []
    completed_response: Any = None
    try:
        for event in stream:
            event_type = str(_item_value(event, "type") or "")
            if event_type in {"response.output_text.delta", "response.text.delta"}:
                delta = _item_value(event, "delta")
                if delta:
                    text = str(delta)
                    content_parts.append(text)
                    yield ModelStreamEvent(type="delta", delta=text)
                continue
            if event_type in {"response.completed", "response.done"}:
                completed_response = _item_value(event, "response") or _item_value(event, "data")
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    if completed_response is not None:
        try:
            tool_calls = [
                ToolCall(
                    id=str(_item_value(item, "call_id") or _item_value(item, "id") or ""),
                    name=str(_item_value(item, "name") or ""),
                    arguments=_parse_tool_arguments(_item_value(item, "arguments") or ""),
                )
                for item in (_item_value(completed_response, "output") or [])
                if _item_value(item, "type") == "function_call"
            ]
        except RuntimeError as error:
            raise RuntimeError(f"Failed to parse model tool call arguments: {error}") from error
        content = _extract_responses_text(completed_response) or "".join(content_parts)
        usage = _parse_token_usage(_item_value(completed_response, "usage"), response_api=True)
    else:
        tool_calls = []
        content = "".join(content_parts)
        usage = None

    if not content and not tool_calls:
        raise RuntimeError("Model returned an empty response.")
    if usage is None:
        usage = estimate_token_usage(kwargs, content, "", tool_calls)
    yield ModelStreamEvent(type="final", response=ModelResponse(content=content, tool_calls=tool_calls, usage=usage))


def estimate_token_usage(
    request_payload: object,
    content: str,
    reasoning_content: str,
    tool_calls: list[ToolCall],
) -> TokenUsage:
    try:
        serialized_input = json.dumps(request_payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized_input = str(request_payload)
    serialized_output = content + reasoning_content
    if tool_calls:
        serialized_output += json.dumps(
            [{"name": item.name, "arguments": item.arguments} for item in tool_calls],
            ensure_ascii=False,
            default=str,
        )
    input_tokens = max(1, len(serialized_input) // 4)
    output_tokens = max(1, len(serialized_output) // 4) if serialized_output else 0
    reasoning_tokens = max(1, len(reasoning_content) // 4) if reasoning_content else 0
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=reasoning_tokens,
        source="estimated",
    )


def _parse_token_usage(raw: object, *, response_api: bool) -> TokenUsage | None:
    if raw is None:
        return None
    input_key = "input_tokens" if response_api else "prompt_tokens"
    output_key = "output_tokens" if response_api else "completion_tokens"
    input_tokens = _usage_int(raw, input_key)
    output_tokens = _usage_int(raw, output_key)
    if input_tokens is None and not response_api:
        input_tokens = _usage_int(raw, "input_tokens")
    if output_tokens is None and not response_api:
        output_tokens = _usage_int(raw, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    total_tokens = _usage_int(raw, "total_tokens")
    input_details = _item_value(raw, "input_tokens_details") or _item_value(raw, "prompt_tokens_details")
    output_details = _item_value(raw, "output_tokens_details") or _item_value(raw, "completion_tokens_details")
    cached_tokens = _usage_int(input_details, "cached_tokens") or _usage_int(raw, "prompt_cache_hit_tokens") or 0
    reasoning_tokens = _usage_int(output_details, "reasoning_tokens") or 0
    return TokenUsage(
        input_tokens=max(0, input_tokens),
        output_tokens=max(0, output_tokens),
        total_tokens=max(0, total_tokens if total_tokens is not None else input_tokens + output_tokens),
        cached_input_tokens=max(0, cached_tokens),
        reasoning_tokens=max(0, reasoning_tokens),
    )


def _usage_int(raw: object, key: str) -> int | None:
    value = _item_value(raw, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_responses_kwargs(
    messages: list[ChatMessage],
    profile: ModelProfile,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    instructions, input_items = _serialize_responses_input(messages)
    options = _compact_optional_dict(profile.responses)
    if profile.effort:
        reasoning = dict(options.get("reasoning") or {})
        reasoning["effort"] = profile.effort
        options["reasoning"] = reasoning
    kwargs: dict[str, Any] = {
        "model": profile.model,
        "input": input_items,
        "stream": True,
    }
    if instructions:
        kwargs["instructions"] = instructions
    kwargs.update(_responses_api_options(options))
    if tools is not None:
        response_tools = _serialize_responses_tools(tools, options)
        if response_tools:
            kwargs["tools"] = response_tools
    return kwargs


def _serialize_responses_input(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            if message.content.strip():
                instructions.append(message.content.strip())
            continue
        if message.role == "tool":
            if not message.tool_call_id:
                raise RuntimeError("tool messages require tool_call_id")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            if message.content:
                input_items.append({"role": "assistant", "content": message.content})
            for tool_call in message.tool_calls:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    }
                )
            continue
        if message.role in {"user", "assistant"}:
            input_items.append({"role": message.role, "content": message.content})
            continue
        input_items.append({"role": "user", "content": f"{message.role}: {message.content}"})
    return "\n\n".join(instructions), input_items


def _serialize_responses_tools(tools: list[dict[str, Any]], options: dict[str, Any]) -> list[dict[str, Any]]:
    strict = bool(options.get("strict_function_tools", False))
    response_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise RuntimeError("Responses API tools must use canonical function ToolSpec entries.")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Responses API function tools require a non-empty name.")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        response_tools.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters,
                "strict": strict,
            }
        )

    built_in = options.get("built_in_tools")
    if isinstance(built_in, dict):
        response_tools.extend(_serialize_responses_built_in_tools(built_in))
    return response_tools


def _serialize_responses_built_in_tools(built_in: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    web_search = built_in.get("web_search")
    if isinstance(web_search, dict) and web_search.get("enabled"):
        tools.append({"type": "web_search_preview"})

    file_search = built_in.get("file_search")
    if isinstance(file_search, dict) and file_search.get("enabled"):
        vector_store_ids = file_search.get("vector_store_ids") or []
        payload: dict[str, Any] = {"type": "file_search"}
        if vector_store_ids:
            payload["vector_store_ids"] = vector_store_ids
        tools.append(payload)

    remote_mcp = built_in.get("remote_mcp")
    if isinstance(remote_mcp, list):
        tools.extend(dict(tool) for tool in remote_mcp if isinstance(tool, dict))
    return tools


def _extract_responses_text(response: Any) -> str:
    output_text = _item_value(response, "output_text")
    if output_text:
        return str(output_text)
    text_parts: list[str] = []
    for item in (_item_value(response, "output") or []):
        if _item_value(item, "type") != "message":
            continue
        content = _item_value(item, "content") or []
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for part in content:
            part_type = _item_value(part, "type")
            if part_type in {"output_text", "text"}:
                text = _item_value(part, "text")
                if text:
                    text_parts.append(str(text))
    return "".join(text_parts)


def _compact_optional_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item is not None}


def _responses_api_options(options: dict[str, Any]) -> dict[str, Any]:
    api_options: dict[str, Any] = {}
    for key in _RESPONSES_API_OPTION_KEYS:
        if key not in options:
            continue
        value = _compact_optional_value(options[key])
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        api_options[key] = value
    return api_options


def _compact_optional_value(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, raw in value.items():
            item = _compact_optional_value(raw)
            if item is not None:
                compacted[key] = item
        return compacted
    if isinstance(value, list):
        return [_compact_optional_value(item) for item in value]
    return value


def _item_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


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
