from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI, OpenAIError


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def call_model(messages: list[ChatMessage]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Create .env from .env.example and set your API key.")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL. Set it in .env.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            temperature=0.1,
        )
    except OpenAIError as error:
        raise RuntimeError(f"Model request failed: {error}") from error

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned an empty response.")

    return content
