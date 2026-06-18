from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


# Fill these values for quick local diagnostics. Environment variables with the
# same names still override these constants when present.
FILE_API_KEY = "sk-8626c6f51c4d94baa37df872b160500cfcde095d78b6c2c2b8ef3a78ce7"
FILE_BASE_URL = "https://tk.ljcbaby.top/v1"
FILE_MODEL = "gpt-5.5"
FILE_PROMPT_CACHE_KEY = "myagent-test-cache"
FILE_PROMPT_CACHE_RETENTION = "24h"
FILE_PREVIOUS_RESPONSE_ID = ""
FILE_REASONING_EFFORT = ""


CACHE_TEST_PREFIX = "\n".join(
    [
        (
            "Cache test static prefix. This paragraph is intentionally repeated "
            "so the request crosses the prompt caching threshold. Keep this text "
            "exactly the same between runs. It represents stable agent system "
            "instructions, tool policy, project context, and long reusable prompt "
            "content that should appear at the beginning of each request."
        )
        for _ in range(90)
    ]
)


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def configured_value(env_name: str, file_value: str) -> str | None:
    env_value = optional_env(env_name)
    if env_value:
        return env_value
    value = file_value.strip()
    return value or None


def main() -> None:
    api_key = configured_value("OPENAI_API_KEY", FILE_API_KEY)
    if not api_key or api_key == "sk-your-api-key-here":
        raise RuntimeError("Fill FILE_API_KEY or set OPENAI_API_KEY.")

    base_url = configured_value("OPENAI_BASE_URL", FILE_BASE_URL) or "https://api.openai.com/v1"
    model = configured_value("OPENAI_MODEL", FILE_MODEL) or "gpt-5"

    client = OpenAI(api_key=api_key, base_url=base_url)
    cache_test_input = (
        f"{CACHE_TEST_PREFIX}\n\n"
        "Dynamic question at the end: If you can read this message, reply in Chinese "
        "with exactly: Responses API cache test request succeeded."
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": "You are a concise assistant. Answer in Chinese.",
        "input": [
            {
                "role": "user",
                "content": cache_test_input,
            }
        ],
    }

    prompt_cache_key = configured_value("OPENAI_PROMPT_CACHE_KEY", FILE_PROMPT_CACHE_KEY)
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key

    prompt_cache_retention = configured_value("OPENAI_PROMPT_CACHE_RETENTION", FILE_PROMPT_CACHE_RETENTION)
    if prompt_cache_retention:
        kwargs["prompt_cache_retention"] = prompt_cache_retention

    previous_response_id = configured_value("OPENAI_PREVIOUS_RESPONSE_ID", FILE_PREVIOUS_RESPONSE_ID)
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    reasoning_effort = configured_value("OPENAI_REASONING_EFFORT", FILE_REASONING_EFFORT)
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)

    print("response_id:", response.id)
    print("status:", response.status)
    print("output_text:")
    print(response.output_text)

    usage = getattr(response, "usage", None)
    if usage is not None:
        print("usage:")
        print(json.dumps(usage.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
