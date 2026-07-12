#!/usr/bin/env python3
"""List models exposed by an OpenAI-compatible endpoint and test one request."""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI, OpenAIError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--api-key", help="API key; alternatively set OPENAI_API_KEY")
    parser.add_argument("--model", help="model to test; defaults to the first returned model")
    parser.add_argument("--timeout", type=float, default=60, help="request timeout in seconds")
    parser.add_argument("--skip-request", action="store_true", help="only list models")
    args = parser.parse_args()

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        print("错误：请通过 --api-key 或 OPENAI_API_KEY 提供 API Key。", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    print(f"网址: {base_url}")
    print("Key: 已提供（不会显示明文）")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)

    try:
        models = sorted(item.id for item in client.models.list().data)
    except OpenAIError as error:
        print(f"模型列表请求失败: {error}")
        return 1

    print(f"模型列表请求成功，共 {len(models)} 个模型：")
    for model in models:
        print(f"  - {model}")
    if args.skip_request:
        return 0

    model = args.model or (models[0] if models else "")
    if not model:
        print("错误：站点没有返回模型，请使用 --model 指定模型。", file=sys.stderr)
        return 2
    print(f"测试请求: model={model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with: OK"}],
            max_tokens=16,
        )
        content = response.choices[0].message.content or "（空响应）"
    except OpenAIError as error:
        print(f"测试请求失败: {error}")
        return 1
    print("测试请求成功")
    print(f"响应: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
