"""Thin wrapper around the Azure OpenAI chat deployment (text + vision)."""
import base64
import json
from functools import lru_cache

from openai import AzureOpenAI

import config


@lru_cache(maxsize=1)
def get_client() -> AzureOpenAI:
    if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI settings missing - check AZURE_OPENAI_* in .env")
    return AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )


def chat(messages: list[dict], json_mode: bool = False, max_tokens: int = config.MAX_COMPLETION_TOKENS) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = get_client().chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict], max_tokens: int = config.MAX_COMPLETION_TOKENS) -> dict:
    raw = chat(messages, json_mode=True, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate stray text around the JSON object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def describe_image(png_bytes: bytes, instruction: str) -> str:
    """Send one image to the vision-capable deployment and return its text description."""
    b64 = base64.b64encode(png_bytes).decode()
    return chat([
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]},
    ], max_tokens=3000)
