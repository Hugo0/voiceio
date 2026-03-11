"""Multi-provider chat completions API client.

Supports OpenRouter, OpenAI, Anthropic (native Messages API), Together, Groq,
local Ollama (via /v1/chat/completions), etc. Zero dependencies beyond stdlib.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from voiceio.config import AutocorrectConfig

log = logging.getLogger(__name__)


def _is_anthropic(base_url: str) -> bool:
    """Check if the base URL points to Anthropic's native API."""
    return "api.anthropic.com" in base_url


def resolve_api_key(cfg: AutocorrectConfig) -> str:
    """Resolve API key from config or environment variables."""
    if cfg.api_key:
        return cfg.api_key
    # Check common env vars in priority order
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var, "")
        if val:
            return val
    return ""


def _anthropic_request(
    base_url: str,
    model: str,
    system: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int,
    timeout: float,
) -> str | None:
    """Send a request using Anthropic's native Messages API."""
    url = f"{base_url}/messages"

    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        body["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    # Anthropic returns content as a list of blocks
    blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return text.strip() or None


def _openai_request(
    base_url: str,
    model: str,
    system: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int,
    timeout: float,
) -> str | None:
    """Send a request using the OpenAI chat completions format."""
    url = f"{base_url}/chat/completions"

    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": all_messages,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def chat(
    cfg: AutocorrectConfig,
    system: str,
    user_message: str,
    *,
    api_key: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Send a chat completion request. Returns response text or None on failure.

    Automatically detects Anthropic's native API vs OpenAI-compatible format
    based on the configured base_url.
    """
    key = api_key or resolve_api_key(cfg)
    if not key:
        return None

    base_url = cfg.base_url.rstrip("/")
    messages = [{"role": "user", "content": user_message}]

    try:
        if _is_anthropic(base_url):
            return _anthropic_request(
                base_url, cfg.model, system, messages, key, max_tokens, cfg.timeout_secs,
            )
        return _openai_request(
            base_url, cfg.model, system, messages, key, max_tokens, cfg.timeout_secs,
        )
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode()[:200]
        except Exception:
            pass
        log.warning("API request failed (HTTP %d): %s", e.code, body_text)
        return None
    except Exception as e:
        log.warning("API request failed: %s", e)
        return None


def detect_provider(api_key: str) -> tuple[str, str]:
    """Detect provider from API key prefix. Returns (base_url, model)."""
    if api_key.startswith("sk-or-"):
        return "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4"
    if api_key.startswith("sk-ant-"):
        return "https://api.anthropic.com/v1", "claude-sonnet-4-20250514"
    if api_key.startswith(("sk-proj-", "sk-")):
        return "https://api.openai.com/v1", "gpt-4o-mini"
    # Default to OpenRouter (works with most keys)
    return "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4"


def check_api_key(cfg: AutocorrectConfig, api_key: str = "") -> bool:
    """Validate an API key with a minimal request."""
    key = api_key or resolve_api_key(cfg)
    if not key:
        return False

    base_url = cfg.base_url.rstrip("/")
    messages = [{"role": "user", "content": "hi"}]

    try:
        if _is_anthropic(base_url):
            _anthropic_request(base_url, cfg.model, "", messages, key, 1, 10)
        else:
            _openai_request(base_url, cfg.model, "", messages, key, 1, 10)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False
        # Other errors (rate limit, etc.) mean the key itself is valid
        return e.code != 403
    except Exception:
        return False
