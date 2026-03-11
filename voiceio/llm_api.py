"""OpenAI-compatible chat completions API client.

Supports any provider: OpenRouter, OpenAI, Anthropic, Together, Groq,
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


def chat(
    cfg: AutocorrectConfig,
    system: str,
    user_message: str,
    *,
    api_key: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Send a chat completion request. Returns response text or None on failure.

    Uses the OpenAI /v1/chat/completions format, which is supported by
    OpenRouter, OpenAI, Anthropic, Together, Groq, Ollama, and others.
    """
    key = api_key or resolve_api_key(cfg)
    if not key:
        return None

    base_url = cfg.base_url.rstrip("/")
    url = f"{base_url}/chat/completions"

    body = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    }

    payload = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=cfg.timeout_secs) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
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
    url = f"{base_url}/chat/completions"

    body = {
        "model": cfg.model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False
        # Other errors (rate limit, etc.) mean the key itself is valid
        return e.code != 403
    except Exception:
        return False
