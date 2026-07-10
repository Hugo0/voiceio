"""Multi-provider chat completions API client.

Supports OpenRouter, OpenAI, Anthropic (native Messages API), Together, Groq,
local Ollama (via /v1/chat/completions), etc. Zero dependencies beyond stdlib.
"""
from __future__ import annotations

import http.client
import io
import json
import logging
import os
import threading
import urllib.error
from urllib.parse import urlsplit

from voiceio.config import AutocorrectConfig

log = logging.getLogger(__name__)

# Keep-alive connection pool: a fresh TLS handshake costs ~0.2-0.5s per call,
# which lands directly on postcorrect's stop-to-commit latency. Connections
# are checked OUT for the duration of a request and returned only on clean
# completion, so a deadline-abandoned postcorrect thread can never share a
# socket with a later call.
_POOL_LOCK = threading.Lock()
_CONN_POOL: dict[str, http.client.HTTPConnection] = {}


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """POST JSON over a pooled keep-alive connection; return the parsed reply.

    A stale pooled connection (server closed it while idle) gets one retry on
    a fresh one. Non-2xx raises urllib.error.HTTPError so callers keep their
    existing status-code handling.
    """
    parts = urlsplit(url)
    pool_key = f"{parts.scheme}://{parts.netloc}"
    path = parts.path or "/"
    payload = json.dumps(body).encode()

    for attempt in (1, 2):
        with _POOL_LOCK:
            conn = _CONN_POOL.pop(pool_key, None)
        pooled = conn is not None
        if conn is None:
            cls = (http.client.HTTPSConnection if parts.scheme == "https"
                   else http.client.HTTPConnection)
            conn = cls(parts.hostname, parts.port, timeout=timeout)
        elif conn.sock is not None:
            conn.sock.settimeout(timeout)
        try:
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except (http.client.HTTPException, OSError):
            conn.close()
            if not pooled or attempt == 2:
                raise
            continue  # stale keep-alive — retry once on a fresh connection
        if resp.status // 100 != 2:
            conn.close()
            raise urllib.error.HTTPError(
                url, resp.status, resp.reason, dict(resp.getheaders()),
                io.BytesIO(data),
            )
        with _POOL_LOCK:
            if pool_key in _CONN_POOL:
                conn.close()  # keep at most one idle connection per host
            else:
                _CONN_POOL[pool_key] = conn
        return json.loads(data)
    raise AssertionError("unreachable")


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")
_consent_warned = False


def _is_anthropic(base_url: str) -> bool:
    """Check if the base URL points to Anthropic's native API."""
    return "api.anthropic.com" in base_url


def _is_local(base_url: str) -> bool:
    """True for loopback endpoints (Ollama etc.) that never leave the machine."""
    return any(h in base_url for h in _LOCAL_HOSTS)


def _cloud_call_allowed(cfg: AutocorrectConfig) -> bool:
    """Consent gate for cloud LLM calls (never applies to local endpoints).

    Fails open to local-only behaviour: with no recorded consent we log one
    warning and return False so the caller keeps its un-corrected text. An
    api_key explicitly set in config.toml counts as consent (recorded once)
    so existing setups keep working; env-var keys never count on their own.
    """
    global _consent_warned
    from voiceio import consent

    if consent.has_cloud_consent():
        return True
    if cfg.api_key:
        consent.record_consent(source="configured-key")
        return True
    if not _consent_warned:
        log.warning(
            "Cloud LLM call skipped: no cloud consent recorded. Run the setup "
            "wizard or 'voiceio correct' to enable cloud features, or set an "
            "api_key in config.toml. Staying local-only.",
        )
        _consent_warned = True
    return False


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

    data = _post_json(url, headers, body, timeout)
    # Anthropic returns content as a list of blocks; some thinking models
    # may also include `thinking` blocks which we ignore.
    blocks = data.get("content") or []
    if not isinstance(blocks, list):
        log.warning("Unexpected Anthropic response shape: %s", str(data)[:200])
        return None
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

    data = _post_json(url, headers, body, timeout)
    # Be defensive: thinking models (Kimi K2.6, GPT reasoning, etc.) can
    # return content=None when the answer is in `reasoning` / `reasoning_content`
    # instead. Also some malformed responses lack `choices` entirely.
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        log.warning("Unexpected response shape: %s", str(data)[:200])
        return None
    text = (
        msg.get("content")
        or msg.get("reasoning_content")
        or msg.get("reasoning")
        or ""
    )
    return text.strip() or None


def chat(
    cfg: AutocorrectConfig,
    system: str,
    user_message: str,
    *,
    api_key: str = "",
    max_tokens: int = 4096,
) -> str | None:
    """Send a chat completion request. Returns response text or None on failure.

    Automatically detects Anthropic's native API vs OpenAI-compatible format
    based on the configured base_url.
    """
    key = api_key or resolve_api_key(cfg)
    if not key:
        return None

    base_url = cfg.base_url.rstrip("/")
    if not _is_local(base_url) and not _cloud_call_allowed(cfg):
        return None
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
        return "https://openrouter.ai/api/v1", "moonshotai/kimi-k2-0905"
    if api_key.startswith("sk-ant-"):
        return "https://api.anthropic.com/v1", "claude-sonnet-4-20250514"
    if api_key.startswith(("sk-proj-", "sk-")):
        return "https://api.openai.com/v1", "gpt-4o-mini"
    # Default to OpenRouter (works with most keys)
    return "https://openrouter.ai/api/v1", "moonshotai/kimi-k2-0905"


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
