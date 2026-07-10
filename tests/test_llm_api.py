"""Tests for voiceio.llm_api — OpenAI-compatible chat completions client."""
from __future__ import annotations

import http.client
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from voiceio import llm_api
from voiceio.config import AutocorrectConfig
from voiceio.llm_api import chat, check_api_key, detect_provider, resolve_api_key


@pytest.fixture(autouse=True)
def _clear_pool():
    """Keep-alive pool is process-global state — isolate every test."""
    llm_api._CONN_POOL.clear()
    yield
    llm_api._CONN_POOL.clear()


def _fake_conn(data: dict | None, status: int = 200) -> MagicMock:
    """A mock http.client connection returning one JSON response."""
    resp = MagicMock()
    resp.status = status
    resp.reason = "OK" if status == 200 else "ERR"
    resp.read.return_value = json.dumps(data or {}).encode()
    resp.getheaders.return_value = []
    conn = MagicMock()
    conn.getresponse.return_value = resp
    return conn


def _request_kwargs(conn: MagicMock) -> tuple[str, dict, dict]:
    """(path, headers, parsed body) of the single request made on conn."""
    args, kwargs = conn.request.call_args
    return args[1], kwargs["headers"], json.loads(kwargs["body"])


def _cfg(**kwargs) -> AutocorrectConfig:
    defaults = {"api_key": "test-key", "model": "test-model"}
    defaults.update(kwargs)
    return AutocorrectConfig(**defaults)


# ── resolve_api_key ──────────────────────────────────────────────────────


def test_resolve_from_config():
    cfg = _cfg(api_key="sk-from-config")
    assert resolve_api_key(cfg) == "sk-from-config"


@patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-from-env"}, clear=False)
def test_resolve_from_env():
    cfg = _cfg(api_key="")
    assert resolve_api_key(cfg) == "sk-from-env"


@patch.dict("os.environ", {}, clear=True)
def test_resolve_empty():
    cfg = _cfg(api_key="")
    assert resolve_api_key(cfg) == ""


# ── chat ─────────────────────────────────────────────────────────────────


def test_chat_success():
    conn = _fake_conn({"choices": [{"message": {"content": "Fixed text."}}]})
    with patch("http.client.HTTPSConnection", return_value=conn):
        result = chat(_cfg(), "system prompt", "user message")
    assert result == "Fixed text."

    path, headers, body = _request_kwargs(conn)
    assert path == "/v1/chat/completions" or path.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert body["model"] == "test-model"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


def test_chat_timeout():
    conn = MagicMock()
    conn.request.side_effect = OSError("timed out")
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") is None


def test_chat_http_401():
    conn = _fake_conn({"error": "invalid key"}, status=401)
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") is None


def test_chat_no_api_key():
    cfg = _cfg(api_key="")
    with patch.dict("os.environ", {}, clear=True):
        assert chat(cfg, "sys", "msg") is None


def test_chat_malformed_response():
    conn = _fake_conn({"unexpected": "format"})
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") is None


def test_chat_null_content_returns_none():
    """Thinking models (Kimi K2.6 etc.) can return content=None — must not crash."""
    conn = _fake_conn({"choices": [{"message": {"content": None}}]})
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") is None


def test_chat_falls_back_to_reasoning_field():
    """When content is null but reasoning is present, use reasoning text."""
    conn = _fake_conn({
        "choices": [{"message": {"content": None, "reasoning": "the answer"}}],
    })
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") == "the answer"


def test_chat_falls_back_to_reasoning_content_field():
    """Some providers expose `reasoning_content` instead of `reasoning`."""
    conn = _fake_conn({
        "choices": [{"message": {
            "content": None,
            "reasoning_content": "thought-through answer",
        }}],
    })
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert chat(_cfg(), "sys", "msg") == "thought-through answer"


def test_chat_anthropic_null_content_array():
    """Anthropic native API: content=null shouldn't crash."""
    conn = _fake_conn({"content": None})
    with patch("http.client.HTTPSConnection", return_value=conn):
        cfg = _cfg(base_url="https://api.anthropic.com/v1")
        assert chat(cfg, "sys", "msg") is None


# ── check_api_key ────────────────────────────────────────────────────────


def test_check_valid_key():
    conn = _fake_conn({"choices": [{"message": {"content": ""}}]})
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert check_api_key(_cfg(), "sk-valid") is True


def test_check_invalid_key():
    conn = _fake_conn({"error": "unauthorized"}, status=401)
    with patch("http.client.HTTPSConnection", return_value=conn):
        assert check_api_key(_cfg(), "sk-invalid") is False


def test_check_empty_key():
    cfg = _cfg(api_key="")
    with patch.dict("os.environ", {}, clear=True):
        assert check_api_key(cfg) is False


# ── Anthropic native API ────────────────────────────────────────────────


def test_chat_anthropic_native():
    conn = _fake_conn({"content": [{"type": "text", "text": "Fixed text."}]})
    with patch("http.client.HTTPSConnection", return_value=conn):
        cfg = _cfg(base_url="https://api.anthropic.com/v1")
        result = chat(cfg, "system prompt", "user message")
    assert result == "Fixed text."

    path, headers, body = _request_kwargs(conn)
    assert path.endswith("/messages")
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert body["system"] == "system prompt"
    assert body["messages"] == [{"role": "user", "content": "user message"}]


def test_check_api_key_anthropic():
    conn = _fake_conn({"content": [{"type": "text", "text": ""}]})
    with patch("http.client.HTTPSConnection", return_value=conn):
        cfg = _cfg(base_url="https://api.anthropic.com/v1")
        assert check_api_key(cfg, "sk-ant-test") is True
    path, _, _ = _request_kwargs(conn)
    assert path.endswith("/messages")


# ── detect_provider ─────────────────────────────────────────────────────


def test_detect_openrouter():
    base_url, model = detect_provider("sk-or-abc123")
    assert "openrouter" in base_url
    assert "kimi" in model


def test_detect_anthropic():
    base_url, model = detect_provider("sk-ant-abc123")
    assert "anthropic.com" in base_url
    assert "claude" in model


def test_detect_openai():
    base_url, model = detect_provider("sk-proj-abc123")
    assert "openai.com" in base_url


def test_detect_unknown_defaults_openrouter():
    base_url, _ = detect_provider("unknown-key-format")
    assert "openrouter" in base_url


# ── keep-alive connection pool ───────────────────────────────────────────


def test_post_json_reuses_connection():
    conn = _fake_conn({"ok": 1})
    with patch("http.client.HTTPSConnection", return_value=conn) as cls:
        r1 = llm_api._post_json("https://api.example.com/v1/x", {}, {}, 5)
        r2 = llm_api._post_json("https://api.example.com/v1/x", {}, {}, 5)
    assert r1 == {"ok": 1} and r2 == {"ok": 1}
    cls.assert_called_once()  # second call reused the pooled connection
    assert conn.request.call_count == 2


def test_post_json_retries_stale_pooled_connection():
    stale = MagicMock()
    stale.sock = None
    stale.request.side_effect = http.client.BadStatusLine("")  # server closed it
    llm_api._CONN_POOL["https://api.example.com"] = stale
    fresh = _fake_conn({"ok": 2})
    with patch("http.client.HTTPSConnection", return_value=fresh):
        r = llm_api._post_json("https://api.example.com/v1/x", {}, {}, 5)
    assert r == {"ok": 2}
    stale.close.assert_called_once()


def test_post_json_fresh_connection_error_propagates():
    conn = MagicMock()
    conn.request.side_effect = OSError("connection refused")
    with patch("http.client.HTTPSConnection", return_value=conn):
        with pytest.raises(OSError):
            llm_api._post_json("https://api.example.com/v1/x", {}, {}, 5)


def test_post_json_http_error_compatible():
    conn = _fake_conn({"error": "bad key"}, status=401)
    with patch("http.client.HTTPSConnection", return_value=conn):
        with pytest.raises(urllib.error.HTTPError) as exc:
            llm_api._post_json("https://api.example.com/v1/x", {}, {}, 5)
    assert exc.value.code == 401
    assert b"bad key" in exc.value.read()
    assert not llm_api._CONN_POOL  # error connections are not pooled
