"""Tests for voiceio.llm_api — OpenAI-compatible chat completions client."""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

from voiceio.config import AutocorrectConfig
from voiceio.llm_api import chat, check_api_key, resolve_api_key


def _mock_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


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


@patch("urllib.request.urlopen")
def test_chat_success(mock_urlopen):
    mock_urlopen.return_value = _mock_response({
        "choices": [{"message": {"content": "Fixed text."}}]
    })
    cfg = _cfg()
    result = chat(cfg, "system prompt", "user message")
    assert result == "Fixed text."

    # Verify request format
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert req.get_header("Authorization") == "Bearer test-key"
    assert req.get_header("Content-type") == "application/json"
    body = json.loads(req.data)
    assert body["model"] == "test-model"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


@patch("urllib.request.urlopen")
def test_chat_timeout(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("timed out")
    cfg = _cfg()
    assert chat(cfg, "sys", "msg") is None


@patch("urllib.request.urlopen")
def test_chat_http_401(mock_urlopen):
    err = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    err.read = lambda: b"invalid key"
    mock_urlopen.side_effect = err
    cfg = _cfg()
    assert chat(cfg, "sys", "msg") is None


def test_chat_no_api_key():
    cfg = _cfg(api_key="")
    with patch.dict("os.environ", {}, clear=True):
        assert chat(cfg, "sys", "msg") is None


@patch("urllib.request.urlopen")
def test_chat_malformed_response(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"unexpected": "format"})
    cfg = _cfg()
    assert chat(cfg, "sys", "msg") is None


# ── check_api_key ────────────────────────────────────────────────────────


@patch("urllib.request.urlopen")
def test_check_valid_key(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"choices": [{"message": {"content": ""}}]})
    cfg = _cfg()
    assert check_api_key(cfg, "sk-valid") is True


@patch("urllib.request.urlopen")
def test_check_invalid_key(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    cfg = _cfg()
    assert check_api_key(cfg, "sk-invalid") is False


def test_check_empty_key():
    cfg = _cfg(api_key="")
    with patch.dict("os.environ", {}, clear=True):
        assert check_api_key(cfg) is False
