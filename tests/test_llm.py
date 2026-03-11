"""Tests for LLM post-processing via Ollama."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from voiceio.config import LLMConfig
from voiceio.llm import LLMProcessor, _strip_echo


def _make_processor(**kwargs) -> LLMProcessor:
    defaults = {"enabled": True, "model": "phi3:mini", "base_url": "http://localhost:11434"}
    defaults.update(kwargs)
    return LLMProcessor(LLMConfig(**defaults))


def _mock_response(data: dict) -> MagicMock:
    """Create a mock urllib response with JSON body."""
    body = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestProcess:
    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"response": "The sky is blue."}
        )
        proc = _make_processor()
        result = proc.process("the skys is blue")
        assert result == "The sky is blue."

    @patch("urllib.request.urlopen")
    def test_timeout_returns_original(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        proc = _make_processor()
        result = proc.process("hello world")
        assert result == "hello world"

    @patch("urllib.request.urlopen")
    def test_connection_error_returns_original(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("refused")
        proc = _make_processor()
        result = proc.process("hello world")
        assert result == "hello world"

    @patch("urllib.request.urlopen")
    def test_hallucination_too_long(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"response": "x " * 500}  # way longer than input
        )
        proc = _make_processor()
        result = proc.process("short text")
        assert result == "short text"

    @patch("urllib.request.urlopen")
    def test_hallucination_too_short(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"response": "x"}
        )
        proc = _make_processor()
        original = "this is a much longer piece of text that should not shrink to one char"
        result = proc.process(original)
        assert result == original

    @patch("urllib.request.urlopen")
    def test_empty_response_returns_original(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"response": ""})
        proc = _make_processor()
        result = proc.process("hello world")
        assert result == "hello world"

    def test_empty_passthrough(self):
        proc = _make_processor()
        assert proc.process("") == ""
        assert proc.process("   ") == "   "

    @patch("urllib.request.urlopen")
    def test_prompt_echo_stripped(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"response": "Here is the corrected text: The sky is blue."}
        )
        proc = _make_processor()
        result = proc.process("the skys is blue")
        assert result == "The sky is blue."

    @patch("urllib.request.urlopen")
    def test_quoted_response_stripped(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"response": '"The sky is blue."'}
        )
        proc = _make_processor()
        result = proc.process("the skys is blue")
        assert result == "The sky is blue."


class TestIsAvailable:
    @patch("urllib.request.urlopen")
    def test_running_with_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "phi3:mini"}, {"name": "llama3.2:1b"}]}
        )
        proc = _make_processor(model="phi3:mini")
        assert proc.is_available() is True

    @patch("urllib.request.urlopen")
    def test_running_model_not_found(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "llama3.2:1b"}]}
        )
        proc = _make_processor(model="phi3:mini")
        assert proc.is_available() is False

    @patch("urllib.request.urlopen")
    def test_not_running(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("refused")
        proc = _make_processor()
        assert proc.is_available() is False

    @patch("urllib.request.urlopen")
    def test_caches_result(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "phi3:mini"}]}
        )
        proc = _make_processor(model="phi3:mini")
        proc.is_available()
        proc.is_available()
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_auto_select_first_model(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "mistral:7b"}, {"name": "phi3:mini"}]}
        )
        proc = _make_processor(model="")
        assert proc.is_available() is True
        assert proc._model == "mistral:7b"

    @patch("urllib.request.urlopen")
    def test_model_matches_with_tag(self, mock_urlopen):
        """Model 'phi3' should match 'phi3:latest' from Ollama."""
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "phi3:latest"}]}
        )
        proc = _make_processor(model="phi3")
        assert proc.is_available() is True


class TestListModels:
    @patch("urllib.request.urlopen")
    def test_returns_model_names(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            {"models": [{"name": "phi3:mini"}, {"name": "mistral:7b"}]}
        )
        proc = _make_processor()
        assert proc.list_models() == ["phi3:mini", "mistral:7b"]

    @patch("urllib.request.urlopen")
    def test_returns_empty_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("refused")
        proc = _make_processor()
        assert proc.list_models() == []


class TestStripEcho:
    def test_here_is_prefix(self):
        assert _strip_echo("Here is the corrected text: Hello.") == "Hello."

    def test_corrected_text_prefix(self):
        assert _strip_echo("Corrected text: Hello.") == "Hello."

    def test_quoted_response(self):
        assert _strip_echo('"Hello world."') == "Hello world."

    def test_no_echo(self):
        assert _strip_echo("Hello world.") == "Hello world."


class TestIntegration:
    """Integration tests requiring a running Ollama instance. Skipped by default."""

    @pytest.fixture(autouse=True)
    def _require_ollama(self):
        proc = _make_processor(model="")
        if not proc.is_available():
            pytest.skip("Ollama not running")
        self.proc = proc

    def test_real_correction(self):
        result = self.proc.process("the skys is blue and the grass are green")
        # Should fix at least "skys" → "sky" or "sky's" and "are" → "is"
        assert "sky" in result.lower()
        assert result != "the skys is blue and the grass are green"

    def test_real_latency(self):
        import time
        t0 = time.monotonic()
        self.proc.process("hello world how are you doing today")
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0  # generous timeout

    def test_real_idempotency(self):
        first = self.proc.process("The sky is blue and the grass is green.")
        second = self.proc.process(first)
        # LLM may introduce minor variations; check they're at least close
        assert len(second) > 0
        # Length ratio should be stable (not hallucinating)
        ratio = len(second) / len(first)
        assert 0.8 < ratio < 1.2
