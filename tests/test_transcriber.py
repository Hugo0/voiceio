"""Tests for transcriber subprocess protocol."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from voiceio.config import ModelConfig


class TestTranscriberProtocol:
    """Test the subprocess communication protocol without real whisper."""

    def test_read_with_timeout_returns_none_on_timeout(self):
        from voiceio.transcriber import Transcriber

        # Mock the worker startup
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        # Now test timeout: readline blocks forever
        import threading
        def block_forever():
            threading.Event().wait()

        mock_proc.stdout.readline.side_effect = block_forever
        result = t._read_with_timeout(0.1)
        assert result is None

    def test_crash_recovery_flag(self):
        from voiceio.transcriber import Transcriber, MAX_RESTARTS

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        assert t._restarts == 0

        # Simulate worker death
        mock_proc.poll.return_value = 1  # process exited

        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            t._ensure_worker()

        assert t._restarts == 1

    def test_max_restarts_raises(self):
        from voiceio.transcriber import Transcriber, MAX_RESTARTS

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        t._restarts = MAX_RESTARTS
        mock_proc.poll.return_value = 1

        with pytest.raises(RuntimeError, match="crashed"):
            t._ensure_worker()
