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
        # Recent crash: budget must NOT reset, so it stays dead.
        import time as _time
        t._last_crash_time = _time.monotonic()

        with pytest.raises(RuntimeError, match="crashed"):
            t._ensure_worker()


class TestTimeoutScaling:
    """Fix #9: read timeout scales with audio length (never kill long audio)."""

    def test_floor_for_short_audio(self):
        from voiceio.transcriber import transcribe_timeout, TRANSCRIBE_TIMEOUT
        assert transcribe_timeout(1.0) == TRANSCRIBE_TIMEOUT
        assert transcribe_timeout(0.0) == TRANSCRIBE_TIMEOUT

    def test_scales_above_floor_for_long_audio(self):
        from voiceio.transcriber import transcribe_timeout
        # 200s dictation -> 200 * 1.5 = 300s, well above the 30s floor.
        assert transcribe_timeout(200.0) == 300.0
        assert transcribe_timeout(60.0) == 90.0


class TestRestartBudgetReset:
    """Fix #10: a sustained crash-free period resets the restart budget."""

    def _make(self):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            from voiceio.transcriber import Transcriber
            t = Transcriber(ModelConfig())
        return t, mock_proc

    def test_healthy_period_resets_counter(self):
        from voiceio.transcriber import MAX_RESTARTS, RESTART_RESET_SECS
        import time as _time
        t, mock_proc = self._make()

        # Simulate having already hit the ceiling long ago. Patch monotonic
        # instead of subtracting from it: on a freshly-booted machine (CI),
        # monotonic() is small and the subtraction would go negative,
        # disabling the reset guard entirely.
        t._restarts = MAX_RESTARTS
        t._last_crash_time = 1.0
        mock_proc.poll.return_value = 1  # worker died again

        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc), \
                patch("voiceio.transcriber.time") as mock_time:
            mock_time.monotonic.return_value = 1.0 + RESTART_RESET_SECS + 60
            t._ensure_worker()  # must NOT raise — counter reset then incremented

        assert t._restarts == 1

    def test_rapid_crashes_still_give_up(self):
        from voiceio.transcriber import MAX_RESTARTS
        import time as _time
        t, mock_proc = self._make()

        t._restarts = MAX_RESTARTS
        t._last_crash_time = _time.monotonic()  # just crashed — no reset
        mock_proc.poll.return_value = 1

        with pytest.raises(RuntimeError, match="crashed"):
            t._ensure_worker()


class TestTimeoutRaises:
    """Timeout/invalid response raise TranscriptionError (never fake silence)."""

    def _make(self):
        from voiceio.transcriber import Transcriber
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        with patch("voiceio.transcriber.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())
        return t, mock_proc

    def test_timeout_raises_and_clears_segments(self):
        import numpy as np
        import pytest
        from voiceio.transcriber import TranscriptionError
        t, mock_proc = self._make()
        t.last_segments = [{"text": "stale"}]
        with patch.object(t, "_read_with_timeout", return_value=None), \
             patch.object(t, "_kill_worker"), patch.object(t, "_ensure_worker"):
            with pytest.raises(TranscriptionError):
                t.transcribe(np.zeros(16000, dtype=np.float32))
        assert t.last_segments == []

    def test_invalid_response_raises_and_clears_segments(self):
        import numpy as np
        import pytest
        from voiceio.transcriber import TranscriptionError
        t, mock_proc = self._make()
        t.last_segments = [{"text": "stale"}]
        with patch.object(t, "_read_with_timeout", return_value="not json{"):
            with pytest.raises(TranscriptionError):
                t.transcribe(np.zeros(16000, dtype=np.float32))
        assert t.last_segments == []
