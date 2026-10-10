"""Tests for transcriber subprocess protocol."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from voiceio.config import ModelConfig


class TestTranscriberProtocol:
    """Test the subprocess communication protocol without real whisper."""

    def test_read_with_timeout_returns_none_on_timeout(self):
        from voiceio.core.decoders.faster_whisper import Transcriber

        # Mock the worker startup
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        # Now test timeout: readline blocks forever
        import threading
        def block_forever():
            threading.Event().wait()

        mock_proc.stdout.readline.side_effect = block_forever
        result = t._read_with_timeout(0.1)
        assert result is None

    def test_crash_recovery_flag(self):
        from voiceio.core.decoders.faster_whisper import Transcriber, MAX_RESTARTS

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        assert t._restarts == 0

        # Simulate worker death
        mock_proc.poll.return_value = 1  # process exited

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            t._ensure_worker()

        assert t._restarts == 1

    def test_max_restarts_raises(self):
        from voiceio.core.decoders.faster_whisper import Transcriber, MAX_RESTARTS

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
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
        from voiceio.core.decoders.base import TRANSCRIBE_TIMEOUT, transcribe_timeout
        assert transcribe_timeout(1.0) == TRANSCRIBE_TIMEOUT
        assert transcribe_timeout(0.0) == TRANSCRIBE_TIMEOUT

    def test_scales_above_floor_for_long_audio(self):
        from voiceio.core.decoders.base import transcribe_timeout
        # 200s dictation -> 200 * 1.5 = 300s, well above the 30s floor.
        assert transcribe_timeout(200.0) == 600.0
        assert transcribe_timeout(60.0) == 180.0


class TestRestartBudgetReset:
    """Fix #10: a sustained crash-free period resets the restart budget."""

    def _make(self):
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            from voiceio.core.decoders.faster_whisper import Transcriber
            t = Transcriber(ModelConfig())
        return t, mock_proc

    def test_healthy_period_resets_counter(self):
        from voiceio.core.decoders.faster_whisper import MAX_RESTARTS, RESTART_RESET_SECS
        import time as _time
        t, mock_proc = self._make()

        # Simulate having already hit the ceiling long ago. Patch monotonic
        # instead of subtracting from it: on a freshly-booted machine (CI),
        # monotonic() is small and the subtraction would go negative,
        # disabling the reset guard entirely.
        t._restarts = MAX_RESTARTS
        t._last_crash_time = 1.0
        mock_proc.poll.return_value = 1  # worker died again

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc), \
                patch("voiceio.core.decoders.faster_whisper.time") as mock_time:
            mock_time.monotonic.return_value = 1.0 + RESTART_RESET_SECS + 60
            t._ensure_worker()  # must NOT raise — counter reset then incremented

        assert t._restarts == 1

    def test_rapid_crashes_still_give_up(self):
        from voiceio.core.decoders.faster_whisper import MAX_RESTARTS
        import time as _time
        t, mock_proc = self._make()

        t._restarts = MAX_RESTARTS
        t._last_crash_time = _time.monotonic()  # just crashed — no reset
        mock_proc.poll.return_value = 1

        with pytest.raises(RuntimeError, match="crashed"):
            t._ensure_worker()


class TestTimeoutRaises:
    """Timeout/invalid response raise DecodeError (never fake silence)."""

    def _make(self):
        from voiceio.core.decoders.faster_whisper import Transcriber
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())
        return t, mock_proc

    def test_timeout_raises(self):
        import numpy as np
        import pytest
        from voiceio.core.decoders.faster_whisper import DecodeError
        t, mock_proc = self._make()
        with patch.object(t, "_read_with_timeout", return_value=None), \
             patch.object(t, "_kill_worker"), patch.object(t, "_ensure_worker"):
            with pytest.raises(DecodeError):
                t.decode(np.zeros(16000, dtype=np.float32))

    def test_invalid_response_raises(self):
        import numpy as np
        import pytest
        from voiceio.core.decoders.faster_whisper import DecodeError
        t, mock_proc = self._make()
        with patch.object(t, "_read_with_timeout", return_value="not json{"):
            with pytest.raises(DecodeError):
                t.decode(np.zeros(16000, dtype=np.float32))


class TestWorkerStartTimeout:
    """A worker that never reports READY must not wedge the daemon.

    faster-whisper's model load contacts huggingface_hub even for a cached
    model; on a blackholed route that connect hangs indefinitely. An unbounded
    READY handshake then holds the transcribe lock forever and every later
    dictation silently returns "".
    """

    def test_start_raises_instead_of_hanging(self):
        import threading
        from voiceio.core.decoders.faster_whisper import Transcriber, DecodeError

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = lambda: threading.Event().wait()

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc), \
             patch("voiceio.core.decoders.faster_whisper.WORKER_START_TIMEOUT", 0.1):
            with pytest.raises(DecodeError, match="failed to start"):
                Transcriber(ModelConfig())

        # The hung process is reaped, not leaked.
        assert mock_proc.terminate.called or mock_proc.kill.called

    def test_start_raises_on_garbage_handshake(self):
        from voiceio.core.decoders.faster_whisper import Transcriber, DecodeError

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.return_value = "Traceback: boom\n"

        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(DecodeError, match="failed to start"):
                Transcriber(ModelConfig())


class TestDecodeTimeoutIsLazy:
    """The timed-out call fails regardless, so reloading the model inline would
    hold the lock through a multi-second start and starve the final pass."""

    def test_decode_timeout_does_not_reload_inline(self):
        import numpy as np
        from voiceio.core.decoders.faster_whisper import Transcriber, DecodeError

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = "READY\n"
        mock_proc.poll.return_value = None
        with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=mock_proc):
            t = Transcriber(ModelConfig())

        with patch.object(t, "_read_with_timeout", return_value=None), \
             patch.object(t, "_kill_worker") as kill, \
             patch.object(t, "_ensure_worker") as ensure:
            with pytest.raises(DecodeError):
                t.decode(np.zeros(16000, dtype=np.float32))

        assert kill.called
        # Only the entry call — never a second, restarting one.
        assert ensure.call_count == 1


class TestWorkerModelLoad:
    """A cached model must load without touching the network."""

    def test_prefers_local_cache(self):
        from voiceio.core.decoders.faster_whisper import worker

        with patch("voiceio.core.decoders.faster_whisper.worker.WhisperModel") as wm:
            worker.load_model("small", device="auto", compute_type="int8")
        assert wm.call_args.kwargs["local_files_only"] is True

    def test_falls_back_to_network_when_not_cached(self):
        from voiceio.core.decoders.faster_whisper import worker

        sentinel = object()
        with patch("voiceio.core.decoders.faster_whisper.worker.WhisperModel",
                   side_effect=[OSError("not cached"), sentinel]) as wm:
            got = worker.load_model("small")
        assert got is sentinel
        assert "local_files_only" not in wm.call_args.kwargs

    def test_every_whisper_load_goes_through_the_guarded_helper(self):
        """A raw WhisperModel(...) anywhere else reintroduces the hang: the
        teacher audit and the wizard both load models on this same box."""
        import re
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "voiceio"
        offenders = []
        for py in src.rglob("*.py"):
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if re.search(r"(?<!def )\bWhisperModel\(", line):
                    # worker.load_model is the one sanctioned construction site.
                    if py.name == "worker.py":
                        continue
                    offenders.append(f"{py.relative_to(src)}:{i}")
        assert not offenders, f"unguarded WhisperModel loads: {offenders}"


def test_describe_transcript_never_contains_text():
    from voiceio.privacy import describe_transcript
    assert describe_transcript("my secret plan") == "3 words"
    assert describe_transcript("") == "(silence)"


def test_promptless_distil_models_take_no_bias():
    from voiceio.config import ModelConfig
    from voiceio.core.decoders.base import BiasMode
    from voiceio.core.decoders.faster_whisper import Transcriber
    assert Transcriber(ModelConfig(name="distil-small.en")).capabilities.bias == BiasMode.NONE
    assert Transcriber(ModelConfig(name="distil-large-v3")).capabilities.bias != BiasMode.NONE
    assert Transcriber(ModelConfig(name="small")).capabilities.bias != BiasMode.NONE


def test_no_log_call_formats_the_transcript_itself():
    """The daemon writes DEBUG to ~/.local/state/voiceio/voiceio.log, so even
    debug lines must not carry the user's words: no log call may pass a
    transcript variable (or its ["text"]) straight through as an argument."""
    import ast
    from pathlib import Path

    import voiceio
    names = {"text", "new_text", "final_text", "to_type", "_typed_text", "result",
             "corrected", "hyp", "transcript", "msg"}

    def leaks(node):
        if isinstance(node, ast.BoolOp):
            return any(leaks(v) for v in node.values)
        if isinstance(node, ast.JoinedStr):
            return any(leaks(v.value) for v in node.values if isinstance(v, ast.FormattedValue))
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Slice):          # text[:60] is still the words
                return leaks(node.value)
            return isinstance(key, ast.Constant) and key.value == "text"
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "strip":
            return leaks(node.func.value)
        return (isinstance(node, ast.Name) and node.id in names) or \
            (isinstance(node, ast.Attribute) and node.attr in names)

    hits = []
    for path in Path(voiceio.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and getattr(node.func.value, "id", "") == "log"
                    and any(leaks(a) for a in node.args[1:])):
                hits.append(f"{path.name}:{node.lineno}")
    assert not hits, hits
