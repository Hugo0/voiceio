"""Tests for robustness features: stream health, tray watchdog, audio backoff."""
from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from voiceio.config import Config


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_app_wiring.py)
# ---------------------------------------------------------------------------

def _make_vio(mock_transcriber=None):
    """Create a VoiceIO instance with mocked backends."""
    mock_hotkey = MagicMock()
    mock_hotkey.name = "socket"
    mock_typer = MagicMock()
    mock_typer.name = "clipboard"
    if mock_transcriber is None:
        mock_transcriber = MagicMock()

    with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
         patch("voiceio.app.Transcriber", return_value=mock_transcriber), \
         patch("voiceio.app.plat.detect") as mock_detect:
        mock_detect.return_value = MagicMock(display_server="wayland", desktop="gnome")

        from voiceio.app import VoiceIO
        vio = VoiceIO(Config())
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        vio.recorder._stream = mock_stream
        vio.recorder._last_callback_time = time.monotonic()
        return vio, mock_typer, mock_transcriber


def _feed_audio(vio, chunks=20):
    """Feed fake audio data into the recorder."""
    for _ in range(chunks):
        data = np.full((1024, 1), 0.3, dtype=np.float32)
        vio.recorder._callback(data, 1024, None, None)


def _allow_stop(vio):
    """Set timestamps far back so debounce allows stop."""
    vio._record_start = time.monotonic() - 5.0
    vio._last_hotkey = time.monotonic() - 5.0


def _make_recorder():
    """Create a standalone AudioRecorder with mocked dependencies."""
    cfg = MagicMock()
    cfg.sample_rate = 16000
    cfg.device = "default"
    cfg.prebuffer_secs = 1.0
    cfg.silence_threshold = 0.01
    cfg.silence_duration = 0.6
    cfg.auto_stop_silence_secs = 5.0

    with patch("voiceio.recorder.sd"):
        from voiceio.recorder import AudioRecorder
        vad = MagicMock()
        vad.is_speech.return_value = False
        rec = AudioRecorder(cfg, vad=vad)
    return rec


# ===========================================================================
# 1. stream_health() tests
# ===========================================================================

class TestStreamHealth:
    def test_healthy_stream(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        rec._stream = mock_stream
        rec._last_callback_time = time.monotonic()

        ok, reason = rec.stream_health()
        assert ok is True
        assert reason == ""

    def test_stream_is_none(self):
        rec = _make_recorder()
        rec._stream = None

        ok, reason = rec.stream_health()
        assert ok is False
        assert reason == "stream is None"

    def test_stream_closed(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.closed = True
        rec._stream = mock_stream

        ok, reason = rec.stream_health()
        assert ok is False
        assert reason == "stream closed"

    def test_stream_not_active(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.closed = False
        mock_stream.active = False
        rec._stream = mock_stream

        ok, reason = rec.stream_health()
        assert ok is False
        assert reason == "stream not active"

    def test_stream_stopped(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.closed = False
        mock_stream.active = True
        mock_stream.stopped = True
        rec._stream = mock_stream

        ok, reason = rec.stream_health()
        assert ok is False
        assert reason == "stream stopped"

    def test_stale_heartbeat(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        rec._stream = mock_stream
        # Set heartbeat to 10 seconds ago (exceeds _HEARTBEAT_TIMEOUT of 5s)
        rec._last_callback_time = time.monotonic() - 10.0

        ok, reason = rec.stream_health()
        assert ok is False
        assert "no audio callback for" in reason

    def test_healthy_with_no_heartbeat_yet(self):
        """Before any callback fires, heartbeat is 0.0 — should be healthy."""
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        rec._stream = mock_stream
        rec._last_callback_time = 0.0  # never set

        ok, reason = rec.stream_health()
        assert ok is True
        assert reason == ""


# ===========================================================================
# 2. has_signal() tests
# ===========================================================================

class TestHasSignal:
    def test_empty_ring_buffer(self):
        rec = _make_recorder()
        # Ring buffer is empty by default (no audio fed)
        assert rec.has_signal() is False

    def test_all_zeros(self):
        rec = _make_recorder()
        # Feed silent audio (all zeros)
        silent = np.zeros((1024, 1), dtype=np.float32)
        rec._ring.append(silent)

        assert rec.has_signal() is False

    def test_real_audio(self):
        rec = _make_recorder()
        # Feed audio with real signal
        audio = np.full((1024, 1), 0.05, dtype=np.float32)
        rec._ring.append(audio)

        assert rec.has_signal() is True

    def test_barely_above_threshold(self):
        """Signal just above 1e-4 threshold should be detected."""
        rec = _make_recorder()
        audio = np.full((1024, 1), 2e-4, dtype=np.float32)
        rec._ring.append(audio)

        assert rec.has_signal() is True

    def test_barely_below_threshold(self):
        """Signal at exactly 1e-4 or below should not be detected."""
        rec = _make_recorder()
        audio = np.full((1024, 1), 1e-5, dtype=np.float32)
        rec._ring.append(audio)

        assert rec.has_signal() is False


# ===========================================================================
# 3. reopen_stream() tests
# ===========================================================================

class TestReopenStream:
    def test_closes_and_reopens(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        rec._stream = mock_stream
        rec._last_callback_time = time.monotonic()

        with patch("voiceio.recorder.sd") as mock_sd:
            new_stream = MagicMock()
            mock_sd.InputStream.return_value = new_stream
            rec.reopen_stream()

        # Old stream should have been stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        # New stream should be opened and started
        new_stream.start.assert_called_once()
        assert rec._stream is new_stream

    def test_resets_heartbeat(self):
        rec = _make_recorder()
        mock_stream = MagicMock()
        rec._stream = mock_stream
        rec._last_callback_time = time.monotonic() - 100.0

        with patch("voiceio.recorder.sd") as mock_sd:
            mock_sd.InputStream.return_value = MagicMock()
            rec.reopen_stream()

        assert rec._last_callback_time == 0.0


# ===========================================================================
# 4. tray is_alive() tests
# ===========================================================================

class TestTrayIsAlive:
    def test_indicator_alive(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        with patch("voiceio.tray._proc", mock_proc), \
             patch("voiceio.tray._backend", "indicator"):
            from voiceio import tray
            assert tray.is_alive() is True

    def test_indicator_dead(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited with code 1

        with patch("voiceio.tray._proc", mock_proc), \
             patch("voiceio.tray._backend", "indicator"):
            from voiceio import tray
            assert tray.is_alive() is False

    def test_indicator_proc_none(self):
        with patch("voiceio.tray._proc", None), \
             patch("voiceio.tray._backend", "indicator"):
            from voiceio import tray
            assert tray.is_alive() is False

    def test_pystray_alive(self):
        with patch("voiceio.tray._backend", "pystray"):
            from voiceio import tray
            assert tray.is_alive() is True

    def test_no_backend(self):
        with patch("voiceio.tray._backend", None):
            from voiceio import tray
            assert tray.is_alive() is False


# ===========================================================================
# 5. Health watchdog audio backoff tests
# ===========================================================================

class TestHealthWatchdogAudioBackoff:
    def test_stream_failure_triggers_reopen(self):
        vio, _, _ = _make_vio()
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock()
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray"):
            vio._check_health()

        vio.recorder.reopen_stream.assert_called_once()

    def test_successful_recovery_resets_backoff(self):
        vio, _, _ = _make_vio()
        vio._stream_fail_count = 3
        vio._next_stream_retry = 0  # allow retry
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock()  # succeeds (no exception)
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray"):
            vio._check_health()

        assert vio._stream_fail_count == 0
        assert vio._next_stream_retry == 0

    def test_failed_recovery_increments_backoff(self):
        vio, _, _ = _make_vio()
        vio._stream_fail_count = 0
        vio._next_stream_retry = 0
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock(side_effect=OSError("device gone"))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray"):
            vio._check_health()

        assert vio._stream_fail_count == 1
        assert vio._next_stream_retry > 0

    def test_backoff_skips_retry(self):
        """When in backoff, _check_health should not attempt reopen."""
        vio, _, _ = _make_vio()
        vio._stream_fail_count = 2
        vio._next_stream_retry = time.monotonic() + 999  # far future
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock()
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray"):
            vio._check_health()

        vio.recorder.reopen_stream.assert_not_called()

    def test_external_recovery_resets_backoff(self):
        """If stream becomes healthy on its own, backoff resets."""
        vio, _, _ = _make_vio()
        vio._stream_fail_count = 3
        vio._next_stream_retry = time.monotonic() + 100
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray"):
            vio._check_health()

        assert vio._stream_fail_count == 0
        assert vio._next_stream_retry == 0

    def test_repeated_failures_increase_delay(self):
        """Each failure should increase the backoff delay."""
        vio, _, _ = _make_vio()
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock(side_effect=OSError("device gone"))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        delays = []
        with patch("voiceio.app.tray"):
            for _ in range(3):
                vio._next_stream_retry = 0  # allow retry each time
                before = time.monotonic()
                vio._check_health()
                delays.append(vio._next_stream_retry - before)

        # Each delay should be larger (10, 20, 40)
        assert delays[0] < delays[1] < delays[2]


# ===========================================================================
# 6. Pre-flight stream check in _do_start()
# ===========================================================================

class TestPreFlightStreamCheck:
    def test_unhealthy_stream_gets_reopened(self):
        vio, _, _ = _make_vio()
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock()
        vio.recorder.has_signal = MagicMock(return_value=True)

        vio.on_hotkey()

        vio.recorder.reopen_stream.assert_called_once()
        assert vio.recorder.is_recording

    def test_recording_aborts_if_reopen_fails(self):
        from voiceio.app import _State

        vio, _, _ = _make_vio()
        vio.recorder.stream_health = MagicMock(return_value=(False, "stream not active"))
        vio.recorder.reopen_stream = MagicMock(side_effect=OSError("no device"))

        vio.on_hotkey()

        # Recording should NOT have started
        assert not vio.recorder.is_recording
        assert vio._state == _State.IDLE

    def test_healthy_stream_no_reopen(self):
        vio, _, _ = _make_vio()
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.recorder.reopen_stream = MagicMock()
        vio.recorder.has_signal = MagicMock(return_value=True)

        vio.on_hotkey()

        vio.recorder.reopen_stream.assert_not_called()
        assert vio.recorder.is_recording


# ===========================================================================
# 7. Streaming worker exception handling
# ===========================================================================

class TestStreamingWorkerExceptionHandling:
    def test_worker_catches_exception_and_continues(self):
        """Worker loop should catch exceptions in _transcribe_and_apply and keep running."""
        mock_transcriber = MagicMock()
        mock_typer = MagicMock()
        mock_typer.name = "clipboard"
        mock_recorder = MagicMock()
        mock_recorder.sample_rate = 16000
        mock_recorder.get_audio_so_far.return_value = np.zeros(32000, dtype=np.float32)

        from voiceio.streaming import StreamingSession

        session = StreamingSession(
            mock_transcriber, mock_typer, mock_recorder,
            generation=0,
        )

        call_count = 0

        def flaky_transcribe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient error")
            # On 3rd call, stop the session to let the test finish
            session._stop_event.set()

        session._transcribe_and_apply = flaky_transcribe

        # Start the worker (runs in its own thread)
        session._stop_event = threading.Event()
        session._pending = threading.Event()
        session._pending.set()  # trigger immediate processing

        worker = threading.Thread(target=session._worker_loop, daemon=True)
        worker.start()

        # Wait for the worker to process through the exceptions
        worker.join(timeout=5)
        assert not worker.is_alive(), "Worker thread should have exited"
        # Worker called our function multiple times without dying
        assert call_count >= 2, f"Expected at least 2 calls, got {call_count}"

    def test_final_transcription_exception_does_not_crash(self):
        """Exception during final transcription should be caught."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = RuntimeError("model crash")
        mock_typer = MagicMock()
        mock_typer.name = "clipboard"
        mock_recorder = MagicMock()
        mock_recorder.sample_rate = 16000
        mock_recorder.get_audio_so_far.return_value = np.zeros(32000, dtype=np.float32)

        from voiceio.streaming import StreamingSession

        session = StreamingSession(
            mock_transcriber, mock_typer, mock_recorder,
            generation=0,
        )

        # Set up for immediate stop (skip the main loop, just do final pass)
        session._stop_event.set()
        session._pending.set()
        session._final_audio = np.zeros(32000, dtype=np.float32)

        # Should not raise — exceptions caught inside _worker_loop
        worker = threading.Thread(target=session._worker_loop, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "Worker thread should have exited"


# ===========================================================================
# 8. Tray watchdog in _check_health()
# ===========================================================================

class TestTrayWatchdog:
    def test_tray_restart_when_dead(self):
        vio, _, _ = _make_vio()
        vio.cfg.tray.enabled = True
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray") as mock_tray:
            mock_tray.is_alive.return_value = False
            vio._check_health()
            mock_tray.restart.assert_called_once_with(vio.on_hotkey)

    def test_tray_not_restarted_when_alive(self):
        vio, _, _ = _make_vio()
        vio.cfg.tray.enabled = True
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray") as mock_tray:
            mock_tray.is_alive.return_value = True
            vio._check_health()
            mock_tray.restart.assert_not_called()

    def test_tray_not_checked_when_disabled(self):
        vio, _, _ = _make_vio()
        vio.cfg.tray.enabled = False
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)

        with patch("voiceio.app.tray") as mock_tray:
            vio._check_health()
            mock_tray.is_alive.assert_not_called()


# ===========================================================================
# 9. Boot-race and typer re-probe (regression tests for 0.3.6 fix)
# ===========================================================================

_linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Boot-race scenario is Linux-specific "
           "(macOS/Windows detect display server from sys.platform)",
)


class TestBootRaceAndReProbe:
    """Tests for the fix where platform.detect's lru_cache froze a stale
    display=unknown result at boot, defeating all re-probe logic."""

    @_linux_only
    def test_redetect_clears_lru_cache(self):
        """_redetect_platform must call cache_clear so env changes take effect."""
        from voiceio.app import _redetect_platform
        from voiceio import platform as plat

        # Warm the cache with display=unknown (simulating boot before env ready)
        with patch.dict("os.environ", {}, clear=True):
            plat.detect.cache_clear()
            p1 = plat.detect()
            assert p1.display_server == "unknown"

        # Env vars appear. Plain detect() would return cached unknown;
        # _redetect_platform must clear cache and return the fresh values.
        with patch.dict("os.environ", {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_CURRENT_DESKTOP": "GNOME",
        }):
            p2 = _redetect_platform()
            assert p2.display_server == "wayland"
            assert p2.desktop == "gnome"

        plat.detect.cache_clear()

    def test_import_graphical_env_is_one_shot(self):
        """After all vars are present, _import_graphical_env becomes a no-op
        (no subprocess every health-check cycle)."""
        from voiceio import app as app_mod
        from voiceio.app import _import_graphical_env

        app_mod._graphical_env_complete = False

        with patch.dict("os.environ", {
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_SESSION_TYPE": "wayland",
            "XDG_CURRENT_DESKTOP": "GNOME",
            "XDG_SESSION_DESKTOP": "gnome",
        }):
            with patch("voiceio.app.subprocess.check_output") as mock_sub:
                _import_graphical_env()
                assert app_mod._graphical_env_complete is True
                mock_sub.assert_not_called()  # all vars present → no subprocess
                _import_graphical_env()
                mock_sub.assert_not_called()  # still no subprocess

        app_mod._graphical_env_complete = False

    def test_try_upgrade_typer_upgrades_from_clipboard_to_ibus(self):
        """When chain has ibus first and current is clipboard, upgrade."""
        vio, _, _ = _make_vio()
        vio.platform = MagicMock(display_server="wayland", desktop="gnome")
        vio._typer = MagicMock()
        vio._typer.name = "clipboard"

        # Make the ClipboardTyper isinstance check false (we're using MagicMock)
        better = MagicMock()
        better.name = "ibus"

        with patch("voiceio.app.typer_chain.resolve", return_value=[
            ("ibus", better, MagicMock(ok=True)),
            ("clipboard", vio._typer, MagicMock(ok=True)),
        ]), patch("voiceio.app.typer_chain.select", return_value=better), \
             patch("voiceio.app.typer_chain._get_chain",
                   return_value=["ibus", "clipboard"]), \
             patch.object(vio, "_ensure_ibus_engine") as mock_ensure:
            result = vio._try_upgrade_typer(reason="test")
            assert result is True
            assert vio._typer is better
            mock_ensure.assert_called_once()

    def test_try_upgrade_typer_no_upgrade_when_already_best(self):
        """If already on the first entry in the chain, no upgrade."""
        vio, _, _ = _make_vio()
        vio.platform = MagicMock(display_server="wayland", desktop="gnome")
        vio._typer = MagicMock()
        vio._typer.name = "ibus"

        with patch("voiceio.app.typer_chain._get_chain",
                   return_value=["ibus", "clipboard"]):
            result = vio._try_upgrade_typer(reason="test")
            assert result is False

    def test_try_upgrade_typer_no_upgrade_when_unknown_platform(self):
        """Regression: with display=unknown the chain is ['clipboard'], and
        upgrade should early-return False (this was silent and defeated the
        health loop before the lru_cache fix)."""
        vio, _, _ = _make_vio()
        vio.platform = MagicMock(display_server="unknown", desktop="unknown")
        vio._typer = MagicMock()
        vio._typer.name = "clipboard"

        with patch("voiceio.app.typer_chain._get_chain",
                   return_value=["clipboard"]):
            result = vio._try_upgrade_typer(reason="test")
            assert result is False

    def test_boot_race_healed_by_health_check(self):
        """Simulate: __init__ ran with display=unknown and picked clipboard.
        Later, env vars become available and health check should upgrade."""
        from voiceio import app as app_mod
        from voiceio import platform as plat

        vio, _, _ = _make_vio()
        # Simulate the stale state as if we started at boot
        vio.platform = MagicMock(display_server="unknown", desktop="unknown")
        vio._typer = MagicMock()
        vio._typer.name = "clipboard"
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.transcriber.is_worker_alive = MagicMock(return_value=True)
        vio.cfg.tray.enabled = False

        # Env arrives; health check should re-import, re-detect, and upgrade.
        better = MagicMock()
        better.name = "ibus"
        better.probe.return_value = MagicMock(ok=True)

        # Force env import to go through; reset one-shot flag
        app_mod._graphical_env_complete = False

        fresh_platform = MagicMock(display_server="wayland", desktop="gnome")

        with patch.dict("os.environ", {
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_SESSION_TYPE": "wayland",
            "XDG_CURRENT_DESKTOP": "GNOME",
            "XDG_SESSION_DESKTOP": "gnome",
        }), patch("voiceio.app._redetect_platform", return_value=fresh_platform), \
             patch("voiceio.app.typer_chain._get_chain",
                   return_value=["ibus", "clipboard"]), \
             patch("voiceio.app.typer_chain.resolve", return_value=[
                 ("ibus", better, MagicMock(ok=True)),
             ]), patch("voiceio.app.typer_chain.select", return_value=better), \
             patch.object(vio, "_ensure_ibus_engine"):
            vio._check_health()

        assert vio._typer is better
        assert vio.platform is fresh_platform
        plat.detect.cache_clear()
        app_mod._graphical_env_complete = False

    def test_on_typer_broken_defers_until_idle(self):
        """Mid-recording, _on_typer_broken must NOT hot-swap the typer."""
        from voiceio.app import _State

        vio, _, _ = _make_vio()
        vio._state = _State.RECORDING
        original_typer = vio._typer

        # Call the handler directly (no thread) to check it respects state.
        # The real handler kicks off a thread; we test the internal method.
        with patch.object(vio, "_try_upgrade_typer") as mock_upgrade:
            # Simulate the thread having already waited: state is still RECORDING
            # Force deadline-past by calling with state still RECORDING
            # We invoke _deferred_typer_upgrade with a very short timeout
            # by mocking time.monotonic and time.sleep.
            with patch("voiceio.app.time.monotonic",
                       side_effect=[0, 100, 100, 100]), \
                 patch("voiceio.app.time.sleep"):
                vio._deferred_typer_upgrade()
            mock_upgrade.assert_not_called()
        assert vio._typer is original_typer

    def test_on_typer_broken_runs_when_idle(self):
        """Once IDLE is reached, the deferred upgrade runs."""
        from voiceio.app import _State

        vio, _, _ = _make_vio()
        vio._state = _State.IDLE

        with patch.object(vio, "_try_upgrade_typer") as mock_upgrade, \
             patch("voiceio.app._import_graphical_env"), \
             patch("voiceio.app._redetect_platform",
                   return_value=vio.platform), \
             patch("voiceio.app.time.sleep"):
            vio._deferred_typer_upgrade()
            mock_upgrade.assert_called_once_with(reason="streaming-failure")
