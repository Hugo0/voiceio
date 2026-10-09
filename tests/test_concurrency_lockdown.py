"""Regression tests for the concurrency lockdown + IBus good-citizenship work.

Covers:
  #1 IBus input-source is never re-forced while IDLE; mid-record switch-away
     falls back to clipboard instead of fighting the user.
  #2 IBus deactivation is generation-checked and restores the user's source.
  #3 Watchdog typer/platform mutation goes through lock-guarded, IDLE-gated
     swap helpers.
  #8 SIGTERM runs the same shutdown path as Ctrl-C.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from voiceio.config import Config
from voiceio.typers.base import TyperBackend


def _make_app(typer_name="clipboard"):
    """VoiceIO with mocked backends; typer.name configurable."""
    mock_hotkey = MagicMock()
    mock_hotkey.name = "socket"
    mock_typer = MagicMock(spec=TyperBackend)
    mock_typer.name = typer_name

    with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as mock_detect:
        mock_detect.return_value = MagicMock(
            display_server="wayland", desktop="gnome", is_gnome=True,
        )
        from voiceio.app import VoiceIO
        vio = VoiceIO(Config())
    mock_stream = MagicMock()
    mock_stream.active = True
    mock_stream.closed = False
    mock_stream.stopped = False
    vio.recorder._stream = mock_stream
    vio.recorder._last_callback_time = time.monotonic()
    return vio


# ── Fix #3: lock-guarded, IDLE-gated swaps ──────────────────────────────

class TestGuardedSwaps:
    def test_swap_typer_only_when_idle(self):
        from voiceio.app import _State
        vio = _make_app()
        new = MagicMock(spec=TyperBackend)
        new.name = "ydotool"

        vio._state = _State.RECORDING
        assert vio._swap_typer(new, "test") is False
        assert vio._typer is not new  # not swapped mid-recording

        vio._state = _State.IDLE
        assert vio._swap_typer(new, "test") is True
        assert vio._typer is new

    def test_swap_platform_only_when_idle(self):
        from voiceio.app import _State
        vio = _make_app()
        original = vio.platform
        new_platform = MagicMock(desktop="kde")

        vio._state = _State.FINALIZING
        assert vio._swap_platform(new_platform) is False
        assert vio.platform is original

        vio._state = _State.IDLE
        assert vio._swap_platform(new_platform) is True
        assert vio.platform is new_platform

    def test_health_upkeep_skipped_when_not_idle(self):
        """_check_health must not run typer upkeep during a live recording."""
        from voiceio.app import _State
        vio = _make_app()
        vio._state = _State.RECORDING
        vio.transcriber.healthy.return_value = True
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.cfg.tray.enabled = False
        vio._health_typer_upkeep = MagicMock()
        vio._check_health()
        vio._health_typer_upkeep.assert_not_called()


# ── Fix #2: generation-checked deactivation + source restore ─────────────
# (Generation semantics and the restore itself live in IBusSession — see
# tests/test_ibus_session.py. Here: the app hands it the right generation.)

class TestIBusDeactivation:
    def test_release_uses_the_daemon_generation(self):
        vio = _make_app(typer_name="ibus")
        vio._generation = 6
        with patch("voiceio.ibus.session.threading.Thread") as MockThread:
            vio._ibus._wanted = 1  # a live claim
            vio._ibus.release(5)  # a newer recording already started
            MockThread.assert_not_called()
            vio._ibus.release(6)
            MockThread.assert_called_once()

    def test_release_noop_without_claim(self):
        """Non-IBus typers never claim, so their releases do nothing."""
        vio = _make_app(typer_name="clipboard")
        with patch("voiceio.ibus.session.threading.Thread") as MockThread:
            vio._ibus.release(vio._generation)
            MockThread.assert_not_called()

    def test_do_stop_does_not_deactivate_synchronously(self):
        """Fix #2: deactivation is deferred to after the final commit, not run
        inside _do_stop (which would race the finalizer's commit)."""
        from voiceio.app import _State
        vio = _make_app(typer_name="ibus")
        vio._ibus = MagicMock()
        vio._state = _State.RECORDING
        vio._record_start = time.monotonic() - 2
        # Give the recorder something to return so a finalizer is spawned.
        vio.recorder.stop = MagicMock(return_value=None)
        vio._session = MagicMock()
        with patch("voiceio.app.threading.Thread"):
            vio._do_stop()
        # For the streaming path the finalizer owns deactivation, so _do_stop
        # itself must not have called it.
        vio._ibus.release.assert_not_called()


# ── Fix #1: no re-force while IDLE; mid-record fallback ──────────────────

class TestIBusGoodCitizen:
    def _idle_healthy_ibus(self):
        from voiceio.app import _State
        vio = _make_app(typer_name="ibus")
        vio._state = _State.IDLE
        vio.transcriber.healthy.return_value = True
        vio.recorder.stream_health = MagicMock(return_value=(True, ""))
        vio.cfg.tray.enabled = False
        vio._health_typer_upkeep = MagicMock()
        vio._ibus._proc = MagicMock()
        vio._ibus._proc.poll.return_value = None  # alive
        vio._ibus.ping = MagicMock(return_value=True)  # responsive
        vio._ibus._fixed_sources = MagicMock()
        return vio

    def test_idle_never_touches_input_source(self):
        vio = self._idle_healthy_ibus()
        vio._ibus.hijacked = MagicMock()
        vio._check_health()
        # A responsive engine while IDLE => no source re-forcing at all.
        assert vio._ibus._fixed_sources.method_calls == []
        vio._ibus.hijacked.assert_not_called()

    def test_recording_watches_for_hijack(self):
        from voiceio.app import _State
        vio = self._idle_healthy_ibus()
        vio._state = _State.RECORDING
        vio._ibus.hijacked = MagicMock(return_value=False)
        vio._check_health()
        vio._ibus.hijacked.assert_called_once()

    def _recording_with_claim(self, switched_away: bool):
        from voiceio.app import _State
        vio = self._idle_healthy_ibus()
        vio._state = _State.RECORDING
        sources = vio._ibus._fixed_sources
        sources.switched_away.return_value = switched_away
        vio._ibus._wanted, vio._ibus._claimer = vio._generation, sources
        vio._session = MagicMock()
        return vio

    def test_hijack_falls_back_to_clipboard(self):
        vio = self._recording_with_claim(switched_away=True)
        with patch("voiceio.typers.clipboard.ClipboardTyper", return_value=MagicMock()):
            vio._check_health()
        vio._session.set_typer.assert_called_once()

    def test_hijack_is_idempotent(self):
        vio = self._recording_with_claim(switched_away=True)
        with patch("voiceio.typers.clipboard.ClipboardTyper", return_value=MagicMock()):
            vio._check_health()
            vio._session.set_typer.reset_mock()
            vio._check_health()  # already fell back
        vio._session.set_typer.assert_not_called()

    def test_no_hijack_when_still_on_voiceio(self):
        vio = self._recording_with_claim(switched_away=False)
        vio._check_health()
        vio._session.set_typer.assert_not_called()


# ── Fix #8: SIGTERM shutdown wiring ─────────────────────────────────────

class TestSigterm:
    def test_run_registers_sigterm(self, monkeypatch, tmp_path):
        import signal as _signal
        vio = _make_app(typer_name="clipboard")

        import voiceio.config as vcfg
        monkeypatch.setattr(vcfg, "PID_PATH", tmp_path / "pid")
        monkeypatch.setattr(vcfg, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr("voiceio.pidlock.lock_pid_file", lambda fd: None)
        # Neutralize anything that could touch the real session/system.
        monkeypatch.setattr("voiceio.app._import_graphical_env", lambda: None)
        monkeypatch.setattr("voiceio.app._redetect_platform", lambda: vio.platform)
        vio._ibus = MagicMock()
        vio._try_upgrade_typer = MagicMock()
        vio.cfg.tray.enabled = False
        vio.cfg.feedback.sound_enabled = False
        vio._hotkey = MagicMock()
        vio._socket = None
        vio._tts_hotkey = None

        handlers = {}
        monkeypatch.setattr(
            "voiceio.app.signal.signal",
            lambda sig, handler: handlers.__setitem__(sig, handler),
        )

        vio._shutdown.set()  # make run()'s wait return immediately
        vio.run()

        assert _signal.SIGTERM in handlers
        # Invoking the handler must trigger the shared shutdown path.
        vio._shutdown.clear()
        handlers[_signal.SIGTERM]()
        assert vio._shutdown.is_set()
        vio._ibus.shutdown.assert_called_once()  # same cleanup path
