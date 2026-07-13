"""Test that VoiceIO app wires up correctly with mocked backends."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voiceio.config import Config


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
        cfg = Config()
        # Tests must never touch the real system clipboard
        cfg.output.copy_to_clipboard = "off"
        vio = VoiceIO(cfg)
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_stream.closed = False
        mock_stream.stopped = False
        vio.recorder._stream = mock_stream  # skip real audio
        vio.recorder._last_callback_time = time.monotonic()  # healthy heartbeat
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


class TestVoiceIOInit:
    def test_init_with_mocked_backends(self):
        vio, mock_typer, _ = _make_vio()
        assert vio._typer is mock_typer

    def test_on_hotkey_toggle_cycle(self):
        mock_trans = MagicMock()
        mock_trans.transcribe.return_value = "hello world"
        vio, _, _ = _make_vio(mock_trans)

        # Start recording
        vio.on_hotkey()
        assert vio.recorder.is_recording

        _feed_audio(vio)
        _allow_stop(vio)

        # Stop recording — recorder stops synchronously now
        vio.on_hotkey()
        assert not vio.recorder.is_recording


class TestHotkeyDebounce:
    """Verify that duplicate hotkey events are properly debounced."""

    def test_rapid_duplicate_ignored(self):
        """Two on_hotkey calls within the debounce window should only trigger once."""
        vio, _, _ = _make_vio()

        vio.on_hotkey()
        assert vio.recorder.is_recording

        # Simulate duplicate from socket backend ~50ms later
        vio.on_hotkey()
        # Should still be recording (duplicate was debounced, not treated as stop)
        assert vio.recorder.is_recording

    def test_stop_after_debounce_window(self):
        """on_hotkey after debounce window should stop recording."""
        vio, _, _ = _make_vio()

        vio.on_hotkey()
        assert vio.recorder.is_recording

        _feed_audio(vio)
        _allow_stop(vio)

        vio.on_hotkey()
        # Recorder stops synchronously in the new design
        assert not vio.recorder.is_recording

    def test_concurrent_hotkey_no_phantom_recording(self):
        """Socket event waiting behind lock must not start phantom recording.

        This is the critical race: evdev stops recording (takes time),
        socket event waits on lock, then must be debounced when lock releases.
        """
        vio, _, _ = _make_vio()

        # Start recording
        vio.on_hotkey()
        assert vio.recorder.is_recording

        _feed_audio(vio)
        _allow_stop(vio)

        # Simulate: evdev thread stops recording, socket thread waits then fires
        results = []

        def evdev_stop():
            vio.on_hotkey()
            results.append(("evdev", vio.recorder.is_recording))

        def socket_delayed():
            time.sleep(0.05)  # socket arrives 50ms after evdev
            vio.on_hotkey()
            results.append(("socket", vio.recorder.is_recording))

        t1 = threading.Thread(target=evdev_stop)
        t2 = threading.Thread(target=socket_delayed)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Recording must be stopped and NOT restarted by socket
        assert not vio.recorder.is_recording, \
            f"Phantom recording started! results={results}"


class TestStateMachine:
    """Test the explicit state machine transitions."""

    def test_idle_to_recording(self):
        from voiceio.app import _State
        vio, _, _ = _make_vio()
        assert vio._state == _State.IDLE

        vio.on_hotkey()
        assert vio._state == _State.RECORDING
        assert vio.recorder.is_recording

    def test_recording_to_idle_batch(self):
        """Non-streaming stop goes directly to IDLE."""
        from voiceio.app import _State
        vio, _, _ = _make_vio()
        vio._streaming = False

        vio.on_hotkey()
        assert vio._state == _State.RECORDING

        _feed_audio(vio)
        _allow_stop(vio)

        vio.on_hotkey()
        assert vio._state == _State.IDLE
        assert not vio.recorder.is_recording

    def test_generation_increments_on_stop(self):
        vio, _, _ = _make_vio()
        gen_before = vio._generation

        vio.on_hotkey()
        _feed_audio(vio)
        _allow_stop(vio)
        vio.on_hotkey()

        assert vio._generation == gen_before + 1

    def test_can_start_during_finalizing(self):
        """User can start a new recording while old one finalizes."""
        from voiceio.app import _State
        mock_trans = MagicMock()
        mock_trans.transcribe.return_value = "hello"
        vio, _, _ = _make_vio(mock_trans)

        # Start and stop (enters FINALIZING for streaming mode)
        vio.on_hotkey()
        _feed_audio(vio)
        _allow_stop(vio)
        vio.on_hotkey()
        assert not vio.recorder.is_recording

        # Immediately start new recording (should work even if finalizing)
        vio._last_hotkey = time.monotonic() - 5.0
        vio.on_hotkey()
        assert vio._state == _State.RECORDING
        assert vio.recorder.is_recording

    def test_auto_stop_never_starts_recording(self):
        """_request_stop only stops, never starts."""
        from voiceio.app import _State
        vio, _, _ = _make_vio()
        assert vio._state == _State.IDLE

        # _request_stop when idle should do nothing
        vio._request_stop()
        assert vio._state == _State.IDLE
        assert not vio.recorder.is_recording


class TestClipboardMirror:
    """copy_to_clipboard: interim/final text mirrored to the clipboard."""

    def test_interim_copy_skipped_for_clipboard_typer(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "clipboard"
        vio.cfg.output.copy_to_clipboard = "live"
        with patch("voiceio.clipboard_read.copy_text") as mock_copy:
            vio._on_interim_text("hello")
            mock_copy.assert_not_called()

    def test_interim_copy_fires_for_other_typers(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        vio.cfg.output.copy_to_clipboard = "live"
        with patch("voiceio.clipboard_read.copy_text") as mock_copy:
            vio._on_interim_text("hello")
            mock_copy.assert_called_once_with("hello")

    def test_interim_copy_off_unless_live(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        vio.cfg.output.copy_to_clipboard = "final"
        with patch("voiceio.clipboard_read.copy_text") as mock_copy:
            vio._on_interim_text("hello")
            mock_copy.assert_not_called()

    def test_final_copy_respects_off(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        vio.cfg.output.copy_to_clipboard = "off"
        with patch("voiceio.clipboard_read.copy_text") as mock_copy:
            vio._copy_result_async("hello")
            time.sleep(0.1)
            mock_copy.assert_not_called()

    def test_final_copy_runs_async(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        vio.cfg.output.copy_to_clipboard = "final"
        done = threading.Event()
        with patch("voiceio.clipboard_read.copy_text",
                   side_effect=lambda t: done.set()) as mock_copy:
            vio._copy_result_async("hello world")
            assert done.wait(timeout=2)
            mock_copy.assert_called_once_with("hello world")


class TestEngineLifecycle:
    """Dormant-engine reaping is normal; recovery must not steal the source."""

    def _vio_ibus(self, proc_poll=-15):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        proc = MagicMock()
        proc.poll.return_value = proc_poll
        proc.returncode = -15
        vio._engine_proc = proc
        return vio, proc

    def test_idle_engine_death_not_recovered(self):
        """ibus-daemon reaping a dormant engine must NOT trigger the
        activate/dormant dance (it grabbed the input source every cycle)."""
        vio, proc = self._vio_ibus(proc_poll=-15)
        with patch.object(vio, "_ensure_ibus_engine") as ensure, \
             patch.object(vio, "_health_typer_upkeep"):
            vio._check_health()
        ensure.assert_not_called()
        assert vio._engine_proc is None

    def test_mid_recording_engine_death_recovered(self):
        from voiceio.app import _State
        vio, proc = self._vio_ibus(proc_poll=-15)
        vio._state = _State.RECORDING
        with patch.object(vio, "_ensure_ibus_engine") as ensure:
            vio._check_health()
        ensure.assert_called_once()

    def test_single_missed_ping_tolerated(self):
        """One missed ping on a loaded system is not a zombie."""
        vio, proc = self._vio_ibus(proc_poll=None)
        with patch.object(vio, "_ping_ibus_engine", return_value=False), \
             patch.object(vio, "_ensure_ibus_engine") as ensure, \
             patch.object(vio, "_health_typer_upkeep"):
            vio._check_health()
        ensure.assert_not_called()
        proc.kill.assert_not_called()
        assert vio._engine_ping_fails == 1

    def test_two_missed_pings_restart_zombie(self):
        vio, proc = self._vio_ibus(proc_poll=None)
        with patch.object(vio, "_ping_ibus_engine", return_value=False), \
             patch.object(vio, "_ensure_ibus_engine") as ensure, \
             patch.object(vio, "_health_typer_upkeep"):
            vio._check_health()
            vio._check_health()
        ensure.assert_called_once()
        proc.kill.assert_called_once()

    def test_successful_ping_resets_counter(self):
        vio, proc = self._vio_ibus(proc_poll=None)
        vio._engine_ping_fails = 1
        with patch.object(vio, "_ping_ibus_engine", return_value=True), \
             patch.object(vio, "_ensure_ibus_engine") as ensure, \
             patch.object(vio, "_health_typer_upkeep"):
            vio._check_health()
        ensure.assert_not_called()
        assert vio._engine_ping_fails == 0

    def test_record_start_respawns_reaped_engine(self):
        vio, mock_typer, _ = _make_vio()
        mock_typer.name = "ibus"
        vio._engine_proc = None  # reaped while dormant
        spawned = threading.Event()
        with patch.object(vio, "_ensure_ibus_engine", side_effect=lambda: spawned.set()), \
             patch.object(vio, "_switch_gnome_input_source"):
            vio.on_hotkey()  # start recording
            assert spawned.wait(timeout=2), "engine respawn not triggered"
        assert vio.recorder.is_recording


class TestEngineRelease:
    """Going dormant must release ibus's GLOBAL engine, not just gsettings."""

    def test_release_sets_user_engine_unconditionally(self):
        """Release must not query first: a stale reading during the async
        activation made it skip itself and leave voiceio globally active."""
        vio, _, _ = _make_vio()
        vio._prev_ibus_engine = "xkb:us::eng"
        with patch("voiceio.app.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            vio._release_ibus_engine()
        assert run.call_count == 1
        assert run.call_args[0][0] == ["ibus", "engine", "xkb:us::eng"]

    def test_fallback_derived_from_sources_when_prev_stale(self):
        """A stale 'voiceio' prev (inherited from a crashed daemon) must not
        be restored; derive the user's xkb layout instead."""
        vio, _, _ = _make_vio()
        vio._prev_ibus_engine = "voiceio"
        with patch("voiceio.app.subprocess.run") as run:
            run.return_value = MagicMock(
                returncode=0, stdout="[('xkb', 'de'), ('ibus', 'voiceio')]\n",
            )
            assert vio._ibus_engine_fallback() == "xkb:de::eng"

    def test_restore_input_source_releases_engine(self):
        vio, _, _ = _make_vio()
        vio.platform.is_gnome = True
        with patch.object(vio, "_set_gnome_input_source_index"), \
             patch.object(vio, "_release_ibus_engine") as release:
            vio._restore_input_source()
        release.assert_called_once()


def test_activation_claims_global_engine():
    """gsettings 'current' is legacy/no-op on newer GNOME — activation must
    also claim the global engine directly (mirror of the dormant release)."""
    vio, _, _ = _make_vio()
    vio.platform.is_gnome = True
    with patch("voiceio.app.subprocess.run") as run:
        run.side_effect = [
            MagicMock(returncode=0, stdout="[('xkb', 'us'), ('ibus', 'voiceio')]\n"),  # sources
            MagicMock(returncode=0, stdout="0\n"),   # current (prev)
            MagicMock(returncode=0),                 # gsettings set current
            MagicMock(returncode=0),                 # ibus engine voiceio
        ]
        with patch("time.sleep"):
            vio._switch_gnome_input_source("voiceio")
    cmds = [c[0][0] for c in run.call_args_list]
    assert ["ibus", "engine", "voiceio"] in cmds
