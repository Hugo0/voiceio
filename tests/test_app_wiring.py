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
        vio = VoiceIO(Config())
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
