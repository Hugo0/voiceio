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
        vio.recorder._stream = MagicMock()  # skip real audio
        return vio, mock_typer, mock_transcriber


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

        # Feed some audio
        for _ in range(20):
            data = np.full((1024, 1), 0.3, dtype=np.float32)
            vio.recorder._callback(data, 1024, None, None)

        # Set record_start far back so debounce allows stop
        vio._record_start = time.monotonic() - 5.0
        vio._last_hotkey = time.monotonic() - 5.0

        # Stop recording
        vio.on_hotkey()
        assert not vio.recorder.is_recording


class TestHotkeyDebounce:
    """Verify that duplicate hotkey events are properly debounced."""

    def test_rapid_duplicate_ignored(self):
        """Two on_hotkey calls within 0.3s should only trigger once."""
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

        # Feed audio
        for _ in range(20):
            data = np.full((1024, 1), 0.3, dtype=np.float32)
            vio.recorder._callback(data, 1024, None, None)

        # Move timestamps back so debounce allows through
        vio._record_start = time.monotonic() - 2.0
        vio._last_hotkey = time.monotonic() - 2.0

        vio.on_hotkey()
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

        # Feed audio
        for _ in range(20):
            data = np.full((1024, 1), 0.3, dtype=np.float32)
            vio.recorder._callback(data, 1024, None, None)

        # Allow stop
        vio._record_start = time.monotonic() - 2.0
        vio._last_hotkey = time.monotonic() - 2.0

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
