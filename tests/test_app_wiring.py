"""Test that VoiceIO app wires up correctly with mocked backends."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voiceio.config import Config


class TestVoiceIOInit:
    """Verify VoiceIO can be constructed without real hardware."""

    def test_init_with_mocked_backends(self):
        from voiceio.backends import ProbeResult

        mock_hotkey = MagicMock()
        mock_hotkey.name = "socket"
        mock_hotkey.probe.return_value = ProbeResult(ok=True)

        mock_typer = MagicMock()
        mock_typer.name = "clipboard"
        mock_typer.probe.return_value = ProbeResult(ok=True)

        mock_transcriber = MagicMock()

        with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
             patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
             patch("voiceio.app.Transcriber", return_value=mock_transcriber), \
             patch("voiceio.app.plat.detect") as mock_detect:
            mock_detect.return_value = MagicMock(display_server="wayland", desktop="gnome")

            from voiceio.app import VoiceIO
            vio = VoiceIO(Config())

            assert vio._hotkey is mock_hotkey
            assert vio._typer is mock_typer

    def test_on_hotkey_toggle_cycle(self):
        """Test on_hotkey start/stop without real audio."""
        from voiceio.backends import ProbeResult

        mock_hotkey = MagicMock()
        mock_hotkey.name = "socket"

        mock_typer = MagicMock()
        mock_typer.name = "clipboard"

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "hello world"

        with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
             patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
             patch("voiceio.app.Transcriber", return_value=mock_transcriber), \
             patch("voiceio.app.plat.detect") as mock_detect:
            mock_detect.return_value = MagicMock(display_server="wayland", desktop="gnome")

            from voiceio.app import VoiceIO
            vio = VoiceIO(Config())
            vio.recorder._stream = MagicMock()  # skip real audio

            # First toggle: start recording
            vio.on_hotkey()
            assert vio.recorder.is_recording

            # Simulate some audio coming in
            for _ in range(20):
                data = np.full((1024, 1), 0.3, dtype=np.float32)
                vio.recorder._callback(data, 1024, None, None)

            # Hack: set _record_start far enough back
            import time
            vio._record_start = time.monotonic() - 5.0

            # Second toggle: stop recording
            vio.on_hotkey()
            assert not vio.recorder.is_recording

    def test_double_press_cancels_recording(self):
        """Rapid double-press (< 0.5s) cancels without typing."""
        mock_hotkey = MagicMock()
        mock_hotkey.name = "socket"
        mock_typer = MagicMock()
        mock_typer.name = "clipboard"

        with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
             patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
             patch("voiceio.app.Transcriber") as mock_trans_cls, \
             patch("voiceio.app.plat.detect") as mock_detect:
            mock_detect.return_value = MagicMock(display_server="wayland", desktop="gnome")

            from voiceio.app import VoiceIO
            vio = VoiceIO(Config())
            vio.recorder._stream = MagicMock()

            # Start recording
            vio.on_hotkey()
            assert vio.recorder.is_recording

            # Immediately double-press (< 0.5s) - cancels
            vio.on_hotkey()
            assert not vio.recorder.is_recording
            mock_typer.type_text.assert_not_called()

    def test_min_recording_duration_enforced(self):
        """Press between cancel_window and min_recording should be ignored."""
        import time
        mock_hotkey = MagicMock()
        mock_hotkey.name = "socket"
        mock_typer = MagicMock()
        mock_typer.name = "clipboard"

        cfg = Config()

        with patch("voiceio.app.hotkey_chain.select", return_value=mock_hotkey), \
             patch("voiceio.app.typer_chain.select", return_value=mock_typer), \
             patch("voiceio.app.Transcriber") as mock_trans_cls, \
             patch("voiceio.app.plat.detect") as mock_detect:
            mock_detect.return_value = MagicMock(display_server="wayland", desktop="gnome")

            from voiceio.app import VoiceIO
            vio = VoiceIO(cfg)
            vio.recorder._stream = MagicMock()

            # Start recording
            vio.on_hotkey()
            assert vio.recorder.is_recording

            # Set record_start past cancel window but before min duration
            vio._record_start = time.monotonic() - (cfg.output.cancel_window_secs + 0.1)

            # Press again - should be ignored
            vio.on_hotkey()
            assert vio.recorder.is_recording  # still recording
