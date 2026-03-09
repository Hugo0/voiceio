"""Integration tests for AudioRecorder — simulate full recording cycle without real audio hardware."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from voiceio.config import AudioConfig
from voiceio.recorder import AudioRecorder


def _make_recorder(**overrides) -> AudioRecorder:
    cfg = AudioConfig(sample_rate=16000, prebuffer_secs=overrides.get("prebuffer_secs", 0.5))
    rec = AudioRecorder(cfg)
    rec._stream = MagicMock()  # don't open real audio
    return rec


def _simulate_callbacks(rec: AudioRecorder, n_frames: int = 1024, n_callbacks: int = 5, value: float = 0.1):
    """Simulate sounddevice delivering 2D (frames, 1) chunks."""
    for _ in range(n_callbacks):
        data = np.full((n_frames, 1), value, dtype=np.float32)
        rec._callback(data, n_frames, None, None)


class TestRecordingCycle:
    def test_start_stop_returns_audio(self):
        rec = _make_recorder()
        rec.start()
        _simulate_callbacks(rec, n_callbacks=10)
        audio = rec.stop()
        assert audio is not None
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_stop_without_start_returns_none(self):
        rec = _make_recorder()
        assert rec.stop() is None

    def test_double_start_is_idempotent(self):
        rec = _make_recorder()
        rec.start()
        rec.start()  # should not crash or reset
        _simulate_callbacks(rec)
        audio = rec.stop()
        assert audio is not None

    def test_very_short_recording_skipped(self):
        """Less than 0.3s of audio should return None."""
        rec = _make_recorder()
        rec.start()
        # Only 1 callback of 1024 frames = 0.064s at 16kHz
        _simulate_callbacks(rec, n_callbacks=1)
        audio = rec.stop()
        assert audio is None

    def test_prebuffer_included_in_output(self):
        """Pre-buffer audio should appear at the start of the recording."""
        rec = _make_recorder(prebuffer_secs=0.5)

        # Fill ring buffer with recognizable values before recording starts
        for _ in range(10):
            prebuf_data = np.full((1600, 1), 0.99, dtype=np.float32)
            rec._callback(prebuf_data, 1600, None, None)

        assert not rec.is_recording  # ring buffer fills but no recording yet

        rec.start()
        # Record some different-valued audio
        _simulate_callbacks(rec, n_callbacks=10, value=0.5)
        audio = rec.stop()

        assert audio is not None
        # First samples should be from prebuffer (0.99), not from recording (0.5)
        assert audio[0] == pytest.approx(0.99, abs=0.01)

    def test_callbacks_before_recording_dont_accumulate(self):
        """Callbacks when not recording should only feed ring buffer, not chunks."""
        rec = _make_recorder(prebuffer_secs=0.1)

        # 100 callbacks while not recording
        _simulate_callbacks(rec, n_callbacks=100, value=0.1)

        rec.start()
        _simulate_callbacks(rec, n_callbacks=10, value=0.5)
        audio = rec.stop()

        assert audio is not None
        # Should be ~prebuffer + 10 callbacks, NOT 110 callbacks worth
        max_expected = int(0.1 * 16000) + 10 * 1024 + 100  # small margin
        assert len(audio) < max_expected

    def test_is_recording_state(self):
        rec = _make_recorder()
        assert not rec.is_recording
        rec.start()
        assert rec.is_recording
        rec.stop()
        assert not rec.is_recording

    def test_multiple_record_cycles(self):
        """Can start/stop multiple times without errors."""
        rec = _make_recorder()
        for _ in range(3):
            rec.start()
            _simulate_callbacks(rec, n_callbacks=10)
            audio = rec.stop()
            assert audio is not None


class TestStreamingVAD:
    def test_speech_pause_callback_fires(self):
        """on_speech_pause should fire after silence following speech."""
        pauses = []
        cfg = AudioConfig(sample_rate=16000, prebuffer_secs=0.0)
        rec = AudioRecorder(cfg, on_speech_pause=lambda: pauses.append(1))
        rec._stream = MagicMock()

        rec.start()

        # Simulate 2 seconds of "speech" (loud audio)
        for _ in range(32):
            data = np.full((1024, 1), 0.5, dtype=np.float32)
            rec._callback(data, 1024, None, None)

        # Simulate 1 second of silence
        for _ in range(16):
            data = np.full((1024, 1), 0.001, dtype=np.float32)
            rec._callback(data, 1024, None, None)

        assert len(pauses) > 0, "on_speech_pause should have been called"

    def test_no_pause_callback_without_enough_audio(self):
        """Shouldn't fire if less than 1s of new audio since last transcription."""
        pauses = []
        cfg = AudioConfig(sample_rate=16000, prebuffer_secs=0.0)
        rec = AudioRecorder(cfg, on_speech_pause=lambda: pauses.append(1))
        rec._stream = MagicMock()

        rec.start()
        # Only 512 samples of speech (~0.03s) then silence
        rec._callback(np.full((512, 1), 0.5, dtype=np.float32), 512, None, None)
        # 3 silence callbacks (not enough total audio to cross 1s threshold)
        for _ in range(3):
            rec._callback(np.full((1024, 1), 0.001, dtype=np.float32), 1024, None, None)

        assert len(pauses) == 0
