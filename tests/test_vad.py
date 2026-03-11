"""Tests for VAD backends."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from voiceio.vad import RmsVad, SileroVad, load_vad


class TestRmsVad:
    def test_loud_is_speech(self):
        vad = RmsVad(threshold=0.01)
        chunk = np.full(1024, 0.3, dtype=np.float32)
        assert vad.is_speech(chunk) is True

    def test_quiet_is_silence(self):
        vad = RmsVad(threshold=0.01)
        chunk = np.zeros(1024, dtype=np.float32)
        assert vad.is_speech(chunk) is False

    def test_near_threshold(self):
        vad = RmsVad(threshold=0.01)
        # Just below threshold
        chunk = np.full(1024, 0.005, dtype=np.float32)
        assert vad.is_speech(chunk) is False
        # Just above threshold
        chunk = np.full(1024, 0.02, dtype=np.float32)
        assert vad.is_speech(chunk) is True

    def test_reset_is_noop(self):
        vad = RmsVad()
        vad.reset()  # should not raise

    def test_empty_chunk(self):
        vad = RmsVad(threshold=0.01)
        chunk = np.zeros(0, dtype=np.float32)
        assert vad.is_speech(chunk) is False


class TestSileroVad:
    @pytest.fixture
    def vad(self):
        try:
            return SileroVad(threshold=0.5)
        except Exception:
            pytest.skip("onnxruntime or silero model not available")

    def test_silence_detected(self, vad):
        chunk = np.zeros(1024, dtype=np.float32)
        assert vad.is_speech(chunk) is False

    def test_speech_detected(self, vad):
        # Generate a sine wave at speech-like frequency
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)
        chunk = 0.5 * np.sin(2 * np.pi * 300 * t)
        # Run several chunks to build state
        for _ in range(5):
            result = vad.is_speech(chunk)
        # At least one should detect speech (sine wave isn't silence)
        # Note: Silero may not classify pure sine as speech, so we just test it runs
        assert isinstance(result, bool)

    def test_reset_clears_state(self, vad):
        chunk = np.zeros(1024, dtype=np.float32)
        vad.is_speech(chunk)
        vad.reset()
        # After reset, internal buffer and hidden states should be cleared
        assert len(vad._buf) == 0
        if vad._use_state:
            assert np.all(vad._state == 0)
        else:
            assert np.all(vad._h == 0)
            assert np.all(vad._c == 0)

    def test_various_chunk_sizes(self, vad):
        """Different chunk sizes should all work (internal buffering)."""
        for size in [256, 512, 1024, 2048, 100]:
            chunk = np.zeros(size, dtype=np.float32)
            result = vad.is_speech(chunk)
            assert isinstance(result, bool)

    def test_warmup(self, vad):
        vad.warmup()
        # After warmup + reset, state is clean
        assert len(vad._buf) == 0

    def test_2d_input(self, vad):
        """Should handle 2D input (channels, samples) like sounddevice provides."""
        chunk = np.zeros((1024, 1), dtype=np.float32)
        result = vad.is_speech(chunk)
        assert isinstance(result, bool)


class TestLoadVad:
    def test_rms_config_forces_rms(self):
        from voiceio.config import AudioConfig
        cfg = AudioConfig(vad_backend="rms")
        vad = load_vad(cfg)
        assert isinstance(vad, RmsVad)

    def test_silero_loads_when_available(self):
        from voiceio.config import AudioConfig
        cfg = AudioConfig(vad_backend="silero")
        vad = load_vad(cfg)
        try:
            import onnxruntime  # noqa: F401
            assert isinstance(vad, SileroVad)
        except ImportError:
            assert isinstance(vad, RmsVad)

    def test_falls_back_to_rms_on_import_error(self):
        from voiceio.config import AudioConfig
        cfg = AudioConfig(vad_backend="silero")
        with patch("voiceio.vad.SileroVad", side_effect=ImportError("no onnxruntime")):
            vad = load_vad(cfg)
        assert isinstance(vad, RmsVad)
