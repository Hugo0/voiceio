"""Tests for audio normalization, clipping metering, and adaptive VAD."""
from __future__ import annotations

import numpy as np

from voiceio.transcriber import normalize_audio
from voiceio.vad import RmsVad


class TestNormalizeAudio:
    def test_quiet_audio_is_boosted(self):
        audio = np.full(16000, 0.005, dtype=np.float32)  # ~-46 dBFS
        out = normalize_audio(audio)
        rms = float(np.sqrt(np.mean(out**2)))
        assert 0.05 < rms < 0.15  # near -20 dBFS target

    def test_gain_capped_at_30db(self):
        audio = np.full(16000, 1e-4, dtype=np.float32)
        out = normalize_audio(audio)
        assert float(np.max(np.abs(out))) <= 1e-4 * 10 ** (30 / 20) * 1.01

    def test_peak_never_exceeds_minus_1dbfs_when_boosting(self):
        # Quiet RMS but one loud transient
        audio = np.full(16000, 0.005, dtype=np.float32)
        audio[8000] = 0.5
        out = normalize_audio(audio)
        assert float(np.max(np.abs(out))) <= 10 ** (-1 / 20) + 1e-6

    def test_normal_level_untouched(self):
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(16000) * 0.1).astype(np.float32)
        out = normalize_audio(audio)
        assert out is audio  # gain ~1 → returned unchanged

    def test_silence_untouched(self):
        audio = np.zeros(16000, dtype=np.float32)
        assert normalize_audio(audio) is audio

    def test_empty(self):
        audio = np.zeros(0, dtype=np.float32)
        assert len(normalize_audio(audio)) == 0

    def test_output_dtype_float32(self):
        audio = np.full(16000, 0.005, dtype=np.float32)
        assert normalize_audio(audio).dtype == np.float32


class TestRecorderMeter:
    def _recorder(self):
        from unittest.mock import MagicMock, patch
        from voiceio.config import AudioConfig
        from voiceio.recorder import AudioRecorder

        rec = AudioRecorder(AudioConfig())
        rec._stream = MagicMock()
        return rec, patch

    def test_clipping_detected_on_flat_top(self):
        rec, _ = self._recorder()
        rec._recording = True
        rec._chunks = []
        chunk = np.full((1024, 1), 0.995, dtype=np.float32)  # saturated
        rec._callback(chunk, 1024, None, None)
        meter = rec.get_meter()
        assert meter["clip_ratio"] > 0.9
        assert meter["peak"] >= 0.99

    def test_isolated_peak_not_counted_as_clipping(self):
        rec, _ = self._recorder()
        rec._recording = True
        rec._chunks = []
        chunk = np.full((1024, 1), 0.1, dtype=np.float32)
        chunk[500] = 1.0  # single-sample transient (plosive), not a flat top
        rec._callback(chunk, 1024, None, None)
        meter = rec.get_meter()
        assert meter["clip_ratio"] == 0.0
        assert meter["peak"] == 1.0

    def test_meter_resets_on_start(self):
        rec, _ = self._recorder()
        rec._meter_peak = 1.0
        rec._meter_clipped = 500
        rec._meter_samples = 1000
        rec.start()
        meter = rec.get_meter()
        assert meter["peak"] == 0.0
        assert meter["clip_ratio"] == 0.0


class TestAdaptiveRmsVad:
    def test_hot_mic_noise_becomes_silence(self):
        """Ambient noise above the fixed threshold must still read as silence
        once the floor adapts."""
        vad = RmsVad(threshold=0.01)
        noise = np.full(1024, 0.05, dtype=np.float32)  # constant hot-mic hiss
        results = [vad.is_speech(noise) for _ in range(50)]
        assert results[-1] is False  # adapted: hiss is the floor now

    def test_speech_above_adapted_floor_detected(self):
        vad = RmsVad(threshold=0.01)
        noise = np.full(1024, 0.05, dtype=np.float32)
        for _ in range(50):
            vad.is_speech(noise)
        speech = np.full(1024, 0.4, dtype=np.float32)
        assert vad.is_speech(speech) is True

    def test_fixed_threshold_before_adaptation(self):
        """First frames behave exactly like the fixed threshold."""
        vad = RmsVad(threshold=0.01)
        assert vad.is_speech(np.full(1024, 0.3, dtype=np.float32)) is True
        vad2 = RmsVad(threshold=0.01)
        assert vad2.is_speech(np.zeros(1024, dtype=np.float32)) is False

    def test_normal_mic_unchanged(self):
        """Quiet ambient floor never raises the threshold above the fixed one."""
        vad = RmsVad(threshold=0.01)
        quiet = np.full(1024, 0.001, dtype=np.float32)
        for _ in range(50):
            vad.is_speech(quiet)
        assert vad.is_speech(np.full(1024, 0.02, dtype=np.float32)) is True


class TestPromptBudget:
    """Whisper shares a 448-token budget between hotwords, initial_prompt and
    output. Oversized bias inputs truncated real transcriptions mid-utterance
    (2026-07-05 regression). These bounds must hold."""

    def test_hotwords_capped(self):
        from unittest.mock import MagicMock, patch
        from voiceio.app import _HOTWORDS_MAX_CHARS
        assert _HOTWORDS_MAX_CHARS <= 700

    def test_prompt_builder_default_budget_small(self):
        from voiceio.prompt import PromptBuilder
        pb = PromptBuilder()
        for i in range(10):
            pb.add_transcript("word " * 120)
        assert len(pb.build()) <= 300
