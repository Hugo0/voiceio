"""Tests for local data retention (audio WAVs, pruning, context)."""
from __future__ import annotations

import wave
from unittest.mock import patch

import numpy as np

from voiceio import retention
from voiceio.config import DataConfig


class TestSaveAudio:
    def test_saves_wav(self, tmp_path):
        cfg = DataConfig()
        audio = np.full(16000, 0.1, dtype=np.float32)
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            name = retention.save_audio(audio, ts=1700000000.5, cfg=cfg)
        assert name and name.endswith(".wav")
        with wave.open(str(tmp_path / name)) as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getnframes() == 16000

    def test_disabled(self, tmp_path):
        cfg = DataConfig(retain_audio=False)
        audio = np.full(16000, 0.1, dtype=np.float32)
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            assert retention.save_audio(audio, ts=0, cfg=cfg) is None

    def test_none_and_empty_audio(self, tmp_path):
        cfg = DataConfig()
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            assert retention.save_audio(None, ts=0, cfg=cfg) is None
            assert retention.save_audio(np.zeros(0, dtype=np.float32), ts=0, cfg=cfg) is None

    def test_roundtrip_amplitude(self, tmp_path):
        cfg = DataConfig()
        audio = np.full(1600, 0.5, dtype=np.float32)
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            name = retention.save_audio(audio, ts=1700000001.0, cfg=cfg)
        with wave.open(str(tmp_path / name)) as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        assert abs(pcm[0] / 32767 - 0.5) < 0.001


class TestPrune:
    def test_prunes_oldest_over_cap(self, tmp_path):
        import os
        cfg = DataConfig(max_audio_mb=1)
        # Three ~600KB files: total ~1.8MB > 1MB cap → oldest goes
        for i, name in enumerate(["a.wav", "b.wav", "c.wav"]):
            p = tmp_path / name
            p.write_bytes(b"\0" * 600_000)
            os.utime(p, (1000 + i, 1000 + i))
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            retention.prune(cfg)
        remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
        assert "a.wav" not in remaining
        assert "c.wav" in remaining

    def test_noop_under_cap(self, tmp_path):
        cfg = DataConfig(max_audio_mb=100)
        (tmp_path / "a.wav").write_bytes(b"\0" * 1000)
        with patch.object(retention, "RECORDINGS_DIR", tmp_path):
            retention.prune(cfg)
        assert (tmp_path / "a.wav").exists()

    def test_missing_dir(self, tmp_path):
        with patch.object(retention, "RECORDINGS_DIR", tmp_path / "nope"):
            retention.prune(DataConfig())  # should not raise


class TestHistoryExtra:
    def test_extra_fields_stored(self, tmp_path):
        import json
        from voiceio import history

        p = tmp_path / "h.jsonl"
        history.append(
            "hello", path=p,
            extra={"audio": "x.wav", "context": "Terminal", "model": "small",
                   "skipme": None},
        )
        entry = json.loads(p.read_text())
        assert entry["audio"] == "x.wav"
        assert entry["context"] == "Terminal"
        assert entry["model"] == "small"
        assert "skipme" not in entry

    def test_extra_cannot_clobber_core_fields(self, tmp_path):
        import json
        from voiceio import history

        p = tmp_path / "h.jsonl"
        history.append("hello", path=p, extra={"text": "evil", "ts": 0})
        entry = json.loads(p.read_text())
        assert entry["text"] == "hello"
        assert entry["ts"] != 0
