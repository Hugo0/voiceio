"""Tests for TTS engine chain, espeak, and player."""
from unittest.mock import patch, MagicMock
import io
import struct
import wave

import numpy as np
import pytest

from voiceio.backends import ProbeResult
from voiceio.config import TTSConfig


# ── espeak tests ─────────────────────────────────────────────────────

def _make_wav_bytes(samples: int = 1000, rate: int = 22050) -> bytes:
    """Create minimal WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        data = np.zeros(samples, dtype=np.int16).tobytes()
        wf.writeframes(data)
    return buf.getvalue()


def test_espeak_probe_not_installed():
    from voiceio.tts.espeak import EspeakEngine
    engine = EspeakEngine()
    with patch("shutil.which", return_value=None):
        result = engine.probe()
        assert not result.ok
        assert "not installed" in result.reason


def test_espeak_probe_installed():
    from voiceio.tts.espeak import EspeakEngine
    engine = EspeakEngine()
    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        result = engine.probe()
        assert result.ok


def test_espeak_synthesize():
    from voiceio.tts.espeak import EspeakEngine
    engine = EspeakEngine()
    wav_data = _make_wav_bytes()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=wav_data, stderr=b"")
        audio, sr = engine.synthesize("hello", "", 1.0)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert sr == 22050
        # Check that speed is passed correctly
        args = mock_run.call_args[0][0]
        assert "-s" in args
        idx = args.index("-s")
        assert args[idx + 1] == "175"  # 175 * 1.0


def test_espeak_synthesize_custom_speed():
    from voiceio.tts.espeak import EspeakEngine
    engine = EspeakEngine()
    wav_data = _make_wav_bytes()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=wav_data, stderr=b"")
        engine.synthesize("hello", "en", 1.5)
        args = mock_run.call_args[0][0]
        idx = args.index("-s")
        assert args[idx + 1] == "262"  # int(175 * 1.5)


# ── chain tests ──────────────────────────────────────────────────────

def test_chain_probe_all():
    from voiceio.tts.chain import probe_all
    cfg = TTSConfig(enabled=True, engine="auto")

    # All engines will fail probe in test env, but should not crash
    results = probe_all(cfg)
    assert len(results) > 0
    assert all(isinstance(r[1], ProbeResult) for r in results)


def test_chain_select_specific_engine():
    from voiceio.tts.chain import select
    cfg = TTSConfig(enabled=True, engine="espeak")

    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        engine = select(cfg)
        assert engine is not None
        assert engine.name == "espeak"


def test_chain_select_unknown_engine():
    from voiceio.tts.chain import select
    cfg = TTSConfig(enabled=True, engine="nonexistent")
    engine = select(cfg)
    assert engine is None


def test_chain_select_auto_with_espeak():
    from voiceio.tts.chain import select, _create
    cfg = TTSConfig(enabled=True, engine="auto")

    # Mock so only espeak works
    def mock_create(name, cfg):
        engine = _create(name, cfg)
        if name == "espeak":
            engine.probe = lambda: ProbeResult(ok=True)
        else:
            engine.probe = lambda: ProbeResult(ok=False, reason="not installed")
        return engine

    with patch("voiceio.tts.chain._create", side_effect=mock_create):
        engine = select(cfg)
        assert engine is not None
        assert engine.name == "espeak"


# ── player tests ─────────────────────────────────────────────────────

def test_player_cancel():
    from voiceio.tts.player import TTSPlayer
    player = TTSPlayer()
    assert not player.is_playing()
    player.cancel()  # Should not crash when not playing


def test_player_empty_audio():
    from voiceio.tts.player import TTSPlayer
    player = TTSPlayer()
    # Empty audio should be a no-op
    player.play(np.array([], dtype=np.int16), 22050)
    assert not player.is_playing()


# ── config tests ─────────────────────────────────────────────────────

def test_tts_config_defaults():
    cfg = TTSConfig()
    assert cfg.enabled is True
    assert cfg.engine == "auto"
    assert cfg.hotkey == "ctrl+alt+s"
    assert cfg.voice == ""
    assert cfg.speed == 1.0
    assert cfg.model == ""


def test_tts_config_in_main_config():
    from voiceio.config import Config
    cfg = Config()
    assert hasattr(cfg, "tts")
    assert isinstance(cfg.tts, TTSConfig)
    assert cfg.tts.enabled is True
