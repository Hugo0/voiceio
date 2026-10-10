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


# ── offline selection (allow_network) ────────────────────────────────

def test_cloud_engines_is_declared():
    from voiceio.tts.chain import CLOUD_ENGINES, _ENGINES
    assert "edge-tts" in CLOUD_ENGINES
    # Every named cloud engine must actually exist, or the filter silently
    # stops filtering when an engine is renamed.
    assert CLOUD_ENGINES <= set(_ENGINES)


def test_auto_select_skips_cloud_when_offline():
    """A caller that promised its user nothing leaves the box must not be
    handed edge-tts just because it probed OK."""
    from voiceio.tts import chain

    tried = []

    def fake_create(name, cfg):
        tried.append(name)
        engine = MagicMock()
        engine.name = name
        # piper unavailable, so auto would otherwise fall through to edge-tts
        engine.probe.return_value = ProbeResult(ok=(name == "espeak"))
        return engine

    with patch.object(chain, "_create", fake_create):
        engine = chain.select(TTSConfig(engine="auto"), allow_network=False)

    assert "edge-tts" not in tried
    assert engine.name == "espeak"


def test_auto_select_allows_cloud_by_default():
    from voiceio.tts import chain

    def fake_create(name, cfg):
        engine = MagicMock()
        engine.name = name
        engine.probe.return_value = ProbeResult(ok=(name == "edge-tts"))
        return engine

    with patch.object(chain, "_create", fake_create):
        engine = chain.select(TTSConfig(engine="auto"))

    assert engine.name == "edge-tts"


def test_explicit_cloud_engine_is_refused_not_downgraded_when_offline():
    """Silently substituting espeak would ship the wrong voice unnoticed."""
    from voiceio.tts import chain

    with patch.object(chain, "_create") as create:
        result = chain.select(TTSConfig(engine="edge-tts"), allow_network=False)

    assert result is None
    create.assert_not_called()


def test_explicit_local_engine_unaffected_by_offline_flag():
    from voiceio.tts import chain

    def fake_create(name, cfg):
        engine = MagicMock()
        engine.name = name
        engine.probe.return_value = ProbeResult(ok=True)
        return engine

    with patch.object(chain, "_create", fake_create):
        engine = chain.select(TTSConfig(engine="espeak"), allow_network=False)

    assert engine.name == "espeak"


# ── piper tests ──────────────────────────────────────────────────────
#
# piper is not a test dependency (it lives in the `tts`/`linux` extras), and
# the engine imports it lazily, so these stub the module. That is also what
# makes them a real regression test for the 1.3 API break: a stub that offers
# only the modern API is exactly the shape of a modern install.

def _fake_piper(*, modern: bool = True):
    """A stand-in `piper` package tree. `modern=False` is a pre-1.3 install:
    it has neither SynthesisConfig nor download_voices."""
    piper = MagicMock()
    piper.PiperVoice = MagicMock()
    modules = {"piper": piper}
    if modern:
        piper.SynthesisConfig = _FakeSynthesisConfig
        download_voices = MagicMock()
        modules["piper.download_voices"] = download_voices
    else:
        del piper.SynthesisConfig
        # Pre-1.3 had `piper.download`; the absent key is what breaks the
        # import, matching ModuleNotFoundError on a real old install.
        modules["piper.download_voices"] = None
    return modules


class _FakeSynthesisConfig:
    """Records what the engine asked for."""

    def __init__(self, length_scale=None, **kw):
        self.length_scale = length_scale
        self.kw = kw


def _fake_chunk(samples: int = 100):
    chunk = MagicMock()
    chunk.audio_int16_array = np.zeros(samples, dtype=np.int16)
    return chunk


def test_piper_probe_ok_on_modern_api():
    """The 1.3 API is the whole probe. Probing for the removed
    `piper.download` reported piper unavailable on every current install."""
    from voiceio.tts.piper_engine import PiperEngine

    with patch.dict("sys.modules", _fake_piper()):
        result = PiperEngine().probe()

    assert result.ok


def test_piper_probe_rejects_pre_1_3_install():
    from voiceio.tts.piper_engine import PiperEngine

    with patch.dict("sys.modules", _fake_piper(modern=False)):
        result = PiperEngine().probe()

    assert not result.ok
    assert "1.3" in result.reason
    assert "piper-tts" in result.fix_hint


def test_piper_probe_does_not_import_removed_download_module():
    """`piper.download` is gone in 1.3+. Touching it fails the probe and,
    through auto-selection, answers /tts with 503."""
    from voiceio.tts.piper_engine import PiperEngine

    modules = _fake_piper()
    modules["piper.download"] = None  # ImportError if anyone reaches for it

    with patch.dict("sys.modules", modules):
        assert PiperEngine().probe().ok


def test_piper_speed_becomes_inverse_length_scale():
    """length_scale stretches phonemes, so it is 1/speed — getting this
    backwards makes "faster" slower."""
    from voiceio.tts.piper_engine import PiperEngine

    engine = PiperEngine()
    voice = MagicMock()
    voice.synthesize.return_value = [_fake_chunk()]
    engine._voice = voice
    engine._sample_rate = 22050

    with patch.dict("sys.modules", _fake_piper()):
        engine.synthesize("hello", "", 1.5)

    syn_config = voice.synthesize.call_args.kwargs["syn_config"]
    assert syn_config.length_scale == pytest.approx(1 / 1.5)


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_piper_unset_speed_is_natural_rate(speed):
    from voiceio.tts.piper_engine import PiperEngine

    engine = PiperEngine()
    voice = MagicMock()
    voice.synthesize.return_value = [_fake_chunk()]
    engine._voice = voice
    engine._sample_rate = 22050

    with patch.dict("sys.modules", _fake_piper()):
        engine.synthesize("hello", "", speed)

    assert voice.synthesize.call_args.kwargs["syn_config"].length_scale == 1.0


def test_piper_synthesize_concatenates_chunks():
    from voiceio.tts.piper_engine import PiperEngine

    engine = PiperEngine()
    voice = MagicMock()
    voice.synthesize.return_value = [_fake_chunk(100), _fake_chunk(50)]
    engine._voice = voice
    engine._sample_rate = 22050

    with patch.dict("sys.modules", _fake_piper()):
        audio, rate = engine.synthesize("hello", "", 1.0)

    assert audio.dtype == np.int16
    assert len(audio) == 150
    assert rate == 22050


def test_piper_synthesize_empty_result():
    from voiceio.tts.piper_engine import PiperEngine

    engine = PiperEngine()
    voice = MagicMock()
    voice.synthesize.return_value = []
    engine._voice = voice
    engine._sample_rate = 22050

    with patch.dict("sys.modules", _fake_piper()):
        audio, _ = engine.synthesize("", "", 1.0)

    assert len(audio) == 0
    assert audio.dtype == np.int16


def test_piper_downloads_voice_once_when_missing(tmp_path):
    from voiceio.tts import piper_engine

    modules = _fake_piper()
    download_voice = modules["piper.download_voices"].download_voice
    engine = piper_engine.PiperEngine(model="en_US-lessac-medium")

    with patch.dict("sys.modules", modules), \
            patch.object(piper_engine, "_models_dir", return_value=tmp_path):
        engine._ensure_voice()
        engine._ensure_voice()  # cached — must not download or load twice

    download_voice.assert_called_once_with("en_US-lessac-medium", tmp_path)
    modules["piper"].PiperVoice.load.assert_called_once()


def test_piper_skips_download_when_model_present(tmp_path):
    from voiceio.tts import piper_engine

    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"onnx")
    modules = _fake_piper()
    engine = piper_engine.PiperEngine()

    with patch.dict("sys.modules", modules), \
            patch.object(piper_engine, "_models_dir", return_value=tmp_path):
        engine._ensure_voice()

    modules["piper.download_voices"].download_voice.assert_not_called()


def test_piper_never_writes_to_the_cwd(tmp_path):
    """piper's `download_dir` defaults to the current working directory. A
    daemon's cwd is not ours to litter in."""
    from voiceio.tts import piper_engine

    modules = _fake_piper()
    engine = piper_engine.PiperEngine()

    with patch.dict("sys.modules", modules), \
            patch.object(piper_engine, "_models_dir", return_value=tmp_path):
        engine._ensure_voice()

    assert modules["piper"].PiperVoice.load.call_args.kwargs["download_dir"] == tmp_path


def test_piper_shutdown_releases_voice():
    from voiceio.tts.piper_engine import PiperEngine

    engine = PiperEngine()
    engine._voice = MagicMock()
    engine.shutdown()
    assert engine._voice is None
