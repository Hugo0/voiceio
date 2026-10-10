"""Tests for the shared dictation engine."""
from __future__ import annotations

import numpy as np
import pytest

from fakes import FakeDecoder
from voiceio.config import Config
from voiceio.core.decoders.base import DecodeError
from voiceio.core.engine import Engine


def _engine(**cfg_output):
    cfg = Config()
    for k, v in cfg_output.items():
        setattr(cfg.output, k, v)
    dec = FakeDecoder()
    return Engine.from_config(cfg, decoder=dec), dec


def test_transcribe_runs_pipeline_and_keeps_raw():
    eng, dec = _engine(number_conversion=True, punctuation_cleanup=True)
    dec.transcribe.return_value = "i have twenty five apples"
    t = eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert t.raw == "i have twenty five apples"
    assert "25" in t.text
    assert {"transcribe", "pipeline", "audio_secs"} <= set(t.latency)


def test_transcribe_passes_vocabulary_bias(tmp_path, monkeypatch):
    from voiceio import config
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (config.CONFIG_DIR / "vocabulary.txt").write_text("Kalshi\nHyperNEAT\n")
    eng, dec = _engine()
    dec.transcribe.return_value = "hello"
    eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert "Kalshi" in (dec.biases[-1].hotwords or "")


def test_decode_failure_is_not_silence():
    eng, dec = _engine()
    dec.transcribe.side_effect = DecodeError("timeout")
    with pytest.raises(DecodeError):
        eng.transcribe(np.zeros(16000, dtype=np.float32))


def test_record_usage_feeds_prompt():
    eng, dec = _engine()
    eng.record_usage("we deployed the Kalshi bot")
    assert "Kalshi" in (eng.bias().prompt or "")


def test_session_uses_engine_pipeline_and_bias():
    from voiceio.audio_source import PushAudioSource
    from voiceio.typers.collecting import CollectingTyper
    eng, dec = _engine(number_conversion=True)
    dec.transcribe.return_value = "twenty five"
    src = PushAudioSource(16000)
    session = eng.session(src, CollectingTyper())
    session.start()
    src.feed(np.full(16000, 0.3, dtype=np.float32))
    assert session.stop(src.get_audio_so_far()) == "25"


def test_commit_strips_prefix_learns_and_keeps_history():
    import json

    from voiceio import config
    eng, _ = _engine(voice_input_prefix="[voice]")
    stored = eng.commit("[voice] we deployed the Kalshi bot", raw="we deployed the kalshi bot",
                        duration=2.0, audio=np.zeros(16000, dtype=np.float32))
    assert stored == "we deployed the Kalshi bot"
    assert "Kalshi" in (eng.bias().prompt or "")
    row = json.loads(config.HISTORY_PATH.read_text().splitlines()[-1])
    assert row["text"] == stored and (config.RECORDINGS_DIR / row["audio"]).exists()


def test_commit_without_persist_learns_but_writes_nothing():
    from voiceio import config
    eng, _ = _engine()
    eng.commit("ship the Kalshi fix", raw="ship the kalshi fix", duration=1.0, persist=False)
    assert "Kalshi" in (eng.bias().prompt or "")
    assert not config.HISTORY_PATH.exists()


def test_prepare_final_gives_postcorrect_full_vocabulary_and_title():
    from unittest.mock import MagicMock

    from voiceio import config
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (config.CONFIG_DIR / "vocabulary.txt").write_text("Kalshi\nHyperNEAT\n")
    pc = MagicMock()
    eng = Engine.from_config(Config(), decoder=FakeDecoder(), postcorrect=pc)
    eng.prepare_final(title="editor")
    kw = pc.set_context.call_args.kwargs
    assert "Kalshi" in kw["vocabulary"] and kw["title"] == "editor"
    Engine.from_config(Config(), decoder=FakeDecoder()).prepare_final(title="x")  # no-op
