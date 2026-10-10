"""Decoder adapters and factory (engines faked; no model weights needed)."""
from __future__ import annotations

import io
import json
import sys
import types

import numpy as np
import pytest

from voiceio.config import ModelConfig
from voiceio.core.decoders import BiasMode, build_decoder
from voiceio.core.decoders.base import Bias, DecodeError, Decoder
from voiceio.core.decoders.catalog import CATALOG, info


def test_catalog_entries_declare_license_and_bias():
    assert all(m.license and isinstance(m.bias, BiasMode) for m in CATALOG)
    assert info("small").runtime == "faster-whisper"
    assert info("whispercpp:http://x:1").runtime == "whisper.cpp"


def test_whispercpp_server_sends_prompt_and_parses_text(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["url"], seen["body"] = req.full_url, req.data
        return Resp(json.dumps({"text": " hello Kalshi "}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    dec = build_decoder(ModelConfig(name="whispercpp:http://127.0.0.1:8080"))
    assert isinstance(dec, Decoder)
    out = dec.decode(np.zeros(1600, dtype=np.float32), bias=Bias(hotwords="Kalshi"))
    assert out.text == "hello Kalshi"
    assert seen["url"].endswith("/inference")
    assert b'name="prompt"' in seen["body"] and b"Kalshi" in seen["body"]
    assert dec.capabilities.bias == BiasMode.PROMPT


def test_whispercpp_errors_are_decode_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    dec = build_decoder(ModelConfig(name="whispercpp:http://127.0.0.1:9"))
    with pytest.raises(DecodeError):
        dec.decode(np.zeros(1600, dtype=np.float32))


def _fake_sherpa(monkeypatch, created):
    class Stream:
        def accept_waveform(self, sr, a):
            self.n = len(a)

        @property
        def result(self):
            return types.SimpleNamespace(text=" hello world ")

    class Rec:
        def create_stream(self):
            return Stream()

        def decode_stream(self, s):
            pass

    class OfflineRecognizer:
        @staticmethod
        def from_transducer(**kw):
            created.append(kw)
            return Rec()

    monkeypatch.setitem(sys.modules, "sherpa_onnx",
                        types.SimpleNamespace(OfflineRecognizer=OfflineRecognizer))


def _model_dir(tmp_path):
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (tmp_path / f).write_text("x")
    return tmp_path


def test_sherpa_greedy_declares_no_bias(monkeypatch, tmp_path):
    created = []
    _fake_sherpa(monkeypatch, created)
    dec = build_decoder(ModelConfig(name=f"sherpa:{_model_dir(tmp_path)}"))
    assert dec.capabilities.bias == BiasMode.NONE
    assert dec.decode(np.zeros(16000, dtype=np.float32),
                      bias=Bias(hotwords="Kalshi")).text == "hello world"
    assert "hotwords_file" not in created[-1]


def test_sherpa_mbs_passes_unbounded_hotwords(monkeypatch, tmp_path):
    created = []
    _fake_sherpa(monkeypatch, created)
    dec = build_decoder(ModelConfig(name=f"sherpa:{_model_dir(tmp_path)}",
                                    decoding="modified_beam_search"))
    assert dec.capabilities.bias == BiasMode.UNBOUNDED
    dec.decode(np.zeros(16000, dtype=np.float32), bias=Bias(hotwords="Kalshi, HyperNEAT"))
    kw = created[-1]
    assert kw["decoding_method"] == "modified_beam_search"
    with open(kw["hotwords_file"]) as f:
        assert f.read().split() == ["Kalshi", "HyperNEAT"]


def test_lanes_share_the_thread_budget(monkeypatch):
    import voiceio.transcriber as tr
    from voiceio.worker import auto_cpu_threads
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert (auto_cpu_threads(1), auto_cpu_threads(2)) == (4, 4)
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert (auto_cpu_threads(1), auto_cpu_threads(2)) == (4, 2)

    seen = []
    monkeypatch.setattr(tr, "Transcriber", lambda cfg: seen.append(cfg.cpu_threads) or types.SimpleNamespace(name="small"))
    build_decoder(ModelConfig(name="small"), lanes=2)
    build_decoder(ModelConfig(name="small", cpu_threads=6), lanes=2)
    assert seen == [2, 2, 6, 6]
