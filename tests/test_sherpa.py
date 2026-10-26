"""Parakeet / sherpa-onnx: one recognizer, per-stream hotwords, chunking,
default selection, model download. sherpa-onnx is faked: no weights."""
from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import threading
import types

import numpy as np
import pytest

from voiceio import config
from voiceio.config import ModelConfig
from voiceio.core.decoders import BiasMode, build_decoder, catalog, fetch, sherpa
from voiceio.core.decoders.base import Bias, DecodeError
from voiceio.core.decoders.sherpa import SherpaFeatures, SherpaTransducer, Tokenizer

FIXED = SherpaFeatures(installed=True, length_reward=True, pretokenized_hotwords=True)
TOKENS = "\n".join(f"{p} {i}" for i, p in enumerate(
    ["<unk>", "▁K", "al", "▁Kal", "sh", "hi", "shi", "▁", "K", "a", "l", "s", "h", "i",
     "▁H", "▁Hy", "per", "p", "e", "r", "y", "H", "▁k", "k", "er"])) + "\n"


class FakeSherpa:
    """Records recognizers built and the hotwords of every stream."""

    def __init__(self, text="hello world", block: threading.Event | None = None):
        self.created: list[dict] = []
        self.streams: list[dict] = []
        self.text = text
        self.block = block
        self.running = 0
        self.max_running = 0
        fake = self

        class Stream:
            def __init__(self, hotwords):
                self.rec = {"hotwords": hotwords, "samples": 0}
                fake.streams.append(self.rec)

            def set_option(self, key, value):
                assert isinstance(value, str)
                self.rec[key] = value

            def accept_waveform(self, sr, a):
                assert sr == 16000 and a.dtype == np.float32
                self.rec["samples"] = len(a)

            @property
            def result(self):
                return types.SimpleNamespace(text=f" {fake.text} ")

        class Rec:
            def create_stream(self, hotwords=None):
                return Stream(hotwords)

            def decode_stream(self, s):
                fake.running += 1
                fake.max_running = max(fake.max_running, fake.running)
                if fake.block is not None:
                    fake.block.wait(5)
                fake.running -= 1

        class OfflineRecognizer:
            @staticmethod
            def from_transducer(**kw):
                fake.created.append(kw)
                return Rec()

        self.module = types.SimpleNamespace(OfflineRecognizer=OfflineRecognizer)


@pytest.fixture
def fake(monkeypatch):
    f = FakeSherpa()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", f.module)
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    return f


def _model_dir(tmp_path):
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"):
        (tmp_path / name).write_text("x")
    (tmp_path / "tokens.txt").write_text(TOKENS, encoding="utf-8")
    return tmp_path


def _audio(secs: float) -> np.ndarray:
    return np.full(int(secs * 16000), 0.1, dtype=np.float32)


# ── tokenizer ────────────────────────────────────────────────────────────

def test_tokenizer_merges_by_rank_like_sentencepiece_bpe(tmp_path):
    tok = Tokenizer(_model_dir(tmp_path) / "tokens.txt")
    assert tok.encode("Kalshi") == ["▁Kal", "shi"]
    assert tok.encode("Hyper Kalshi") == ["▁Hy", "per", "▁Kal", "shi"]
    assert tok.encode("Kalshi™") is None          # a character with no piece
    assert "<unk>" not in tok.bpe_vocab() and "▁Kal\t-3\n" in tok.bpe_vocab()


# ── the decoder ──────────────────────────────────────────────────────────

def test_one_recognizer_and_per_stream_pretokenized_proper_nouns(fake, tmp_path):
    dec = SherpaTransducer(_model_dir(tmp_path), label="parakeet-tdt-0.6b-v2")
    assert dec.capabilities.bias == BiasMode.UNBOUNDED
    assert dec.name == "sherpa:parakeet-tdt-0.6b-v2"
    kw = fake.created[0]
    assert kw["decoding_method"] == "modified_beam_search"
    assert (kw["length_reward"], kw["max_active_paths"], kw["hotwords_score"]) == (0.5, 4, 2.0)
    assert "hotwords_file" not in kw and kw["modeling_unit"] == "bpe"
    assert (tmp_path / "bpe.vocab").exists()

    out = dec.decode(_audio(2), final=True, bias=Bias(hotwords="Kalshi, kalshi, Hyper, a/b"))
    assert out.text == "hello world"
    # Lowercase (mined jargon) and unusable terms never become hotwords.
    assert fake.streams[-1]["hotwords"] == "▁Kal shi/▁Hy per"
    assert fake.streams[-1]["length_reward"] == "0.5"   # per stream, as upstream reads it
    dec.decode(_audio(2), bias=Bias(hotwords="Hyper"))
    dec.decode(_audio(2), bias=Bias())
    assert [s["hotwords"] for s in fake.streams[1:]] == ["▁Hy per", None]
    assert len(fake.created) == 1, "a new hotword set must never reload the model"


def test_sherpa_tokenization_fallback_passes_terms(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(True, True, False))
    dec = SherpaTransducer(_model_dir(tmp_path))
    dec.decode(_audio(1), bias=Bias(hotwords="Kalshi, Hyper Kalshi"))
    assert fake.streams[-1]["hotwords"] == "Kalshi/Hyper Kalshi"


def test_without_the_tdt_fix_decodes_greedy_and_unbiased(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(True, False, False))
    dec = build_decoder(ModelConfig(name=f"sherpa:{_model_dir(tmp_path)}"))
    assert dec.capabilities.bias == BiasMode.NONE
    assert dec.name.endswith("@greedy_search")
    assert dec.decode(_audio(1), bias=Bias(hotwords="Kalshi")).text == "hello world"
    kw = fake.created[-1]
    assert kw["decoding_method"] == "greedy_search" and "length_reward" not in kw
    assert fake.streams[-1]["hotwords"] is None and "length_reward" not in fake.streams[-1]


def test_long_audio_is_decoded_in_pieces_of_at_most_30s(fake, tmp_path):
    dec = SherpaTransducer(_model_dir(tmp_path))
    audio = _audio(70)
    audio[int(25 * 16000):int(25.4 * 16000)] = 0.0   # a pause to cut at
    out = dec.decode(audio, final=True)
    sizes = [s["samples"] for s in fake.streams]
    assert sum(sizes) == len(audio) and all(n <= 30 * 16000 for n in sizes)
    assert len(sizes) == 3 and 24.5 * 16000 < sizes[0] <= 25.5 * 16000
    assert out.text == "hello world hello world hello world"


def test_empty_audio_is_silence_not_an_error(fake, tmp_path):
    dec = SherpaTransducer(_model_dir(tmp_path))
    assert dec.decode(np.zeros(0, dtype=np.float32)).text == ""


def test_decodes_run_concurrently_so_stop_never_queues(monkeypatch, tmp_path):
    gate = threading.Event()
    f = FakeSherpa(block=gate)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", f.module)
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    dec = SherpaTransducer(_model_dir(tmp_path))
    assert dec.draft is dec   # lets stop decode past a pending freeze
    threads = [threading.Thread(target=dec.decode, args=(_audio(1),)) for _ in range(2)]
    for t in threads:
        t.start()
    for _ in range(200):
        if f.running == 2:
            break
        threading.Event().wait(0.01)
    assert f.max_running == 2
    assert not dec.wait_idle(timeout=0.05)
    gate.set()
    for t in threads:
        t.join(5)
    assert dec.wait_idle(timeout=1)


def test_engine_errors_are_decode_errors(fake, tmp_path):
    dec = SherpaTransducer(_model_dir(tmp_path))

    def boom(s):
        raise RuntimeError("ort")
    dec._rec.decode_stream = boom
    with pytest.raises(DecodeError):
        dec.decode(_audio(1))
    assert dec.wait_idle(timeout=1)


# ── default selection ────────────────────────────────────────────────────

def test_default_is_parakeet_only_when_sherpa_has_the_fix(monkeypatch):
    assert catalog.default_model("en") == "small"     # conftest: no sherpa
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(True, False, False))
    assert catalog.default_model("en") == "small"
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(True, True, False))
    assert catalog.default_model("en") == catalog.PARAKEET
    assert catalog.default_model("es") == catalog.PARAKEET_V3
    assert catalog.default_model("ja") == "small"      # outside Parakeet v3's languages


def test_config_resolves_unset_or_auto_name_but_an_explicit_choice_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    path = tmp_path / "c.toml"
    path.write_text('[model]\nlanguage = "en"\n')
    assert config.load(path).model.name == catalog.PARAKEET
    path.write_text('[model]\nname = "auto"\n')
    assert config.load(path).model.name == catalog.PARAKEET
    for explicit in ("small", "/home/u/.local/share/voiceio/learn/models/v0001-small-x/ct2"):
        path.write_text(f'[model]\nname = "{explicit}"\n')
        assert config.load(path).model.name == explicit
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures())
    path.write_text('[model]\nlanguage = "en"\n')
    assert config.load(path).model.name == "small"


def test_runtime_and_whisper_detection():
    assert catalog.runtime_of(catalog.PARAKEET) == "sherpa-onnx"
    assert catalog.runtime_of("sherpa:/models/x") == "sherpa-onnx"
    assert catalog.is_whisper("small") and catalog.is_whisper("/learn/models/v1/ct2")
    assert not catalog.is_whisper(catalog.PARAKEET)
    m = catalog.info(catalog.PARAKEET)
    assert m.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/")
    assert len(m.sha256) == 64 and "huggingface" not in m.url


def test_catalog_model_is_fetched_on_first_use(fake, monkeypatch, tmp_path):
    got = []
    (tmp_path / "m").mkdir()
    model = _model_dir(tmp_path / "m")
    monkeypatch.setattr(fetch, "download", lambda name, **k: got.append(name) or model)
    dec = build_decoder(ModelConfig(name=catalog.PARAKEET))
    assert got == [catalog.PARAKEET] and dec.name == f"sherpa:{catalog.PARAKEET}"


def test_no_download_when_sherpa_is_missing(monkeypatch):
    monkeypatch.setattr(fetch, "download", lambda *a, **k: pytest.fail("downloaded"))
    with pytest.raises(DecodeError, match="sherpa-onnx not installed"):
        build_decoder(ModelConfig(name=catalog.PARAKEET))


# ── download ─────────────────────────────────────────────────────────────

def _tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in (("encoder.int8.onnx", b"e"), ("decoder.int8.onnx", b"d"),
                           ("joiner.int8.onnx", b"j"), ("tokens.txt", TOKENS.encode()),
                           ("test_wavs/0.wav", b"w")):
            info = tarfile.TarInfo(f"sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _serve(monkeypatch, body: bytes):
    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: Resp(body))


def _entry(monkeypatch, body: bytes):
    m = catalog.ModelInfo("pk-test", "sherpa-onnx", "1", "x", BiasMode.UNBOUNDED,
                          url="https://github.com/x.tar.bz2",
                          sha256=hashlib.sha256(body).hexdigest(), size_mb=1)
    monkeypatch.setattr(fetch, "info", lambda name: m if name == "pk-test" else None)


def test_download_verifies_and_unpacks_without_sample_wavs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    body = _tarball()
    _entry(monkeypatch, body)
    _serve(monkeypatch, body)
    seen = []
    path = fetch.download("pk-test", progress=lambda d, t: seen.append((d, t)))
    assert path == tmp_path / "pk-test" and fetch.is_downloaded("pk-test")
    assert not (path / "test_wavs").exists()
    assert seen[-1] == (len(body), len(body))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pk-test"]   # no leftovers


def test_download_with_a_bad_checksum_leaves_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    body = _tarball()
    _entry(monkeypatch, body)
    _serve(monkeypatch, body[:-10] + b"0123456789")
    with pytest.raises(OSError, match="checksum"):
        fetch.download("pk-test")
    assert list(tmp_path.iterdir()) == []


def test_patched_wheel_is_picked_for_this_python_and_cpu():
    sums = ("aa  sherpa_onnx-1.13.8-cp312-cp312-linux_x86_64.whl\n"
            "bb  sherpa_onnx-1.13.8-cp313-cp313-manylinux_2_28_x86_64.whl\n"
            "cc  sherpa_onnx-1.13.8-cp313-cp313-linux_aarch64.whl\n")
    assert fetch.patched_sherpa_wheel(sums, (3, 13), "x86_64") == (
        "sherpa_onnx-1.13.8-cp313-cp313-manylinux_2_28_x86_64.whl", "bb")
    assert fetch.patched_sherpa_wheel(sums, (3, 11), "x86_64") is None


# ── learn / doctor / eval ────────────────────────────────────────────────

def test_learn_says_fine_tunes_are_unused_while_parakeet_decodes(monkeypatch):
    from voiceio.cli import learn
    monkeypatch.setattr(learn, "_cfg", lambda: types.SimpleNamespace(
        model=ModelConfig(name=catalog.PARAKEET)))
    assert "isn't used" in learn._whisper_unused()
    monkeypatch.setattr(learn, "_cfg", lambda: types.SimpleNamespace(
        model=ModelConfig(name="small")))
    assert learn._whisper_unused() == ""


def test_maintain_labels_but_does_not_train_for_an_unused_whisper(monkeypatch, capsys):
    import argparse

    from voiceio.cli import learn
    from voiceio.config import LearnConfig
    monkeypatch.setattr(learn, "_cfg", lambda: types.SimpleNamespace(
        model=ModelConfig(name=catalog.PARAKEET)))
    labeled = []
    monkeypatch.setattr(learn, "cmd_label", lambda a: labeled.append(a))
    monkeypatch.setattr(learn, "_announce_learning", lambda n: None)
    monkeypatch.setattr(learn, "_train", lambda *a, **k: pytest.fail("trained"))
    learn._maintain(argparse.Namespace(), LearnConfig())
    assert labeled and "not retraining" in capsys.readouterr().out


def test_doctor_reports_the_decoder(monkeypatch):
    from voiceio.health import _decoder_status, _learning_status
    cfg = config.Config()
    cfg.model.name = catalog.PARAKEET
    assert "not installed" in _decoder_status(cfg)["Decoder"]
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(True, False, False))
    assert "NO vocabulary biasing" in _decoder_status(cfg)["Decoder"]
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    assert "pre-tokenized" in _decoder_status(cfg)["Decoder"]
    assert "aren't used" in _learning_status(cfg)["Voice model"]
    cfg.model.name = "small"
    assert catalog.PARAKEET in _decoder_status(cfg)["Decoder"]
    cfg.model.language = "es"
    assert catalog.PARAKEET_V3 in _decoder_status(cfg)["Decoder"]
    cfg.model.language = "ja"
    assert _decoder_status(cfg) == {}


def test_eval_results_are_found_for_sherpa_models(monkeypatch, tmp_path):
    import json

    from voiceio.learn import evaluate, metrics
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "eval-1.json").write_text(json.dumps({
        "decoder": f"sherpa:{catalog.PARAKEET}", "dataset": "v1", "mode": "offline",
        "metrics": metrics.METRICS_VERSION, "wer": 0.07}))
    monkeypatch.setattr(evaluate, "learn_dir", lambda: tmp_path)
    assert evaluate.latest(catalog.PARAKEET, "v1", beam=2)["wer"] == 0.07
    assert evaluate.latest("small", "v1") is None


def test_wizard_offers_parakeet_first_only_where_it_can_bias(monkeypatch):
    from voiceio.setup import speech
    assert speech.model_choices("en") is speech.MODELS
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    choices = speech.model_choices("en")
    assert choices[0][0] == catalog.PARAKEET and speech.default_choice(choices) == 0
    assert not any("recommended" in c[2] for c in choices[1:])
    assert speech.model_choices("es") is speech.MODELS
