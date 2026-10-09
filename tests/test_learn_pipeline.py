"""The personalization pipeline end to end, with fake models."""
from __future__ import annotations

import json
import wave

import numpy as np

from fakes import FakeDecoder
from voiceio import config
from voiceio.learn import augment, evaluate, review, store, train
from voiceio.learn.store import Segment


def _clip(name: str, secs: float, raw: str, text: str | None = None) -> None:
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(config.RECORDINGS_DIR / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.sin(np.arange(int(secs * 16000)) / 9) * 8000).astype(np.int16).tobytes())
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.HISTORY_PATH, "a") as f:
        f.write(json.dumps({"ts": 1.0, "text": text or raw, "raw": raw,
                            "duration": secs, "audio": name}) + "\n")


def test_clips_skip_missing_audio_and_strip_prefix():
    _clip("a.wav", 2, "hello kalchi", "[voice] Hello Kalshi")
    with open(config.HISTORY_PATH, "a") as f:
        f.write(json.dumps({"text": "no audio", "audio": "gone.wav"}) + "\n")
    clips = list(store.clips())
    assert [c.audio for c in clips] == ["a.wav"]
    assert clips[0].final == "Hello Kalshi"


def test_human_label_wins_over_later_teacher_label():
    store.add_label("a.wav", [Segment(0, 1, "human text")], "human")
    store.add_label("a.wav", [Segment(0, 1, "teacher text")], "teacher:x")
    assert store.load_labels()["a.wav"].text == "human text"


def test_windows_respect_whisper_limit():
    segs = tuple(Segment(i * 5, i * 5 + 4.5, f"s{i}") for i in range(12))  # 60s
    ws = store.windows(segs)
    assert all(w.end - w.start <= store.MAX_WINDOW_SECS for w in ws)
    assert " ".join(w.text for w in ws) == " ".join(s.text for s in segs)


def test_split_is_deterministic():
    assert store.split_of("x.wav") == store.split_of("x.wav")
    splits = {store.split_of(f"{i}.wav") for i in range(200)}
    assert splits == {"train", "dev", "test"}


def test_build_review_eval_export_flow(tmp_path):
    names = [f"{i:03d}.wav" for i in range(60)]
    for n in names:
        _clip(n, 3, "we use kalchi daily")
        store.add_label(n, [Segment(0, 3, "we use Kalshi daily")], "teacher:fake")

    items = review.queue(list(store.clips()), store.load_labels(), ["Kalshi"])
    assert items and items[0].reason.startswith("vocabulary")
    answers = iter(["We use Kalshi daily."] + [""] * 100)
    saved = review.run(items[:2], ask=lambda it: next(answers),
                       save=lambda clip, segs: store.add_label(clip.audio, segs, "human"))
    assert saved == 2

    ds = store.build_dataset()
    manifest = json.loads((ds / "manifest.json").read_text())
    assert sum(manifest["counts"].values()) == 60
    assert manifest["counts"]["test"] > 0

    dec = FakeDecoder()
    dec.transcribe.return_value = "we use kalchi daily"
    res = evaluate.evaluate(dec, ds, terms=["Kalshi"])
    assert res.clips == manifest["counts"]["test"]
    assert res.term_recall == 0.0 and res.wer > 0
    dec.transcribe.return_value = "We use Kalshi daily"
    assert evaluate.evaluate(dec, ds, terms=["Kalshi"]).term_recall == 1.0

    counts = train.export(ds, tmp_path / "export")
    assert counts["train"] == manifest["counts"]["train"]
    row = json.loads((tmp_path / "export" / "train.jsonl").read_text().splitlines()[0])
    with wave.open(str(tmp_path / "export" / row["audio"])) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 3 * 16000


class _FakeTTS:
    name = "fake"

    def synthesize(self, text, voice, speed):
        return (np.zeros(22050) + 100).astype(np.int16), 22050


def test_augment_caps_synthetic_share():
    n = augment.synthesize(["Kalshi", "HyperNEAT"], _FakeTTS(), per_term=5)
    assert n == 10
    rows = augment.rows(max_share=0.3, real_rows=7)
    assert len(rows) == 3  # 3 / (7 + 3) = 30%
    assert all("Kalshi" in r["text"] or "HyperNEAT" in r["text"] for r in rows)
