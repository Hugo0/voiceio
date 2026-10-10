"""The personalization pipeline end to end, with fake models."""
from __future__ import annotations

import json

import pytest
import wave

import numpy as np

from fakes import FakeDecoder
from voiceio import config
from voiceio.learn import augment, evaluate, metrics, review, store, train
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

    # Streaming replays through the dictation path and times stop→text.
    res = evaluate.evaluate(dec, ds, terms=["Kalshi"], limit=2, streaming=True, speed=100)
    assert res.mode == "streaming" and res.clips == 2 and res.failures == 0
    assert res.term_recall == 1.0 and len(res.waits) == 2
    assert "stop→text" in res.summary()

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


def test_learn_jobs_reexec_into_an_idle_scope(monkeypatch):
    import sys

    from voiceio.cli import learn
    monkeypatch.delenv("VOICEIO_IDLE_SCOPE")
    monkeypatch.setattr(learn, "_idle_scope_available", lambda: True)
    monkeypatch.setattr(sys, "orig_argv", ["python3", "-m", "voiceio", "learn", "label"])
    calls = []

    def fake_exec(file, argv):
        calls.append(argv)
        raise SystemExit  # execvp never returns

    monkeypatch.setattr(learn.os, "execvp", fake_exec)
    with pytest.raises(SystemExit):
        learn._yield_cpu()
    argv = calls[0]
    assert "CPUWeight=idle" in argv
    assert argv[argv.index("--") + 1:] == [sys.executable, "-m", "voiceio", "learn", "label"]
    assert learn.os.environ["VOICEIO_IDLE_SCOPE"] == "1"  # the child won't loop


def test_promotion_requires_a_winning_eval():
    base = {"wer": 0.10, "term_recall": 0.40, "repetitions": 0}
    better = {"wer": 0.101, "term_recall": 0.70, "repetitions": 0}   # vocab win, WER within slack
    worse = {"wer": 0.13, "term_recall": 0.70, "repetitions": 0}
    assert evaluate.promotion_verdict(better, base)[0]
    assert not evaluate.promotion_verdict(worse, base)[0]
    assert not evaluate.promotion_verdict({**better, "repetitions": 2}, base)[0]


def test_latest_eval_matches_model_dataset_and_mode():
    runs = store.learn_dir() / "runs"
    runs.mkdir(parents=True)
    for i, (dec, ds, mode) in enumerate([("faster-whisper:small", "v0001", "offline"),
                                         ("faster-whisper:small+draft", "v0001", "streaming"),
                                         ("faster-whisper:small", "v0002", "offline")]):
        (runs / f"eval-2026010{i}.json").write_text(json.dumps(
            {"decoder": dec, "dataset": ds, "mode": mode, "wer": i,
             "metrics": metrics.METRICS_VERSION}))
    (runs / "eval-20260109.json").write_text(json.dumps(   # newer, but stale scoring
        {"decoder": "faster-whisper:small", "dataset": "v0001", "wer": 9}))
    assert evaluate.latest("small", "v0001")["wer"] == 0
    assert evaluate.latest("small", "v0001", mode="streaming")["wer"] == 1
    assert evaluate.latest("medium", "v0001") is None


def test_build_drops_suspect_teacher_segments_but_trusts_humans():
    _clip("t.wav", 20, "x")
    store.add_label("t.wav", [Segment(0, 5, "ship the Kalshi bot"),
                              Segment(5, 6, "Thank you."),              # filler over silence
                              Segment(6, 7, "one two three four five six seven eight")],  # 8 w/s
                    "teacher:fake")
    _clip("h.wav", 6, "x")
    store.add_label("h.wav", [Segment(0, 5, "ship it"), Segment(5, 6, "Thank you.")], "human")
    dropped: dict[str, int] = {}
    kept = store._train_windows(store.load_labels()["t.wav"], dropped)
    assert [store._merge(g).text for g in kept] == ["ship the Kalshi bot"]
    assert dropped == {"filler hallucination": 1, "too many words for the audio": 1}
    teacher = store.Label("t.wav", (Segment(0, 5, "ship the Kalshi bot"),
                                    Segment(5, 6, "Thank you.")), "teacher:fake", 0)
    assert [store._merge(g).text for g in store._train_windows(teacher, {})] \
        == ["ship the Kalshi bot"]
    # A dropped segment between good ones splits the window: its audio (maybe
    # real speech) must never sit inside a window whose text omits it.
    mid = store.Label("m.wav", (Segment(0, 5, "hello there"),
                                Segment(5, 6, "one two three four five six seven eight"),
                                Segment(6, 10, "I am fine")), "teacher:fake", 0)
    groups = store._train_windows(mid, {})
    assert [(g[0].start, g[-1].end) for g in groups] == [(0, 5), (6, 10)]
    human = store._train_windows(store.load_labels()["h.wav"], {})
    assert [store._merge(g).text for g in human] == ["ship it Thank you."]


def test_lr_schedule_warms_up_then_decays_to_zero():
    f = train.lr_schedule(100)
    assert f(0) == 0.1 and f(9) == 1.0
    assert f(10) == 1.0 and 0.0 < f(55) < 1.0 and f(100) == 0.0


def test_pipeline_audit_attributes_changes_to_stages():
    from voiceio.core.pipeline import Pipeline, StageResult
    from voiceio.learn.store import Clip, Label

    class Swap:
        name, final_only = "swap", False

        def __init__(self, a, b):
            self.a, self.b = a, b

        def __call__(self, text, ctx):
            return StageResult(text.replace(self.a, self.b))

    clips = [Clip("a.wav", 0, 3, raw="we use kalchi", final="we used Kalshi!!"),
             Clip("b.wav", 0, 3, raw="no one knows", final="no one knows")]
    labels = {c.audio: Label(c.audio, (Segment(0, 3, ref),), "teacher:x", 0)
              for c, ref in zip(clips, ["we use Kalshi", "no one knows"])}
    pipeline = Pipeline([Swap("kalchi", "Kalshi"), Swap("one", "1")])
    audit = evaluate.pipeline_audit(clips, labels, pipeline)
    fix, harm = audit.stages
    assert (audit.clips, audit.raw_errors) == (2, 1)
    assert (fix.changed, fix.better, fix.worse) == (1, 1, 0)
    assert (harm.changed, harm.better, harm.worse) == (1, 0, 1)
    c = audit.committed  # vs. the local result: "used" is worse; "one" (not "1") better
    assert (c.changed, c.better, c.worse) == (2, 1, 1)
    assert "decoder output" in audit.summary()


def test_maintain_waits_for_enough_new_clips(monkeypatch, capsys):
    import argparse

    from voiceio.cli import learn as cli
    for i in range(6):
        _clip(f"{i}.wav", 3, "x")
        store.add_label(f"{i}.wav", [Segment(0, 3, "hello there")], "teacher:fake")
    ds = store.build_dataset()
    assert store.dataset_clips(ds) == {f"{i}.wav" for i in range(6)}
    _clip("new.wav", 3, "x")
    store.add_label("new.wav", [Segment(0, 3, "a new one")], "teacher:fake")

    monkeypatch.setattr(cli, "cmd_label", lambda args: None)
    monkeypatch.setattr(train, "missing_train_deps", lambda: [])
    monkeypatch.setattr(cli, "_train", lambda *a, **k: pytest.fail("must not retrain"))
    cli.cmd_maintain(argparse.Namespace(min_new=5, base="small", epochs=1))
    assert "1 new labeled clips since v0001" in capsys.readouterr().out


def test_maintain_without_train_extra_labels_and_exits_cleanly(monkeypatch, capsys):
    """A nightly timer must not fail every night, or spend an eval, over an
    optional install."""
    import argparse

    from voiceio.cli import learn as cli
    labeled = []
    monkeypatch.setattr(cli, "cmd_label", lambda args: labeled.append(True))
    monkeypatch.setattr(train, "missing_train_deps", lambda: ["torch", "peft"])
    monkeypatch.setattr(store, "build_dataset", lambda: pytest.fail("must not build"))
    monkeypatch.setattr(cli, "_run_eval", lambda *a, **k: pytest.fail("must not eval"))
    cli.cmd_maintain(argparse.Namespace(min_new=5, base="small", epochs=1))
    assert labeled and "python-voiceio[train]" in capsys.readouterr().out


def test_dev_mix_is_stable_per_row():
    assert train._row_draws("a.wav") == train._row_draws("a.wav")
    assert train._row_draws("a.wav") != train._row_draws("b.wav")
    assert all(0 <= x < 1 for x in train._row_draws("c.wav"))


def test_clip_without_windows_is_not_new_forever():
    _clip("long.wav", 40, "x")
    store.add_label("long.wav", [Segment(0, 40, "one very long segment")], "teacher:fake")
    ds = store.build_dataset()           # >28s segment: no windows, still considered
    assert "long.wav" in store.dataset_clips(ds)


def test_status_shows_scores_and_suggests_a_winning_run(capsys):
    import argparse

    from voiceio.cli import learn as cli
    _clip("s.wav", 3, "x")
    store.add_label("s.wav", [Segment(0, 3, "hi")], "teacher:fake")
    ds = store.build_dataset()
    run = store.learn_dir() / "models" / "v0001-small-x"
    (run / "ct2").mkdir(parents=True)
    runs = store.learn_dir() / "runs"
    runs.mkdir()
    for i, (dec, wer, rec) in enumerate([("faster-whisper:small", 0.10, 0.5),
                                         (f"faster-whisper:{run / 'ct2'}", 0.08, 0.8)]):
        (runs / f"eval-2026010{i}.json").write_text(json.dumps({
            "decoder": dec, "dataset": ds.name, "mode": "offline", "wer": wer,
            "term_recall": rec, "repetitions": 0, "clips": 5, "gold": 0,
            "metrics": metrics.METRICS_VERSION}))
    cli.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "small (active)" in out and "WER 8.0%" in out
    assert "next: voiceio learn promote v0001-small-x" in out


def test_paired_comparison_reports_direction_and_uncertainty():
    clips = [f"{i}.wav" for i in range(20)]
    cur = {"per_clip": [{"file": f, "wer": 0.10} for f in clips]}
    clearly_better = {"per_clip": [{"file": f, "wer": 0.05} for f in clips]}
    p = evaluate.paired(clearly_better, cur)
    assert (p["better"], p["worse"], p["clips"]) == (20, 0, 20) and p["hi"] < 0
    mixed = {"per_clip": [{"file": f, "wer": 0.10 + (0.05 if i % 2 else -0.05)}
                          for i, f in enumerate(clips)]}
    p = evaluate.paired(mixed, cur)
    assert p["lo"] < 0 < p["hi"]
    base = {"wer": 0.1, "term_recall": 0.5, "repetitions": 0}
    _, table = evaluate.promotion_verdict({**base, **mixed}, {**base, **cur})
    assert "may be luck" in table
    assert evaluate.paired({"per_clip": []}, cur) is None


def test_wait_summary_p90_never_exceeds_max():
    r = evaluate.EvalResult("d", "test", waits=[2.54, 4.63, 3.45, 3.10, 6.83, 3.07])
    s = r.wait_summary()
    p90 = float(s.split("p90 ")[1].split("s")[0])
    assert p90 <= 6.83


def test_maintain_auto_promote_switches_restarts_and_says_how_to_undo(monkeypatch):
    import argparse
    from types import SimpleNamespace

    from voiceio import feedback, service
    from voiceio.cli import learn as cli
    from voiceio.learn import evaluate
    for i in range(3):
        _clip(f"{i}.wav", 3, "x")
        store.add_label(f"{i}.wav", [Segment(0, 3, "hello there")], "teacher:fake")
    run = store.learn_dir() / "models" / "v0001-small-x"
    calls = []
    monkeypatch.setattr(cli, "cmd_label", lambda a: None)
    monkeypatch.setattr(train, "missing_train_deps", lambda: [])
    monkeypatch.setattr(cli, "_run_eval", lambda *a, **k: SimpleNamespace(
        wer=0.1, term_recall=0.5, repetitions=0, per_clip=[]))
    monkeypatch.setattr(cli, "_train", lambda *a, **k: run)
    monkeypatch.setattr(evaluate, "promotion_verdict", lambda c, b: (True, "table"))
    monkeypatch.setattr(cli, "cmd_promote", lambda a: calls.append(("promote", a.run)))
    monkeypatch.setattr(service, "is_installed", lambda: True)
    monkeypatch.setattr(service, "restart_when_idle", lambda: calls.append(("restart",)))
    monkeypatch.setattr(feedback, "notify", lambda t, b, urgent=False: calls.append(("notify", b)))

    cli.cmd_maintain(argparse.Namespace(min_new=1, base="small", epochs=1, auto_promote=True))
    assert calls[:2] == [("promote", "v0001-small-x"), ("restart",)]
    assert "voiceio learn rollback" in calls[2][1]
