"""A learning run that is stopped part-way carries on where it was."""
from __future__ import annotations

import argparse
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from voiceio import config
from voiceio.cli import learn as cli
from voiceio.learn import evaluate, store, train
from voiceio.learn.store import Segment


def _clip(name: str) -> None:
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(config.RECORDINGS_DIR / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(3 * 16000, dtype=np.int16).tobytes())
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.HISTORY_PATH, "a") as f:
        f.write(json.dumps({"ts": 1.0, "text": "x", "raw": "x", "duration": 3,
                            "audio": name}) + "\n")
    store.add_label(name, [Segment(0, 3, "hello there")], "teacher:fake")


class Stopped(Exception):
    """Stands in for the watcher ending the worker mid-stage."""


def _fake_work(monkeypatch, train_fn):
    evals, trained = [], []
    monkeypatch.setattr(cli, "cmd_label", lambda a: None)
    monkeypatch.setattr(cli, "_announce_learning", lambda min_new: None)
    monkeypatch.setattr(train, "missing_train_deps", lambda: [])

    def run_eval(model, ds):
        evals.append(model)
        return SimpleNamespace(wer=0.1, term_recall=0.5, repetitions=0, per_clip=[])
    monkeypatch.setattr(cli, "_run_eval", run_eval)
    monkeypatch.setattr(cli, "_train", lambda ds, **k: trained.append(ds.name) or train_fn(ds))
    monkeypatch.setattr(evaluate, "promotion_verdict", lambda c, b: (False, "table"))
    return evals, trained


def test_a_stopped_retrain_carries_on_instead_of_waiting_for_more_clips(monkeypatch, capsys):
    for i in range(3):
        _clip(f"{i}.wav")
    run = store.learn_dir() / "models" / "v0001-small-20261019-0822"

    def stopped(ds):
        raise Stopped
    evals, trained = _fake_work(monkeypatch, stopped)
    with pytest.raises(Stopped):
        cli.cmd_maintain(argparse.Namespace(min_new=3, base="small", epochs=1))
    assert trained == ["v0001"]

    # Before: v0001 counted as done, so it waited for 3 more clips past it.
    evals, trained = _fake_work(monkeypatch, lambda ds: run)
    cli.cmd_maintain(argparse.Namespace(min_new=3, base="small", epochs=1))
    out = capsys.readouterr().out
    assert "carrying on with v0001" in out and trained == ["v0001"]
    assert store.latest_dataset().name == "v0001"          # no second dataset built

    cli.cmd_maintain(argparse.Namespace(min_new=3, base="small", epochs=1))
    assert "retraining waits for 3" in capsys.readouterr().out   # decided: done with it


def test_resume_reuses_saved_scores_and_a_finished_model(monkeypatch):
    for i in range(3):
        _clip(f"{i}.wav")
    ds = store.build_dataset()
    cli._set_pending(ds)
    run = store.learn_dir() / "models" / f"{ds.name}-small-20261019-0822"
    (run / "ct2").mkdir(parents=True)
    (run / "ct2" / "model.bin").write_text("")
    current = config.load().model.name
    saved = {current: {"wer": 0.2, "term_recall": 0.5, "repetitions": 0, "per_clip": []}}
    monkeypatch.setattr(evaluate, "latest", lambda model, dataset, beam=5: saved.get(model))
    evals, trained = _fake_work(monkeypatch, lambda ds: pytest.fail("already trained"))
    cli.cmd_maintain(argparse.Namespace(min_new=50, base="small", epochs=1))
    assert evals == [str(run / "ct2")]        # only the candidate was still unscored
    assert cli._pending_dataset() is None


def test_pending_marker_survives_only_with_its_dataset():
    for i in range(2):
        _clip(f"{i}.wav")
    ds = store.build_dataset()
    cli._set_pending(ds)
    assert cli._pending_dataset() == ds
    cli._set_pending(None)
    assert cli._pending_dataset() is None
    (store.learn_dir() / "maintain-pending.json").write_text('{"dataset": "v0099"}')
    assert cli._pending_dataset() is None and not cli._pending_path().exists()


def test_a_half_built_dataset_is_never_the_latest(monkeypatch):
    _clip("a.wav")
    real = store.json.dumps
    calls = []

    def dies_at_manifest(obj, *a, **k):
        if isinstance(obj, dict) and "schema" in obj:
            raise Stopped
        calls.append(1)
        return real(obj, *a, **k)
    monkeypatch.setattr(store.json, "dumps", dies_at_manifest)
    with pytest.raises(Stopped):
        store.build_dataset()
    monkeypatch.setattr(store.json, "dumps", real)
    assert store.latest_dataset() is None
    assert store.build_dataset().name == "v0001"


def test_train_resumes_the_unfinished_run_of_the_same_dataset_and_base(monkeypatch):
    for i in range(2):
        _clip(f"{i}.wav")
    ds = store.build_dataset()
    folder = store.learn_dir() / "exports" / ds.name
    folder.mkdir(parents=True)
    for name in ("train.jsonl", "dev.jsonl", train.HOTWORDS_FILE):
        (folder / name).write_text("")
    models = store.learn_dir() / "models"
    old = models / f"{ds.name}-small-20261018-0330"          # finished before
    (old / "ct2").mkdir(parents=True)
    (old / "ct2" / "model.bin").write_text("")
    stopped = models / f"{ds.name}-small-20261019-0822"      # stopped mid-training
    stopped.mkdir()
    (stopped / train.CHECKPOINT).write_text("x")
    (models / f"{ds.name}-small.en-20261019-0900").mkdir()   # another base: not ours
    got = {}
    monkeypatch.setattr(train, "require_train_deps", lambda: None)
    monkeypatch.setattr(train, "export", lambda *a, **k: pytest.fail("re-exported"))
    monkeypatch.setattr(train, "train", lambda folder, out, cfg: got.update(out=out, cfg=cfg)
                        or out / "hf")

    def convert(hf, out):
        out.mkdir()
        (out / "model.bin").write_text("")
    monkeypatch.setattr(train, "convert", convert)
    config.set_value("learn", "threads", 3)
    assert cli._train(ds, base="small", epochs=1) == stopped and got["out"] == stopped
    assert got["cfg"].threads == 3
    assert not (stopped / train.CHECKPOINT).exists()        # finished: resume state dropped
    assert cli._finished_run(ds, "small") == stopped


# ── the training loop itself ─────────────────────────────────────────────

def _rows(n=10):
    rng = np.random.default_rng(7)
    return [{"audio": f"w{i}", "x": rng.normal(size=3).tolist(), "y": float(rng.normal())}
            for i in range(n)]


def _fit(tmp_path, *, stop_at=None, checkpoint_secs=0.0, fp="fp"):
    """fit() on a tiny model whose forward draws from mix_rng and dropout, so
    every random state a resume must restore is in play."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    mix_rng = np.random.default_rng(1)
    calls = [0]

    def forward(chunk, *, fixed=False):
        calls[0] += 1
        if stop_at is not None and calls[0] == stop_at:
            raise Stopped
        x = torch.tensor([r["x"] for r in chunk])
        noise = 0.0 if fixed else float(mix_rng.random())
        y = torch.tensor([[r["y"] + noise] for r in chunk])
        return torch.nn.functional.mse_loss(model(x), y)
    cfg = train.TrainConfig(epochs=3, batch_size=3, lr=1e-2)
    rows = _rows()
    best = train.fit(model, forward, rows[:7], rows[7:], cfg, checkpoint=tmp_path / "ck.pt",
                     fp=fp, order_rng=np.random.default_rng(0), mix_rng=mix_rng,
                     save_best=lambda: None, checkpoint_secs=checkpoint_secs)
    return model, best, calls[0]


@pytest.mark.parametrize("stop_at", [2, 4, 6, 9])   # mid-epoch, at a dev pass, epoch 2, 3
def test_a_resumed_run_takes_exactly_the_steps_of_an_uninterrupted_one(tmp_path, stop_at):
    torch = pytest.importorskip("torch")
    (tmp_path / "a").mkdir()
    whole, best_whole, calls_whole = _fit(tmp_path / "a")
    assert calls_whole == 12                    # 3 epochs × (3 train steps + 1 dev pass)
    (tmp_path / "b").mkdir()
    with pytest.raises(Stopped):
        _fit(tmp_path / "b", stop_at=stop_at)
    resumed, best_resumed, calls_resumed = _fit(tmp_path / "b")
    # It carried on (at most the step in flight is redone), and ended where
    # the uninterrupted run did, to the last bit of every random draw.
    assert calls_resumed <= calls_whole - (stop_at - 1) + 1
    for p, q in zip(whole.parameters(), resumed.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
    assert best_resumed == pytest.approx(best_whole)


def test_progress_for_other_data_or_settings_starts_over(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    with pytest.raises(Stopped):
        _fit(tmp_path / "a", stop_at=4)
    assert _fit(tmp_path / "a", fp="other")[2] == 12        # all of it again
    assert "starting over" in capsys.readouterr().out
    (tmp_path / "a" / "ck.pt").write_text("not a checkpoint")
    _fit(tmp_path / "a")                                   # unreadable: starts over, no crash


def test_fingerprint_follows_data_and_settings_not_threads(tmp_path):
    for name in ("train.jsonl", "dev.jsonl"):
        (tmp_path / name).write_text('{"audio": "a"}\n')
    fp = train.fingerprint(tmp_path, train.TrainConfig())
    assert train.fingerprint(tmp_path, train.TrainConfig(threads=3)) == fp
    assert train.fingerprint(tmp_path, train.TrainConfig(epochs=9)) != fp
    assert train.fingerprint(tmp_path, train.TrainConfig(), device="cuda") != fp
    (tmp_path / "train.jsonl").write_text('{"audio": "b"}\n')
    assert train.fingerprint(tmp_path, train.TrainConfig()) != fp


def test_encoder_cache_persists_across_runs_only_for_the_same_data(tmp_path):
    keys = ["train/0.wav", "train/1.wav", "dev/0.wav"]
    cache = train.EncoderCache(tmp_path / train.ENCODER_CACHE, keys, frames=4, width=2, fingerprint="a")
    assert cache.missing(keys) == keys and cache.reused == 0
    value = np.arange(8, dtype=np.float16).reshape(1, 4, 2)
    cache.put(["train/1.wav"], value)
    del cache

    again = train.EncoderCache(tmp_path / train.ENCODER_CACHE, keys, frames=4, width=2, fingerprint="a")
    assert again.reused == 1 and again.missing(keys) == ["train/0.wav", "dev/0.wav"]
    np.testing.assert_array_equal(again.get(["train/1.wav"]), value)
    del again

    other = train.EncoderCache(tmp_path / train.ENCODER_CACHE, keys, frames=4, width=2, fingerprint="b")
    assert other.missing(keys) == keys
    del other
    train.discard_resume_state(tmp_path)
    assert not list(tmp_path.glob("encoder-cache*"))
