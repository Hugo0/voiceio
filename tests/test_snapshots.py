"""Snapshot / prune / rollback roundtrip for the rule lifecycle."""
from __future__ import annotations

import json

from voiceio import config, snapshots


def _write_state(corrections: dict, vocab: str) -> None:
    config.CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CORRECTIONS_PATH.write_text(json.dumps(corrections), encoding="utf-8")
    (config.CONFIG_DIR / "vocabulary.txt").write_text(vocab, encoding="utf-8")


def test_snapshot_roundtrip_restores_files():
    _write_state({"teh": "the"}, "Kubernetes\nOllama\n")
    snap = snapshots.snapshot("pre-mining")

    assert snap.exists()
    assert (snap / "corrections.json").exists()
    assert (snap / "vocabulary.txt").exists()

    # Mutate live files, then roll back.
    config.CORRECTIONS_PATH.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    (config.CONFIG_DIR / "vocabulary.txt").write_text("garbage\n", encoding="utf-8")

    assert snapshots.rollback(snap) is True
    assert json.loads(config.CORRECTIONS_PATH.read_text()) == {"teh": "the"}
    assert (config.CONFIG_DIR / "vocabulary.txt").read_text() == "Kubernetes\nOllama\n"


def test_snapshot_skips_missing_sources():
    # No corrections/vocab files yet — snapshot still succeeds, dir is empty.
    snap = snapshots.snapshot("empty")
    assert snap.exists()
    assert not (snap / "corrections.json").exists()
    assert snapshots.rollback(snap) is False


def test_prune_keeps_last_eight(monkeypatch):
    _write_state({"teh": "the"}, "term\n")
    # Force distinct, ordered snapshot dir names (timestamp granularity is 1s).
    names = [f"202401{i:02d}-000000" for i in range(1, 13)]
    it = iter(names)
    monkeypatch.setattr(snapshots.time, "strftime", lambda *a, **k: next(it))

    for _ in names:
        snapshots.snapshot("run")

    remaining = sorted(p.name for p in config.SNAPSHOTS_DIR.iterdir())
    assert len(remaining) == snapshots.KEEP
    # Oldest pruned, newest kept.
    assert remaining[0] == "20240105-000000-run"
    assert remaining[-1] == "20240112-000000-run"
