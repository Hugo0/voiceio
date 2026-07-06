"""Teacher-model audit + corrections fire-logging.

The teacher model is always MOCKED (a fake transcribe fn) — tests never
download or load faster_whisper.
"""
from __future__ import annotations

import json
import time
import wave

import pytest

from voiceio import audit, config
from voiceio.corrections import CorrectionDict

DAY = 86400.0


# ── fixtures / helpers ────────────────────────────────────────────────────

class _Cfg:
    """Minimal stand-in for the parts of Config that run_audit touches."""

    class model:
        vocabulary_file = ""


def _make_wav(name: str, secs: float = 1.0) -> str:
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RECORDINGS_DIR / name
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * secs))
    return name


def _write_history(entries: list[dict]) -> None:
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.HISTORY_PATH, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _write_fire_log(records: list[dict]) -> None:
    config.CORRECTIONS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CORRECTIONS_AUDIT_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_teacher(mapping: dict[str, str]):
    """transcribe(wav_path) -> text, keyed by wav filename."""
    def _tx(wav_path):
        return mapping.get(wav_path.name, "")
    return _tx


def _corrections(rules: dict[str, str]) -> None:
    config.CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CORRECTIONS_PATH.write_text(json.dumps(rules), encoding="utf-8")


# ── corrections fire-logging ──────────────────────────────────────────────

def test_fire_logging_writes_record():
    cd = CorrectionDict()
    cd.add("teh", "the")
    out = cd.apply("i saw teh cat today")
    assert out == "i saw the cat today"

    records = [json.loads(x) for x in
               config.CORRECTIONS_AUDIT_PATH.read_text().splitlines()]
    assert len(records) == 1
    rec = records[0]
    assert rec["wrong"] == "teh"
    assert rec["right"] == "the"
    assert "teh" in rec["snippet"]
    assert "ts" in rec


def test_fire_logging_never_raises_on_oserror(tmp_path, monkeypatch):
    # Point the audit path's parent at an existing *file* so mkdir raises.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(config, "CORRECTIONS_AUDIT_PATH", blocker / "audit.jsonl")

    cd = CorrectionDict()
    cd.add("teh", "the")
    # Must still correct and must not raise.
    assert cd.apply("teh dog") == "the dog"


def test_no_fire_no_log():
    cd = CorrectionDict()
    cd.add("teh", "the")
    cd.apply("nothing to correct here")
    assert not config.CORRECTIONS_AUDIT_PATH.exists()


# ── strike / confirm / retire ─────────────────────────────────────────────

def test_rule_retired_on_strikes():
    now = time.time()
    _corrections({"kubernets": "kubernetes"})
    _make_wav("clip1.wav")
    _write_history([{
        "ts": now, "text": "kubernetes rocks", "raw": "kubernets rocks",
        "audio": "clip1.wav", "duration": 1.0,
    }])
    # Two fires; teacher heard the "wrong" word — it was real → 2 strikes.
    _write_fire_log([
        {"ts": now, "wrong": "kubernets", "right": "kubernetes", "snippet": "x"},
        {"ts": now, "wrong": "kubernets", "right": "kubernetes", "snippet": "x"},
    ])

    report = audit.run_audit(
        _Cfg, transcribe=_fake_teacher({"clip1.wav": "kubernets is a typo word"}),
    )

    assert "kubernets" in report.rules_retired
    # Removed from the live dictionary.
    assert "kubernets" not in CorrectionDict().list_all()
    # Re-blocked so mining won't relearn it.
    assert audit.is_retired("kubernets")


def test_rule_confirmed_not_retired():
    now = time.time()
    _corrections({"teh": "the"})
    _make_wav("clip1.wav")
    _write_history([{
        "ts": now, "text": "the cat", "raw": "teh cat",
        "audio": "clip1.wav", "duration": 1.0,
    }])
    _write_fire_log([
        {"ts": now, "wrong": "teh", "right": "the", "snippet": "x"},
    ])

    report = audit.run_audit(
        _Cfg, transcribe=_fake_teacher({"clip1.wav": "the cat sat"}),
    )
    assert "teh" in report.rules_confirmed
    assert "teh" not in report.rules_retired
    assert not audit.is_retired("teh")


def test_inactivity_expiry():
    now = time.time()
    _corrections({"staleword": "stale"})
    # A fire, but 70 days ago → past the 60-day inactivity window.
    _write_fire_log([
        {"ts": now - 70 * DAY, "wrong": "staleword", "right": "stale", "snippet": "x"},
    ])

    report = audit.run_audit(_Cfg, transcribe=_fake_teacher({}))
    assert "staleword" in report.rules_expired
    assert "staleword" not in CorrectionDict().list_all()


def test_recent_rule_not_expired():
    now = time.time()
    _corrections({"freshword": "fresh"})
    _write_fire_log([
        {"ts": now - 5 * DAY, "wrong": "freshword", "right": "fresh", "snippet": "x"},
    ])
    report = audit.run_audit(_Cfg, transcribe=_fake_teacher({}))
    assert "freshword" not in report.rules_expired


# ── vocabulary verification / aging ───────────────────────────────────────

def test_vocabulary_confirm_and_aging():
    now = time.time()
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (config.CONFIG_DIR / "vocabulary.txt").write_text(
        "Kubernetes\nStaleword\n", encoding="utf-8",
    )
    _make_wav("clip1.wav")
    _write_history([
        {"ts": now, "text": "kubernetes note", "audio": "clip1.wav", "duration": 1.0},
        # Staleword last appeared 100 days ago in a live transcript.
        {"ts": now - 100 * DAY, "text": "staleword mentioned once"},
    ])

    report = audit.run_audit(
        _Cfg, transcribe=_fake_teacher({"clip1.wav": "Kubernetes is up"}),
    )

    assert "Kubernetes" in report.vocab_confirmed
    assert "Staleword" in report.vocab_aged
    # Aged term moved to the bottom (falls out of the truncated hotword budget).
    lines = (config.CONFIG_DIR / "vocabulary.txt").read_text().split()
    assert lines.index("Kubernetes") < lines.index("Staleword")


# ── metrics line ──────────────────────────────────────────────────────────

def test_metrics_line_written():
    now = time.time()
    _make_wav("clip1.wav")
    _write_history([{
        "ts": now, "text": "hello world", "raw": "helo world",
        "audio": "clip1.wav", "duration": 1.0,
    }])

    audit.run_audit(_Cfg, transcribe=_fake_teacher({"clip1.wav": "hello world"}))

    lines = config.METRICS_PATH.read_text().splitlines()
    assert len(lines) == 1
    m = json.loads(lines[0])
    assert m["entries_audited"] == 1
    assert set(m) == {
        "ts", "entries_audited", "audio_secs", "drift_wer",
        "rules_confirmed", "rules_retired", "rules_expired",
    }


# ── drift alarm + rollback ────────────────────────────────────────────────

def test_drift_regression_triggers_rollback(monkeypatch):
    from voiceio import snapshots

    now = time.time()
    # Prior metrics: steady low drift.
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "a", encoding="utf-8") as f:
        for _ in range(4):
            f.write(json.dumps({"ts": now, "drift_wer": 0.05}) + "\n")

    # Snapshot a known-good corrections file.
    _corrections({"teh": "the"})
    (config.CONFIG_DIR / "vocabulary.txt").write_text("term\n", encoding="utf-8")
    snap = snapshots.snapshot("pre-mining")
    # Mining then "damaged" the live file.
    _corrections({"bad": "worse"})

    _make_wav("clip1.wav")
    _write_history([{
        "ts": now, "text": "one two three four five",
        "raw": "one two three four five",
        "audio": "clip1.wav", "duration": 1.0,
    }])

    notes = []
    monkeypatch.setattr(audit, "_notify", lambda r: notes.append(r))

    report = audit.run_audit(
        _Cfg,
        transcribe=_fake_teacher({"clip1.wav": "completely different words entirely"}),
        snapshot_dir=snap,
    )

    assert report.drift_wer > 0.5
    assert report.rolled_back is True
    # Live corrections restored to the pre-mining snapshot.
    assert json.loads(config.CORRECTIONS_PATH.read_text()) == {"teh": "the"}


def test_no_rollback_without_regression():
    now = time.time()
    with open(config.METRICS_PATH, "a", encoding="utf-8") as f:
        for _ in range(4):
            config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            f.write(json.dumps({"ts": now, "drift_wer": 0.5}) + "\n")

    _make_wav("clip1.wav")
    _write_history([{
        "ts": now, "text": "hello world", "raw": "hello world",
        "audio": "clip1.wav", "duration": 1.0,
    }])
    report = audit.run_audit(
        _Cfg, transcribe=_fake_teacher({"clip1.wav": "hello world"}), snapshot_dir=None,
    )
    assert report.rolled_back is False


def test_audio_budget_caps_sampling():
    now = time.time()
    for i in range(5):
        _make_wav(f"clip{i}.wav", secs=1.0)
        _write_history([{
            "ts": now - i, "text": f"line {i}", "raw": f"line {i}",
            "audio": f"clip{i}.wav", "duration": 1.0,
        }])
    seen = []

    def _tx(wav_path):
        seen.append(wav_path.name)
        return "line"

    report = audit.run_audit(_Cfg, transcribe=_tx, max_audio_secs=2.0)
    # Budget of 2s over 1s clips → at most 2 transcribed.
    assert report.entries_audited <= 2
    assert len(seen) <= 2
