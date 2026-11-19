"""`voiceio learn report`: numbers only, never words or paths."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import pytest

from voiceio import config, history, service
from voiceio.learn import metrics, summary
from voiceio.learn.store import learn_dir

NOW = dt.datetime(2026, 11, 19, 12, 0, 0).timestamp()
DAY = 86400
SECRETS = ("Kalshi", "HyperNEAT", "zorblax", "secret-plan.txt", "alice")


def _stamp(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S,000")


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A week of use whose every store holds words and paths."""
    home = os.path.expanduser("~")
    run = learn_dir() / "models" / "v0002-small-20261115-0300"
    model = str(run / "ct2")
    (run / "ct2").mkdir(parents=True)

    history.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": NOW - 1 * DAY, "text": "Kalshi zorblax HyperNEAT", "raw": "kalchi zorblax",
         "duration": 90.0, "context": f"{home}/secret-plan.txt - alice"},
        {"ts": NOW - 3 * DAY, "text": "zorblax again", "duration": 30.0,
         "audio": f"{home}/.local/state/voiceio/recordings/zorblax.wav"},
        {"ts": NOW - 9 * DAY, "text": "too old", "duration": 600.0},
    ]
    history.HISTORY_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))

    ds = learn_dir() / "datasets" / "v0002"
    ds.mkdir(parents=True)
    runs = learn_dir() / "runs"
    runs.mkdir()
    (runs / "eval-20261115-031500.json").write_text(json.dumps({
        "decoder": f"faster-whisper:{model}", "dataset": "v0002", "mode": "offline",
        "metrics": metrics.METRICS_VERSION, "wer": 0.0612, "term_recall": 0.794,
        "clips": 84, "per_clip": [{"audio": f"{home}/zorblax.wav", "ref": "Kalshi zorblax",
                                   "hyp": "kalchi zorblax"}],
    }))
    (learn_dir() / "labels.jsonl").write_text("".join(json.dumps({
        "audio": f"rec-{i}.wav", "source": "teacher:large-v3-turbo", "ts": ts,
        "segments": [{"start": 0.0, "end": 2.0, "text": "HyperNEAT zorblax"}],
    }) + "\n" for i, ts in enumerate((NOW - DAY, NOW - 2 * DAY, NOW - 20 * DAY))))

    log = tmp_path / "voiceio.log"
    log.write_text("\n".join([
        f"{_stamp(NOW - 20 * DAY)} voiceio.streaming ERROR Final transcribe/apply error",
        f"{_stamp(NOW - 2 * DAY)} voiceio.typers.manager WARNING Typer 'ibus' failed: zorblax, trying fallback",
        f"{_stamp(NOW - 2 * DAY)} voiceio.typers.manager INFO Switched to typer: clipboard",
        f"{_stamp(NOW - DAY)} voiceio.streaming ERROR Final transcribe/apply error",
        "Traceback (most recent call last):",
        f'  File "{home}/voiceio/streaming.py", line 280, in _final',
        "ValueError: Kalshi zorblax",
        f"{_stamp(NOW - DAY)} voiceio.app ERROR Processing failed",
        f"{_stamp(NOW - DAY)} voiceio.tray ERROR Tray icon failed: alice",
    ]) + "\n")

    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "learn_timer_enabled", lambda: True)
    monkeypatch.setattr(service, "learn_last_run", lambda: (NOW - 2 * DAY, "success"))
    cfg = config.Config()
    cfg.model.name = model
    return cfg, [log]


def test_report_counts_the_week(seeded):
    cfg, logs = seeded
    out = summary.build(NOW, cfg, log_paths=logs)
    from voiceio import __version__
    assert out == {
        "schema": 1, "date": "2026-11-19", "window_days": 7, "version": __version__,
        "model_family": "whisper", "model_name": "fine-tune v0002-small-20261115-0300",
        "dictations_7d": 2, "audio_minutes_7d": 2.0,
        "wer": 0.0612, "vocab_recall": 0.794, "eval_dataset": "v0002", "eval_clips": 84,
        "eval_date": "2026-11-15",
        "clipboard_fallbacks_7d": 1, "decoder_errors_7d": 2, "log_days_covered": 7,
        "clips_labelled_7d": 2, "learn_scheduled": True, "learn_ran_7d": True,
        "learn_last_run": "2026-11-17", "learn_last_result": "success",
        "unavailable": {},
    }


def _strings(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def test_no_words_or_paths_ever_leave(seeded):
    cfg, logs = seeded
    out = summary.build(NOW, cfg, log_paths=logs)
    text = summary.to_json(out) + summary.format_text(out)
    for secret in (*SECRETS, os.path.expanduser("~"), str(learn_dir()), "ct2", ".wav"):
        assert secret not in text
    assert not [s for s in _strings(out) if "/" in s]


def test_missing_data_is_null_with_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "has_systemd", lambda: False)
    cfg = config.Config()
    cfg.model.name = "parakeet-tdt-0.6b-v2"
    out = summary.build(NOW, cfg, log_paths=[tmp_path / "missing.log"])
    assert out["model_family"] == "parakeet" and out["model_name"] == "parakeet-tdt-0.6b-v2"
    for key in ("dictations_7d", "audio_minutes_7d", "wer", "clipboard_fallbacks_7d",
                "decoder_errors_7d", "learn_scheduled", "learn_ran_7d"):
        assert out[key] is None
        assert out["unavailable"][key]
    assert out["clips_labelled_7d"] == 0
    assert "n/a — no dataset yet" in summary.format_text(out)


def test_eval_of_another_model_is_not_reported(seeded):
    cfg, logs = seeded
    cfg.model.name = "small"
    out = summary.build(NOW, cfg, log_paths=logs)
    assert out["wer"] is None
    assert out["unavailable"]["wer"].startswith("no stored eval of the active model")


@pytest.mark.parametrize("name,label", [
    ("small", "small"),
    ("whispercpp:http://10.1.2.3:8080", "whisper.cpp server"),
    ("sherpa:/home/alice/models/my-parakeet", "local my-parakeet"),
    ("/opt/models/alice-tuned/ct2", "local ct2"),
])
def test_model_label_is_never_a_path(name, label, monkeypatch):
    from voiceio import report
    monkeypatch.setattr(report, "_identities", lambda: [("alice", "<user>")])
    assert summary.model_label(name) == label.replace("alice", "<user>")


def test_cli_prints_json_and_text(seeded, capsys, monkeypatch):
    from voiceio.cli import learn as learn_cli
    cfg, logs = seeded
    monkeypatch.setattr(config, "load", lambda *a, **k: cfg)
    monkeypatch.setattr(summary, "_log_paths", lambda: logs)
    learn_cli.cmd_report(argparse.Namespace(json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["model_family"] == "whisper"
    learn_cli.cmd_report(argparse.Namespace(json=False))
    text = capsys.readouterr().out
    assert "decoder errors" in text and "zorblax" not in text


def test_learn_last_run_parses_systemctl(monkeypatch):
    import subprocess

    def fake(argv, **kw):
        prop = argv[argv.index("-p") + 1]
        out = {"LastTriggerUSec": "@1763260212\n", "Result": "success\n"}[prop]
        return subprocess.CompletedProcess(argv, 0, stdout=out)

    monkeypatch.setattr(service.subprocess, "run", fake)
    assert service.learn_last_run() == (1763260212.0, "success")

    monkeypatch.setattr(service.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="n/a\n"))
    assert service.learn_last_run() == (None, "")
