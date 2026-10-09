import json

from voiceio import config
from voiceio.learn.clean import clean


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_clean_dry_run_then_apply():
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RECORDINGS_DIR / "real.wav").write_bytes(b"x")
    _write(config.HISTORY_PATH, [
        {"text": "Hello world"},                                   # fixture
        {"text": "Hello", "duration": 2.0, "audio": "real.wav"},   # real mic test
        {"text": "ship the Kalshi bot"},
    ])
    _write(config.TRACES_PATH, [
        {"audio": "gone.wav", "latency": {"transcribe": 0.0}},
        {"audio": "real.wav", "latency": {"transcribe": 1.2}},
    ])
    _write(config.POSTCORRECT_PAIRS_PATH,
           [{"before": "Chris chat is the tool", "model": "x"}] * 5
           + [{"before": "a real one", "model": "my/override-model"},
              {"before": "keep me", "model": "google/gemini-2.5-flash-lite"}])

    reports = {r.path.name: r for r in clean(apply=False)}
    assert reports["history.jsonl"].removed == 1
    assert len(config.HISTORY_PATH.read_text().splitlines()) == 3  # dry run

    reports = {r.path.name: r for r in clean(apply=True)}
    assert [json.loads(l)["text"] for l in config.HISTORY_PATH.read_text().splitlines()] \
        == ["Hello", "ship the Kalshi bot"]
    assert reports[config.TRACES_PATH.name].removed == 1
    assert reports["postcorrect_pairs.jsonl"].removed == 6
    assert list(config.HISTORY_PATH.parent.glob("history.jsonl.*.bak"))
