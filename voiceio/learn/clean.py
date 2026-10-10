"""Remove test-fixture rows from the user's data files.

Before the test suite was sandboxed, tests wrote into the real state dir:
"Hello world" history entries, 0-second decode traces, and postcorrect pairs
for fixture models. Left in place they skew every statistic and would end up
as training data. This finds them by signature — never by content alone — and
by default only reports; `apply=True` rewrites each file after a backup.
"""
from __future__ import annotations

import collections
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from voiceio import config

_FIXTURE_TEXTS = frozenset({
    "hello", "hello world", "hello world.", "hello.", "thank you.",
})
_FIXTURE_MODELS = frozenset({"my/override-model"})
# Real dictation rarely repeats verbatim this many times; fixtures do. Repeats
# alone aren't proof (re-running a real clip repeats it too), so a repeated
# input also has to be absent from history or written in a test-speed burst.
_DUPLICATE_THRESHOLD = 5
_BURST_SECS = 0.25


@dataclass
class FileReport:
    path: Path
    total: int = 0
    removed: int = 0
    reasons: collections.Counter = field(default_factory=collections.Counter)


def _read(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _audio_missing(row: dict) -> bool:
    name = row.get("audio")
    return not name or not (config.RECORDINGS_DIR / name).exists()


def _history_reason(row: dict) -> str | None:
    if row.get("text", "").strip().lower() not in _FIXTURE_TEXTS:
        return None
    # Fixtures were written without a duration; a real dictation always has
    # one, even after retention pruned its recording.
    if not row.get("duration") and _audio_missing(row):
        return "fixture text, no recording"
    return None


def _trace_reason(row: dict) -> str | None:
    if (row.get("latency") or {}).get("transcribe") == 0.0 and _audio_missing(row):
        return "zero-time decode, no recording"
    return None


def _ts(row: dict) -> float | None:
    try:
        return float(row["ts"])
    except (KeyError, TypeError, ValueError):
        return None


def _burst_inputs(rows: list[dict]) -> set:
    """Inputs logged twice within `_BURST_SECS` — faster than any human."""
    times = collections.defaultdict(list)
    for r in rows:
        if (t := _ts(r)) is not None:
            times[r.get("before")].append(t)
    return {b for b, ts in times.items()
            if any(y - x < _BURST_SECS for x, y in zip(sorted(ts), sorted(ts)[1:]))}


def _pairs_reasons(rows: list[dict], history: str = "") -> list[str | None]:
    counts = collections.Counter(r.get("before") for r in rows)
    bursts = _burst_inputs(rows)

    def dictated(r: dict) -> bool:
        return any(s and s.lower()[:40] in history
                   for s in (r.get("before"), r.get("after")))

    out = []
    for r in rows:
        if r.get("model") in _FIXTURE_MODELS:
            out.append("fixture model")
        elif counts[r.get("before")] < _DUPLICATE_THRESHOLD:
            out.append(None)
        elif r.get("before") in bursts:
            out.append("repeated input, written in a burst")
        elif not dictated(r):
            out.append("repeated input, never dictated")
        else:
            out.append(None)
    return out


def _clean_file(path: Path, reasons: list[str | None], rows: list[dict],
                apply: bool) -> FileReport:
    report = FileReport(path, total=len(rows))
    keep = []
    for row, reason in zip(rows, reasons):
        if reason:
            report.removed += 1
            report.reasons[reason] += 1
        else:
            keep.append(row)
    if apply and report.removed:
        backup = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        shutil.copy2(path, backup)
        config.secure_write(
            path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    return report


def clean(apply: bool = False) -> list[FileReport]:
    """Find (and with `apply`, remove) fixture rows. Returns per-file counts."""
    reports = []
    history = ""
    if config.HISTORY_PATH.exists():
        history = "\n".join(r.get("text", "") for r in _read(config.HISTORY_PATH)).lower()
    targets = [
        (config.HISTORY_PATH, lambda rows: [_history_reason(r) for r in rows]),
        (config.TRACES_PATH, lambda rows: [_trace_reason(r) for r in rows]),
        (config.POSTCORRECT_PAIRS_PATH, lambda rows: _pairs_reasons(rows, history)),
    ]
    for path, classify in targets:
        if not path.exists():
            continue
        rows = _read(path)
        reports.append(_clean_file(path, classify(rows), rows, apply))
    return reports
