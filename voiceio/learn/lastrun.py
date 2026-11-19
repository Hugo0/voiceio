"""The last `voiceio learn maintain` run: when it started and ended, and what
it did. `learn report` reads it before asking systemd, which can't say
whether a run happened at all (a unit that never ran has Result=success) and
whose timestamp output differs between versions.

Counts and times only, like the report it feeds.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

from voiceio import config

# outcome: running (or killed before it could say), retrained, labelled
# (labels only), skipped (nothing new, or nobody was away), failed, stopped.
OUTCOMES = ("running", "retrained", "labelled", "skipped", "failed", "stopped")


def path() -> Path:
    return config.LOG_DIR / "learn-last-run.json"


def read() -> dict | None:
    """The record, or None when there is none (or it is unreadable)."""
    try:
        rec = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    if not (isinstance(rec, dict) and isinstance(rec.get("started"), (int, float))
            and rec.get("outcome") in OUTCOMES):
        return None
    return rec


def _write(rec: dict) -> None:
    try:
        config.secure_write(path(), json.dumps(rec) + "\n")
    except OSError:   # a record is never worth failing the run over
        pass


def _label_count() -> int:
    from voiceio.learn import store
    try:
        return len(store.load_labels())
    except OSError:
        return 0


def mark_retrained() -> None:
    """Called by the run (or its --when-idle worker) once it trained a model."""
    rec = read()
    if rec is not None:
        _write({**rec, "retrained": True})


@contextlib.contextmanager
def recording() -> Iterator[None]:
    """Record the run inside this block. Clips labelled are counted from the
    label store, so a worker stopped and resumed still counts all of them."""
    before = _label_count()
    _write({"started": time.time(), "ended": None, "outcome": "running",
            "scheduled": bool(os.environ.get("INVOCATION_ID"))})
    outcome: str | None = "failed"
    try:
        yield
        outcome = None
    except SystemExit as e:
        outcome = "failed" if e.code else None
        raise
    except KeyboardInterrupt:
        outcome = "stopped"
        raise
    finally:
        rec = read() or {"started": time.time(), "scheduled": False}
        labelled = max(0, _label_count() - before)
        if outcome is None:
            outcome = ("retrained" if rec.get("retrained")
                       else "labelled" if labelled else "skipped")
        rec.pop("retrained", None)
        _write({**rec, "ended": time.time(), "outcome": outcome, "labelled": labelled})
