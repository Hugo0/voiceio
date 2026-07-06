"""Persistent state for the autocorrect mining pipeline.

Tracks a scan cursor (`last_scan_ts`) so repeat runs only mine new history,
a `dismissed` set of terms the user (or repeated adjudication failure) rejected
so they're never proposed again, and a `deferred` map of ambiguous candidates
awaiting more evidence. Stored in ~/.config/voiceio/autocorrect_state.json.

Deferral policy: when evidence-based adjudication can't reach a unanimous
verdict, the candidate is silently deferred rather than queued for a human.
A deferred word is not re-adjudicated for `DEFER_COOLDOWN_SECS`; after
`MAX_DEFER_FAILURES` failed adjudications it is permanently dismissed. This
gives ambiguous-but-real patterns a path to eventual resolution without ever
building a human review queue.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from voiceio.config import CONFIG_DIR

log = logging.getLogger(__name__)

STATE_PATH = CONFIG_DIR / "autocorrect_state.json"

# A deferred candidate is left alone for two weeks before it's worth spending
# tokens re-adjudicating — long enough that new dictation accumulates fresh
# context evidence, short enough to still resolve within a few weekly runs.
DEFER_COOLDOWN_SECS = 14 * 86400

# After this many failed adjudications a candidate is permanently dismissed.
MAX_DEFER_FAILURES = 3


@dataclass
class AutocorrectState:
    """Mining cursor + dismissed terms + deferred candidates, persisted."""

    last_scan_ts: float = 0.0
    dismissed: set[str] = field(default_factory=set)
    # word (lowercased) -> {count: int, last_seen_ts: float, votes_history: list}
    deferred: dict[str, dict] = field(default_factory=dict)

    def is_dismissed(self, term: str) -> bool:
        return term.lower() in self.dismissed

    def dismiss(self, term: str) -> None:
        if term:
            wl = term.lower()
            self.dismissed.add(wl)
            self.deferred.pop(wl, None)

    def defer(
        self, word: str, *,
        votes: list | None = None,
        ts: float | None = None,
        failure: bool = True,
    ) -> None:
        """Record that `word` was deferred rather than acted on.

        `failure=True` (an adjudication that couldn't reach consensus) counts
        toward the dismissal threshold and starts the cooldown clock. Once the
        count reaches `MAX_DEFER_FAILURES` the word is permanently dismissed.

        `failure=False` is a *capacity* deferral (the per-run adjudication cap
        was hit): it neither counts as a failure nor starts a cooldown, so the
        word stays eligible for the very next run.
        """
        if not word:
            return
        wl = word.lower()
        if wl in self.dismissed:
            return
        entry = self.deferred.get(wl) or {
            "count": 0, "last_seen_ts": 0.0, "votes_history": [],
        }
        if failure:
            entry["count"] = int(entry.get("count", 0)) + 1
            entry["last_seen_ts"] = ts if ts is not None else time.time()
        elif ts is not None:
            entry["last_seen_ts"] = ts
        if votes:
            entry.setdefault("votes_history", []).append(votes)
        if failure and entry["count"] >= MAX_DEFER_FAILURES:
            self.dismissed.add(wl)
            self.deferred.pop(wl, None)
        else:
            self.deferred[wl] = entry

    def in_cooldown(self, word: str, now: float | None = None) -> bool:
        """True if `word` is deferred and still within its cooldown window."""
        entry = self.deferred.get(word.lower())
        if not entry:
            return False
        last = float(entry.get("last_seen_ts", 0.0))
        if last <= 0.0:
            # Capacity deferral (no real failure timestamp) — always ready.
            return False
        now = now if now is not None else time.time()
        return (now - last) < DEFER_COOLDOWN_SECS

    def cooldown_words(self, now: float | None = None) -> set[str]:
        """Deferred words still inside their cooldown (skip re-proposing them)."""
        now = now if now is not None else time.time()
        return {w for w in self.deferred if self.in_cooldown(w, now)}

    def ready_deferred(self, now: float | None = None) -> set[str]:
        """Deferred words past their cooldown, ready for re-adjudication."""
        now = now if now is not None else time.time()
        return {w for w in self.deferred if not self.in_cooldown(w, now)}


def _coerce_deferred(raw) -> dict[str, dict]:
    """Validate the persisted deferred map, dropping malformed entries."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for word, entry in raw.items():
        if not isinstance(word, str) or not isinstance(entry, dict):
            continue
        try:
            count = int(entry.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        try:
            ts = float(entry.get("last_seen_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        history = entry.get("votes_history", [])
        if not isinstance(history, list):
            history = []
        out[word.lower()] = {
            "count": count, "last_seen_ts": ts, "votes_history": history,
        }
    return out


def load_state(path: Path | None = None) -> AutocorrectState:
    """Load state, tolerating a missing or malformed file."""
    p = path or STATE_PATH
    if not p.exists():
        return AutocorrectState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AutocorrectState()
    if not isinstance(raw, dict):
        return AutocorrectState()
    dismissed = raw.get("dismissed", [])
    if not isinstance(dismissed, list):
        dismissed = []
    try:
        ts = float(raw.get("last_scan_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    return AutocorrectState(
        last_scan_ts=ts,
        dismissed={str(d).lower() for d in dismissed if isinstance(d, str)},
        deferred=_coerce_deferred(raw.get("deferred", {})),
    )


def save_state(state: AutocorrectState, path: Path | None = None) -> None:
    """Persist state to disk, creating the config dir if needed."""
    p = path or STATE_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "last_scan_ts": state.last_scan_ts,
                    "dismissed": sorted(state.dismissed),
                    "deferred": state.deferred,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Failed to write autocorrect state: %s", e)
