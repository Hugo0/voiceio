"""Per-term vocabulary usage stats — the signal that makes ranking possible.

Whisper's hotwords channel fits ~40-60 of Hugo's terms (proper nouns cost 5-7
tokens each against a 223-token ceiling), while vocabulary.txt grows without
bound and is never pruned. Selection is therefore permanent, and something has
to decide which terms get the slots.

Until now nothing could: a term's only per-term state was its line position, so
`load_vocabulary` kept whichever terms happened to be oldest. This module gives
each term a usage record — how often it actually shows up in what you dictate,
and when it was last seen — so the budget goes to terms you use.

Kept deliberately cheap: updated off the hot path from the finalize thread, and
mtime-cached on read like CorrectionDict.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _path() -> Path:
    from voiceio import config

    return config.CONFIG_DIR / "vocab_stats.json"


class VocabStats:
    """Usage records keyed by lowercased term: {"hits": int, "last_seen_ts": float}."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._stats: dict[str, dict] = {}
        self._mtime: float | None = None

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else _path()

    def load(self) -> None:
        """Re-read only when the file changed (mtime-gated)."""
        p = self.path
        try:
            mtime = p.stat().st_mtime
        except OSError:
            self._stats = {}
            self._mtime = None
            return
        if self._mtime == mtime:
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._stats = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            self._stats = {}
        self._mtime = mtime

    def save(self) -> None:
        p = self.path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._stats, indent=1), encoding="utf-8")
            tmp.replace(p)
            from voiceio import config

            config._chmod(p, config._SECURE_FILE)
            self._mtime = p.stat().st_mtime
        except OSError as e:
            log.debug("Could not save vocab stats: %s", e)

    def get(self, term: str) -> dict:
        return self._stats.get(term.lower(), {"hits": 0, "last_seen_ts": 0.0})

    def record(self, terms: list[str], *, now: float | None = None) -> None:
        """Bump hit counts for terms seen in one utterance."""
        now = now if now is not None else time.time()
        for t in terms:
            rec = self._stats.setdefault(t.lower(), {"hits": 0, "last_seen_ts": 0.0})
            rec["hits"] = int(rec.get("hits", 0)) + 1
            rec["last_seen_ts"] = now

    def as_dict(self) -> dict[str, dict]:
        return dict(self._stats)


def terms_present(text: str, terms: list[str]) -> list[str]:
    """Which vocabulary terms appear in `text`, word-boundary anchored.

    Word-boundary rather than substring: `audit._audit_vocabulary` uses a bare
    `in` test, which counts "ISA" inside "isabel". Matching here mirrors
    `corrections.apply` so a hit means what a reader would think it means.
    """
    if not text or not terms:
        return []
    low = text.lower()
    out = []
    for t in terms:
        if not t:
            continue
        # Cheap substring pre-filter before the regex, since this runs per
        # utterance over every term in the vocabulary.
        if t.lower() not in low:
            continue
        if re.search(r"\b" + re.escape(t.lower()) + r"\b", low):
            out.append(t)
    return out


def update_from_text(text: str, terms: list[str], *, path: Path | None = None) -> int:
    """Record vocabulary hits for one finalized utterance. Returns hit count.

    Called from the finalize path, never the recording-start path. Failures are
    swallowed: usage bookkeeping must never break a dictation.
    """
    try:
        hits = terms_present(text, terms)
        if not hits:
            return 0
        stats = VocabStats(path)
        stats.load()
        stats.record(hits)
        stats.save()
        return len(hits)
    except Exception as e:  # noqa: BLE001
        log.debug("vocab stats update failed: %s", e)
        return 0


def bootstrap_from_history(terms: list[str], *, path: Path | None = None) -> int:
    """Seed stats from existing history so ranking works from the first run.

    Without this a fresh stats file ranks everything equally and selection falls
    back to file order — i.e. today's behaviour, where the oldest terms win.
    Hugo has 6035 transcripts already; 330 of 334 terms appear in at least one.
    """
    from voiceio import history

    stats = VocabStats(path)
    stats.load()
    if stats.as_dict():
        return 0  # already seeded; don't double-count

    seeded = 0
    for entry in history.read(limit=0):
        text = entry.get("text") or ""
        ts = entry.get("ts") or 0.0
        hits = terms_present(text, terms)
        if hits:
            stats.record(hits, now=ts)
            seeded += len(hits)
    if seeded:
        stats.save()
    return seeded
