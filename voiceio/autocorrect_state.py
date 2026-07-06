"""Persistent state for the autocorrect mining pipeline.

Tracks a scan cursor (`last_scan_ts`) so repeat runs only mine new history,
and a `dismissed` set of terms the user rejected so they're never proposed
again. Stored in ~/.config/voiceio/autocorrect_state.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from voiceio.config import CONFIG_DIR

log = logging.getLogger(__name__)

STATE_PATH = CONFIG_DIR / "autocorrect_state.json"


@dataclass
class AutocorrectState:
    """Mining cursor + dismissed terms, persisted across runs."""

    last_scan_ts: float = 0.0
    dismissed: set[str] = field(default_factory=set)

    def is_dismissed(self, term: str) -> bool:
        return term.lower() in self.dismissed

    def dismiss(self, term: str) -> None:
        if term:
            self.dismissed.add(term.lower())


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
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Failed to write autocorrect state: %s", e)
