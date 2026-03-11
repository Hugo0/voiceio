"""Contextual CLI hints — actionable, silenceable, frequency-limited."""
from __future__ import annotations

import json
import logging
import os
import sys

from voiceio.config import CONFIG_DIR

log = logging.getLogger(__name__)

_HINTS_PATH = CONFIG_DIR / "hints.json"
_MAX_SHOWS = 3


def hint(hint_id: str, message: str) -> None:
    """Show a hint to stderr if conditions are met.

    Guards: stderr is a TTY, VOICEIO_NO_HINTS != "1", show count < MAX_SHOWS.
    Output: dim text on stderr.
    """
    if not sys.stderr.isatty():
        return
    if os.environ.get("VOICEIO_NO_HINTS") == "1":
        return

    state = _load_state()
    count = state.get(hint_id, 0)
    if count >= _MAX_SHOWS:
        return

    # Show hint
    dim = "\033[2m"
    reset = "\033[0m"
    suffix = ""
    if not state:  # first hint ever shown
        suffix = "  (silence hints: VOICEIO_NO_HINTS=1)"
    print(f"{dim}hint: {message}{suffix}{reset}", file=sys.stderr)

    # Update state
    state[hint_id] = count + 1
    _save_state(state)


def _load_state() -> dict:
    try:
        return json.loads(_HINTS_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _HINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HINTS_PATH.write_text(json.dumps(state))
    except Exception:
        log.debug("Failed to save hints state", exc_info=True)
