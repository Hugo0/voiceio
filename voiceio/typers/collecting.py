"""A typer that types into a string instead of into a window.

`StreamingSession` emits its output through a `StreamingTyper` — normally IBus
preedit or a keystroke injector. Headless callers (a server, a test) want the
same text without a desktop: `CollectingTyper` implements the full protocol
against an in-memory buffer, so the session's preedit/commit/delete logic runs
exactly as it does on the desktop and the result is readable as a string.

Preedit is modelled honestly: `text` is committed text plus the live preedit,
which is what a user would see on screen. That distinction matters because the
session only commits once, at the end — a server reading `committed` mid-stream
would see nothing.
"""
from __future__ import annotations

import threading

__all__ = ["CollectingTyper"]


class CollectingTyper:
    """In-memory `StreamingTyper`. Never fails, never blocks."""

    name = "collecting"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._committed = ""
        self._preedit = ""

    # ── TyperBackend ─────────────────────────────────────────────────────
    def probe(self):
        from voiceio.backends import ProbeResult
        return ProbeResult(ok=True)

    def type_text(self, text: str) -> None:
        with self._lock:
            self._committed += text

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self._committed = self._committed[:-n] if n <= len(self._committed) else ""

    # ── StreamingTyper ───────────────────────────────────────────────────
    def update_preedit(self, text: str) -> None:
        with self._lock:
            self._preedit = text

    def commit_text(self, text: str) -> None:
        with self._lock:
            self._preedit = ""
            self._committed += text

    def clear_preedit(self) -> None:
        with self._lock:
            self._preedit = ""

    def shutdown(self) -> None:
        pass

    # ── readback ─────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """What would be on screen: committed text plus the live preedit."""
        with self._lock:
            return self._committed + self._preedit

    @property
    def committed(self) -> str:
        """Only what has actually been committed."""
        with self._lock:
            return self._committed

    def reset(self) -> None:
        with self._lock:
            self._committed = ""
            self._preedit = ""
