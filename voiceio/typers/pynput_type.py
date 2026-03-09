"""Pynput text injection backend — macOS (and X11 fallback)."""
from __future__ import annotations

import logging

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


class PynputTyper:
    """Type text via pynput keyboard controller."""

    name = "pynput"

    def __init__(self):
        self._controller = None

    def _get_controller(self):
        if self._controller is None:
            from pynput.keyboard import Controller
            self._controller = Controller()
        return self._controller

    def probe(self) -> ProbeResult:
        try:
            from pynput.keyboard import Controller  # noqa: F401
        except ImportError:
            return ProbeResult(ok=False, reason="pynput not installed",
                               fix_hint="pip install pynput")

        import os
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland":
            return ProbeResult(ok=False, reason="pynput typing does not work on Wayland")

        return ProbeResult(ok=True)

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._get_controller().type(text)

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        from pynput.keyboard import Key
        kb = self._get_controller()
        for _ in range(n):
            kb.press(Key.backspace)
            kb.release(Key.backspace)
