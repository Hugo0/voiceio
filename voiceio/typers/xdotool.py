"""Xdotool text injection backend for X11."""
from __future__ import annotations

import logging
import shutil
import subprocess

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


class XdotoolTyper:
    """Type text via xdotool (X11)."""

    name = "xdotool"

    def probe(self) -> ProbeResult:
        if not shutil.which("xdotool"):
            from voiceio.platform import pkg_install
            return ProbeResult(ok=False, reason="xdotool not installed",
                               fix_hint=pkg_install("xdotool"))

        import os
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland":
            return ProbeResult(ok=False, reason="xdotool does not work on Wayland",
                               fix_hint="Use ydotool or wtype instead.")

        return ProbeResult(ok=True)

    def type_text(self, text: str) -> None:
        if not text:
            return
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
            check=True, capture_output=True,
        )

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "--delay", "12"] + ["BackSpace"] * n,
            check=True, capture_output=True,
        )
