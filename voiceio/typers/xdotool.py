"""Xdotool text injection backend for X11."""
from __future__ import annotations

import logging
import shutil
import subprocess

from voiceio.backends import ProbeResult
from voiceio.typers.base import KeystrokeTyper

log = logging.getLogger(__name__)


class XdotoolTyper(KeystrokeTyper):
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

    def _type(self, text: str) -> None:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
            check=True, capture_output=True,
        )

    def _delete(self, n: int) -> None:
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "--delay", "12"] + ["BackSpace"] * n,
            check=True, capture_output=True,
        )
