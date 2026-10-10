"""Wtype text injection backend for wlroots compositors (Sway, Hyprland)."""
from __future__ import annotations

import logging
import shutil
import subprocess

from voiceio.backends import ProbeResult
from voiceio.typers.base import KeystrokeTyper

log = logging.getLogger(__name__)


class WtypeTyper(KeystrokeTyper):
    """Type text via wtype (wlroots-based Wayland compositors)."""

    name = "wtype"

    def probe(self) -> ProbeResult:
        if not shutil.which("wtype"):
            from voiceio.platform import pkg_install
            return ProbeResult(ok=False, reason="wtype not installed",
                               fix_hint=pkg_install("wtype"))

        import os
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session != "wayland":
            return ProbeResult(ok=False, reason="wtype requires Wayland")

        # wtype doesn't work on GNOME/Mutter, only wlroots compositors
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "gnome" in desktop:
            return ProbeResult(
                ok=False,
                reason="wtype does not work on GNOME (Mutter doesn't support virtual keyboard protocol)",
                fix_hint="Use ydotool instead.",
            )

        return ProbeResult(ok=True)

    def _type(self, text: str) -> None:
        subprocess.run(
            ["wtype", "--", text],
            check=True, capture_output=True,
        )

    def _delete(self, n: int) -> None:
        # Batch all backspaces into one call: wtype -k BackSpace -k BackSpace ...
        args = []
        for _ in range(n):
            args.extend(["-k", "BackSpace"])
        subprocess.run(
            ["wtype", *args],
            check=True, capture_output=True,
        )
