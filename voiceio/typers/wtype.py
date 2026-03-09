"""Wtype text injection backend — wlroots compositors (Sway, Hyprland)."""
from __future__ import annotations

import logging
import shutil
import subprocess

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


class WtypeTyper:
    """Type text via wtype (wlroots-based Wayland compositors)."""

    name = "wtype"

    def probe(self) -> ProbeResult:
        if not shutil.which("wtype"):
            return ProbeResult(ok=False, reason="wtype not installed",
                               fix_hint="sudo apt install wtype")

        import os
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session != "wayland":
            return ProbeResult(ok=False, reason="wtype requires Wayland")

        # wtype doesn't work on GNOME/Mutter — only wlroots compositors
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "gnome" in desktop:
            return ProbeResult(
                ok=False,
                reason="wtype does not work on GNOME (Mutter doesn't support virtual keyboard protocol)",
                fix_hint="Use ydotool instead.",
            )

        return ProbeResult(ok=True)

    def type_text(self, text: str) -> None:
        if not text:
            return
        subprocess.run(
            ["wtype", "--", text],
            check=True, capture_output=True,
        )

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        # Batch all backspaces into one call: wtype -k BackSpace -k BackSpace ...
        args = []
        for _ in range(n):
            args.extend(["-k", "BackSpace"])
        subprocess.run(
            ["wtype", *args],
            check=True, capture_output=True,
        )
