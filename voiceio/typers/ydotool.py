"""Ydotool text injection backend for Wayland via uinput."""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_ydotool_version() -> tuple[int, ...]:
    """Get ydotool major version. Returns (0,) on failure. Cached."""
    try:
        # v1.x prints version, v0.x doesn't support --version
        result = subprocess.run(
            ["ydotool", "--version"], capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            # e.g. "ydotool 1.0.4"
            parts = result.stdout.strip().split()[-1].split(".")
            return tuple(int(p) for p in parts)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return (0,)


def _needs_daemon() -> bool:
    """v1.x needs ydotoold, v0.x talks to /dev/uinput directly."""
    return _get_ydotool_version() >= (1,)


def _ydotoold_running() -> bool:
    """Check if the ydotoold daemon is running."""
    try:
        result = subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return True  # can't check, assume ok


def _has_uinput_access() -> bool:
    """Check if current user can write to /dev/uinput."""
    try:
        return os.access("/dev/uinput", os.W_OK)
    except OSError:
        return False


class YdotoolTyper:
    """Type text via ydotool (Wayland, needs uinput access)."""

    name = "ydotool"

    def probe(self) -> ProbeResult:
        if not shutil.which("ydotool"):
            return ProbeResult(ok=False, reason="ydotool not installed",
                               fix_hint="sudo apt install ydotool")

        if _needs_daemon():
            # v1.x: needs ydotoold running
            if not _ydotoold_running():
                ydotoold_path = shutil.which("ydotoold") or "ydotoold"
                return ProbeResult(
                    ok=False,
                    reason="ydotoold daemon not running",
                    fix_hint=f"sudo {ydotoold_path} &",
                    fix_cmd=["sudo", ydotoold_path],
                )
        else:
            # v0.x: needs /dev/uinput write access
            if not _has_uinput_access():
                return ProbeResult(
                    ok=False,
                    reason="No write access to /dev/uinput",
                    fix_hint="sudo chmod 0666 /dev/uinput  (or add udev rule)",
                    fix_cmd=["sudo", "chmod", "0666", "/dev/uinput"],
                )

        return ProbeResult(ok=True)

    def __init__(self) -> None:
        self._v1 = _get_ydotool_version() >= (1,)

    def type_text(self, text: str) -> None:
        if not text:
            return
        subprocess.run(
            ["ydotool", "type", "--delay", "10", "--key-delay", "2", "--", text],
            check=True, capture_output=True,
        )

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        if self._v1:
            # v1.x: raw keycode:state syntax
            args = []
            for _ in range(n):
                args.extend(["14:1", "14:0"])
            subprocess.run(
                ["ydotool", "key", *args],
                check=True, capture_output=True,
            )
        else:
            # v0.x: key names, batch all backspaces in one call
            keys = ["Backspace"] * n
            subprocess.run(
                ["ydotool", "key", "--key-delay", "2", *keys],
                check=True, capture_output=True,
            )
