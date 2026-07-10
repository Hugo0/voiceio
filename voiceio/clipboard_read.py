"""Read/write text from/to the system clipboard or primary selection."""
from __future__ import annotations

import logging
import shutil
import subprocess

from voiceio.platform import detect

log = logging.getLogger(__name__)

_TIMEOUT = 2


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def read_text() -> str | None:
    """Read selected/copied text from the clipboard.

    On Linux, tries PRIMARY selection first (mouse-selected text), then
    falls back to CLIPBOARD (ctrl+c copied text).
    Returns None if no text available or tools missing.
    """
    p = detect()

    if p.is_windows:
        try:
            import pyperclip
            text = pyperclip.paste()
            return text.strip() if text and text.strip() else None
        except (ImportError, Exception):
            log.debug("pyperclip not available for clipboard read")
            return None

    if p.is_mac:
        return _run(["pbpaste"])

    # Linux — use platform detection for Wayland vs X11
    if p.is_wayland:
        if shutil.which("wl-paste"):
            text = _run(["wl-paste", "-p", "--no-newline"])
            if text:
                return text
            return _run(["wl-paste", "--no-newline"])
    else:
        if shutil.which("xclip"):
            text = _run(["xclip", "-o", "-selection", "primary"])
            if text:
                return text
            return _run(["xclip", "-o", "-selection", "clipboard"])
        if shutil.which("xsel"):
            text = _run(["xsel", "-o", "--primary"])
            if text:
                return text
            return _run(["xsel", "-o", "--clipboard"])

    log.debug("No clipboard tool found")
    return None


def _copy_via(cmd: list[str], text: str) -> bool:
    """Pipe text into a copy command. Returns True on success."""
    try:
        result = subprocess.run(
            cmd, input=text.encode(), capture_output=True, timeout=_TIMEOUT,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def copy_text(text: str) -> bool:
    """Copy text to the system CLIPBOARD selection.

    Best-effort: returns False (never raises) when no tool is available
    or the copy fails.
    """
    if not text:
        return False
    p = detect()

    if p.is_windows:
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except Exception:
            log.debug("pyperclip not available for clipboard write")
            return False

    if p.is_mac:
        return _copy_via(["pbcopy"], text)

    if p.is_wayland:
        if shutil.which("wl-copy"):
            return _copy_via(["wl-copy", "--"], text)
    else:
        if shutil.which("xclip"):
            return _copy_via(["xclip", "-selection", "clipboard"], text)
        if shutil.which("xsel"):
            return _copy_via(["xsel", "-i", "--clipboard"], text)

    log.debug("No clipboard tool found for copy")
    return False
