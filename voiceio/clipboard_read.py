"""Read text from the system clipboard or primary selection."""
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
