"""User feedback: notifications and audio cues."""
from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_SOUNDS_DIR = Path(__file__).parent / "sounds"

# Cache tool lookups; they don't change during a session
_which = functools.lru_cache(maxsize=16)(shutil.which)


def play_record_start() -> None:
    """Play a subtle sound when recording starts. Runs async."""
    threading.Thread(target=_play_wav, args=(_SOUNDS_DIR / "start.wav",), daemon=True).start()


def play_record_stop() -> None:
    """Play a subtle sound when recording stops. Runs async."""
    threading.Thread(target=_play_wav, args=(_SOUNDS_DIR / "stop.wav",), daemon=True).start()


def play_commit_sound() -> None:
    """Play a short success sound when text is committed. Runs async."""
    threading.Thread(target=_play_wav, args=(_SOUNDS_DIR / "commit.wav",), daemon=True).start()


def notify_clipboard(text: str) -> None:
    """Show desktop notification that text is in clipboard.

    Runs async so it never blocks the typing pipeline.
    """
    preview = text[:80] + ("\u2026" if len(text) > 80 else "")
    threading.Thread(
        target=_send_notification,
        args=("VoiceIO: copied to clipboard", preview),
        daemon=True,
    ).start()


def _send_notification(title: str, body: str) -> None:
    """Send a desktop notification via notify-send (GNOME/freedesktop)."""
    if not _which("notify-send"):
        return
    try:
        subprocess.run(
            ["notify-send", "--app-name=VoiceIO", "-t", "3000", title, body],
            capture_output=True, timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _play_wav(path: Path) -> None:
    """Play a WAV file via paplay, aplay, or pw-play."""
    if not path.exists():
        return
    for player, args in [
        ("paplay", [str(path)]),
        ("pw-play", [str(path)]),
        ("aplay", ["-q", str(path)]),
    ]:
        if not _which(player):
            continue
        try:
            subprocess.run(
                [player, *args],
                capture_output=True, timeout=3,
            )
            return
        except (subprocess.TimeoutExpired, OSError):
            continue
