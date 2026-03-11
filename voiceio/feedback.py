"""User feedback: notifications and audio cues.

Uses a persistent sounddevice OutputStream on the 'pulse' device for reliable
playback. The stream stays open for the lifetime of the daemon, avoiding
PipeWire's dropped-audio issue with short-lived clients (pw-play, paplay).
The 'pulse' device routes through PulseAudio/PipeWire's session manager,
so audio always goes to the same output as other desktop apps.
"""
from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

_SOUNDS_DIR = Path(__file__).parent / "sounds"
_which = functools.lru_cache(maxsize=16)(shutil.which)

_stream: sd.OutputStream | None = None
_stream_lock = threading.Lock()
_sounds: dict[str, np.ndarray] = {}


def open_output_stream(
    samplerate: int = 44100, channels: int = 1, dtype: str = "int16",
) -> sd.OutputStream | None:
    """Open an output stream, trying 'pulse' first on Linux.

    Returns the stream (not yet started) or None if no device works.
    Used by both feedback.py (persistent cue stream) and tts/player.py.
    """
    import sys
    devices_to_try = ["pulse", None] if sys.platform.startswith("linux") else [None]

    for device in devices_to_try:
        try:
            stream = sd.OutputStream(
                samplerate=samplerate, channels=channels, dtype=dtype,
                device=device,
            )
            label = device or "default"
            log.info("Audio output stream open (device=%s, sr=%d)", label, samplerate)
            return stream
        except Exception:
            label = device or "default"
            log.debug("Could not open %s audio output", label, exc_info=True)
    return None


def warm_up() -> None:
    """Open a persistent output stream and pre-load all sounds."""
    global _stream

    # Pre-load all WAV files
    for name in ("start", "stop", "commit"):
        path = _SOUNDS_DIR / f"{name}.wav"
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        # Pad with 100ms silence so PipeWire doesn't clip the tail
        pad = np.zeros(int(rate * 0.1), dtype=np.int16)
        _sounds[name] = np.concatenate([data, pad]).reshape(-1, 1)

    _stream = open_output_stream(samplerate=44100)
    if _stream:
        _stream.start()
    else:
        log.warning("Sound output unavailable — no working audio device found")


def play_record_start() -> None:
    _play("start")


def play_record_stop() -> None:
    _play("stop")


def play_commit_sound() -> None:
    _play("commit")


def _play(name: str) -> None:
    """Write pre-loaded samples to the persistent stream. Non-blocking."""
    if _stream is None or name not in _sounds:
        return
    threading.Thread(target=_play_sync, args=(name,), daemon=True).start()


def _play_sync(name: str) -> None:
    data = _sounds[name]
    try:
        with _stream_lock:
            _stream.write(data)
    except Exception:
        log.debug("Playback failed for %s", name, exc_info=True)


def notify_clipboard(text: str) -> None:
    preview = text[:80] + ("\u2026" if len(text) > 80 else "")
    threading.Thread(
        target=_send_notification,
        args=("VoiceIO: copied to clipboard", preview),
        daemon=True,
    ).start()


def _send_notification(title: str, body: str) -> None:
    import sys

    if sys.platform == "win32":
        _send_notification_windows(title, body)
    elif sys.platform == "darwin":
        _send_notification_macos(title, body)
    else:
        _send_notification_linux(title, body)


def _send_notification_linux(title: str, body: str) -> None:
    if not _which("notify-send"):
        return
    try:
        subprocess.run(
            ["notify-send", "--app-name=VoiceIO", "-t", "3000", title, body],
            capture_output=True, timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _send_notification_windows(title: str, body: str) -> None:
    try:
        from win11toast import notify
        notify(title, body, app_id="VoiceIO", duration="short")
    except ImportError:
        log.debug("win11toast not installed, skipping notification")
    except Exception:
        log.debug("Windows notification failed", exc_info=True)


def _send_notification_macos(title: str, body: str) -> None:
    # Escape backslashes and double quotes for AppleScript string literals
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_body = body.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_body}" with title "{safe_title}"'],
            capture_output=True, timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass
