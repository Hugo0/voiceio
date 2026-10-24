"""User feedback: notifications and audio cues.

Uses a persistent sounddevice OutputStream on the 'pulse' device for reliable
playback. The stream stays open while the daemon is warm (idle_audio.py
decides; by default for its lifetime on mains power), avoiding PipeWire's
dropped-audio issue with short-lived clients (pw-play, paplay).
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
from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError as _e:
    from voiceio.platform import desktop_dep_hint
    raise ImportError(desktop_dep_hint("sounddevice")) from _e

log = logging.getLogger(__name__)

_SOUNDS_DIR = Path(__file__).parent / "sounds"
_which = functools.lru_cache(maxsize=16)(shutil.which)

# Output buffer of the cue stream. PortAudio's default ("high", ~35 ms on
# PipeWire/Pulse) underruns under heavy CPU load — an audible glitch in the
# chime; cue_secs() accounts for it.
CUE_LATENCY_SECS = 0.08

_stream: sd.OutputStream | None = None
_stream_lock = threading.Lock()
_sounds: dict[str, np.ndarray] = {}


def open_output_stream(
    samplerate: int = 44100, channels: int = 1, dtype: str = "int16",
    latency: float | str = "high",
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
                device=device, latency=latency,
            )
            label = device or "default"
            log.info("Audio output stream open (device=%s, sr=%d, latency=%.0f ms)",
                     label, samplerate, stream.latency * 1000)
            return stream
        except Exception:
            label = device or "default"
            log.debug("Could not open %s audio output", label, exc_info=True)
    return None


def warm_up() -> None:
    """Pre-load all sounds and open the cue stream."""
    _load_sounds()
    open_cue_stream()


def _load_sounds() -> None:
    for name in ("start", "stop", "commit", "alert"):
        path = _SOUNDS_DIR / f"{name}.wav"
        if name in _sounds or not path.exists():
            continue
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        # Pad with 100ms silence so PipeWire doesn't clip the tail
        pad = np.zeros(int(rate * 0.1), dtype=np.int16)
        _sounds[name] = np.concatenate([data, pad]).reshape(-1, 1)


def open_cue_stream() -> bool:
    """Open and start the cue stream unless it is open. True when open."""
    global _stream
    with _stream_lock:
        if _stream is not None:
            return True
        _load_sounds()
        stream = open_output_stream(samplerate=44100, latency=CUE_LATENCY_SECS)
        if stream is None:
            log.warning("Sound output unavailable — no working audio device found")
            return False
        try:
            stream.start()
        except Exception:
            log.warning("Could not start the cue stream", exc_info=True)
            stream.close()
            return False
        _stream = stream
        return True


def close_cue_stream() -> None:
    """Close the cue stream (idle policy). A cue still playing finishes first:
    the lock waits for its write, and stop() drains the buffer."""
    global _stream
    with _stream_lock:
        stream, _stream = _stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.debug("Closing the cue stream failed", exc_info=True)
    log.debug("Cue stream closed")


def cue_stream_open() -> bool:
    return _stream is not None


def play_record_start() -> None:
    _play("start")


def play_record_stop() -> None:
    _play("stop")


def play_commit_sound() -> None:
    _play("commit")


def play_alert_sound() -> None:
    _play("alert")


# The usual cause, by far: a hardware mic switch, a mute key or a muted
# source. Each look the same to voiceio — a stream that sends only zeros.
MIC_HINT = "Is the mic switched off or muted (privacy switch, mute key)?"


def mic_alert(title: str, body: str) -> None:
    """A dead or muted mic: an alert sound as well as the notification, since
    you are talking, not looking at the screen."""
    _play("alert")
    notify(title, f"{body} {MIC_HINT}", urgent=True)


def cue_secs(name: str) -> float:
    """How long a cue takes to play out; 0 when sound output is off."""
    if _stream is None or name not in _sounds:
        return 0.0
    return len(_sounds[name]) / _stream.samplerate + _stream.latency


def _play(name: str) -> None:
    """Write pre-loaded samples to the persistent stream. Non-blocking."""
    if _stream is None or name not in _sounds:
        return
    threading.Thread(target=_play_sync, args=(name,), daemon=True).start()


def _play_sync(name: str) -> None:
    data = _sounds[name]
    try:
        with _stream_lock:
            if _stream is None:   # closed by the idle policy meanwhile
                return
            _stream.write(data)
    except Exception:
        log.debug("Playback failed for %s", name, exc_info=True)


def notify(title: str, body: str, *, urgent: bool = False) -> None:
    """Send a desktop notification without blocking the caller.

    ``urgent=True`` marks it critical so the desktop keeps it on screen until
    dismissed — for actionable failures (a muted mic) that a 3s toast loses.
    """
    threading.Thread(
        target=_send_notification, args=(title, body, urgent), daemon=True,
    ).start()


def notify_clipboard(text: str) -> None:
    preview = text[:80] + ("\u2026" if len(text) > 80 else "")
    notify("VoiceIO: copied to clipboard", preview)


def _send_notification(title: str, body: str, urgent: bool = False) -> None:
    import sys

    if sys.platform == "win32":
        _send_notification_windows(title, body)
    elif sys.platform == "darwin":
        _send_notification_macos(title, body)
    else:
        _send_notification_linux(title, body, urgent)


def _send_notification_linux(title: str, body: str, urgent: bool = False) -> None:
    if not _which("notify-send"):
        return
    # Critical urgency makes the desktop hold the notification until dismissed;
    # a normal 3s toast is missed while the user is mid-dictation.
    args = ["notify-send", "--app-name=VoiceIO"]
    args += ["-u", "critical"] if urgent else ["-t", "3000"]
    try:
        subprocess.run([*args, title, body], capture_output=True, timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        pass


def notify_actions(title: str, body: str, actions: dict[str, str],
                   on_action: Callable[[str], None]) -> None:
    """A notification with buttons ({key: label}); ``on_action(key)`` runs
    when one is clicked. Linux only; elsewhere, or with a notify-send too old
    for actions, it degrades to a plain notification. Never blocks."""
    import sys
    if sys.platform != "linux" or not _which("notify-send") or not _notify_send_has_actions():
        notify(title, body)
        return

    def wait() -> None:
        args = ["notify-send", "--app-name=VoiceIO", "--wait",
                *(f"--action={k}={label}" for k, label in actions.items()), title, body]
        try:
            picked = subprocess.run(args, capture_output=True, text=True).stdout.strip()
        except OSError:
            return
        if picked in actions:
            on_action(picked)

    threading.Thread(target=wait, daemon=True).start()


@functools.lru_cache(maxsize=1)
def _notify_send_has_actions() -> bool:
    """--action/--wait arrived in libnotify 0.7.9."""
    try:
        return "--action" in subprocess.run(["notify-send", "--help"], capture_output=True,
                                            text=True, timeout=3).stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


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
