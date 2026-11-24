"""Re-initialising PortAudio, so a reopened mic sees today's devices.

PortAudio enumerates devices and resolves the default input once, at
``Pa_Initialize`` (the ``import sounddevice``). A daemon up for days never sees
a mic plugged in later, nor a new default source: reopening a dead stream
lands on the same stale device. ``refresh()`` terminates and re-initialises
PortAudio so the next open resolves the default, or a configured device name,
against the current list.

``Pa_Terminate`` frees EVERY open PortAudio stream in the process; a Python
stream object left pointing at one would crash on its next call. So every
sounddevice stream in the daemon goes through here:

- the mic (recorder.py): ``refresh(close=...)`` closes it, and its
  ``open_stream`` runs inside ``stream_in_use()``;
- the persistent cue stream (feedback.py): closed by ``refresh()`` and
  reopened after it, its open runs inside ``stream_in_use()``;
- TTS playback (tts/player.py): the whole play, open to close, runs inside
  ``stream_in_use()``, and a refresh asked for meanwhile is skipped.

Imports sounddevice lazily (the base install is headless).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)

# At most one re-init per this many seconds: a mic still dead right after one
# reopens on the device list it already has instead of re-enumerating in a loop.
REFRESH_MIN_INTERVAL_SECS = 30.0

_cond = threading.Condition()
_refreshing = False
_users = 0
_last_refresh = float("-inf")


@contextmanager
def stream_in_use() -> Iterator[None]:
    """Hold while opening or using a sounddevice stream that refresh() does
    not close itself. Waits for a refresh in progress; a refresh asked for
    while any is held is skipped."""
    global _users
    with _cond:
        while _refreshing:
            _cond.wait()
        _users += 1
    try:
        yield
    finally:
        with _cond:
            _users -= 1
            _cond.notify_all()


def refresh(close: Callable[[], None] | None = None, *,
            clock: Callable[[], float] = time.monotonic) -> bool:
    """Close ``close``'s stream and the cue stream, then terminate and
    re-initialise PortAudio. True when it re-initialised. Raises only what
    ``close`` raises (then nothing is re-initialised).

    Skipped (False, after still calling ``close``) when the last one was less
    than REFRESH_MIN_INTERVAL_SECS ago, another runs, or a stream is in use.
    """
    global _refreshing, _last_refresh
    with _cond:
        now = clock()
        if _refreshing or _users or now - _last_refresh < REFRESH_MIN_INTERVAL_SECS:
            skip = ("in use" if _users else "running" if _refreshing else "backoff")
        else:
            skip = ""
            _refreshing = True
            _last_refresh = now
    if skip:
        log.debug("PortAudio re-init skipped (%s)", skip)
        if close is not None:
            close()
        return False

    reopen_cue = False
    try:
        if close is not None:
            close()   # raises to the caller, as a plain close would
        from voiceio import feedback
        reopen_cue = feedback.close_cue_stream()
        try:
            _reinitialize()
            ok = True
        except Exception:
            log.warning("PortAudio re-init failed", exc_info=True)
            ok = False
    finally:
        with _cond:
            _refreshing = False
            _cond.notify_all()
    if reopen_cue:
        from voiceio import feedback
        feedback.open_cue_stream()
    return ok


def _reinitialize() -> None:
    import sounddevice
    # Pa_Initialize is reference-counted: the device list is only rebuilt
    # once the count reaches zero.
    n = sounddevice._initialized
    for _ in range(n):
        sounddevice._terminate()
    for _ in range(max(n, 1)):
        sounddevice._initialize()
    try:
        info = sounddevice.query_devices(kind="input")
        log.info("PortAudio re-initialised: default input is %r (index %s)",
                 info["name"], info["index"])
    except Exception:
        log.info("PortAudio re-initialised: no default input device")
