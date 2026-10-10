"""Idle audio policy: when the mic and chime streams stay open between dictations.

An open capture stream feeds the 1 s pre-buffer, so the first word is never
clipped, but it also keeps PipeWire's graph and the audio device awake (about
2% of a core and a suspended-never sound card on a laptop) and shows the mic
as in use. ``[audio] keep_warm`` trades the two:

  always  open for the daemon's lifetime (how voiceio always worked before)
  auto    open on mains power; on battery, closed ``warm_minutes`` after the
          last dictation
  never   closed shortly after each dictation, opened on the hotkey

The hotkey path only ever opens (``ensure_open``); closing happens on the
watchdog tick or a short timer, under the daemon's state lock and only while
it is IDLE with no output or speech in flight. After a cold open there is no
pre-buffer: the ring holds only what arrived since the open, so the first
syllable after a long idle may be clipped while the device wakes up.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from voiceio import power

if TYPE_CHECKING:
    from voiceio.recorder import AudioRecorder

log = logging.getLogger(__name__)

MODES = ("always", "auto", "never")
# "never": close this long after the last activity, so the commit chime
# (played as the final text lands) is not cut off.
NEVER_GRACE_SECS = 3.0


def normalize_mode(mode: str) -> str:
    if mode in MODES:
        return mode
    log.warning("audio.keep_warm %r unknown (%s), using auto", mode, " | ".join(MODES))
    return "auto"


def describe(mode: str, warm_minutes: float, *,
             on_battery: Callable[[], bool] | None = None) -> str:
    """Doctor line for ``voiceio doctor``."""
    mode = normalize_mode(mode)
    window = f"{warm_minutes:g} min"
    if mode == "always":
        return "kept warm (keep_warm = always)"
    if mode == "never":
        return "closed (opens on the hotkey; keep_warm = never)"
    if (on_battery or power.on_battery)():
        return f"closes after {window} on battery (on battery now)"
    return f"kept warm (on AC) / closes after {window} on battery"


class IdleAudio:
    """Opens and closes the recorder's mic stream and the cue stream.

    ``can_close`` says the daemon is IDLE with nothing in flight (no finalizer,
    no TTS playback); ``lock`` is the daemon's state lock, held while closing
    so a hotkey can never start a recording on a stream being torn down.
    """

    def __init__(
        self,
        mode: str,
        warm_minutes: float,
        *,
        recorder: AudioRecorder,
        cues: Callable[[], bool],
        lock: threading.Lock,
        can_close: Callable[[], bool],
        on_battery: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mode = normalize_mode(mode)
        self._warm_secs = max(0.0, warm_minutes * 60)
        self._recorder = recorder
        self._cues = cues
        self._lock = lock
        self._can_close = can_close
        self._on_battery = on_battery or (lambda: power.on_battery())
        self._clock = clock
        self._last_activity: float | None = None
        self._timer: threading.Timer | None = None

    # ── Policy ──────────────────────────────────────────────────────────

    def wanted(self) -> bool:
        """Should the streams be open right now (while idle)?"""
        if self.mode == "always":
            return True
        if self.mode == "auto" and not self._on_battery():
            return True
        window = self._warm_secs if self.mode == "auto" else NEVER_GRACE_SECS
        last = self._last_activity
        return last is not None and self._clock() - last < window

    def touch(self) -> None:
        """A dictation started or ended, or a cue played: restart the window."""
        self._last_activity = self._clock()
        if self.mode == "never":
            self._schedule(NEVER_GRACE_SECS + 0.1)

    # ── Open (hotkey path) / close (tick) ───────────────────────────────

    def start(self) -> None:
        """Daemon startup: open when the policy wants it now."""
        if self.mode == "always" or self.wanted():
            self.ensure_open()
        else:
            self._recorder.park()
            log.info("Audio closed while idle (keep_warm=%s)", self.mode)

    def ensure_open(self) -> bool:
        """Open both streams if closed, mic first. True on a cold open (the
        mic had no stream, so the pre-buffer is empty). Never raises: a mic that fails to open is left
        to the caller's stream_health() pre-flight."""
        self._last_activity = self._clock()
        if self._recorder.is_open:
            return False
        t0 = time.monotonic()
        try:
            self._recorder.open_stream()
        except Exception:
            log.warning("Opening the mic failed", exc_info=True)
        t_mic = time.monotonic()
        if self._cues():
            from voiceio import feedback   # imports sounddevice: lazily
            feedback.open_cue_stream()
        log.debug("Audio opened: mic %.0f ms, cue stream %.0f ms",
                  (t_mic - t0) * 1000, (time.monotonic() - t_mic) * 1000)
        return True

    def tick(self) -> None:
        """Apply the policy (watchdog, every few seconds, and the never-mode
        timer). Opens when it is wanted again (plugged into AC), closes when
        the window ran out. Skips a cycle rather than wait on a busy daemon."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            if not self._can_close():
                # Busy (recording, finalizing, speaking): the window restarts
                # once it is over.
                self._last_activity = self._clock()
                return
            # Parked = closed by this policy. An unparked stream that died is
            # the watchdog's to recover (with backoff), not reopened here.
            wanted = self.wanted()
            if wanted and self._recorder.is_parked:
                self.ensure_open()
            elif not wanted and not self._recorder.is_parked:
                self._close()
        finally:
            self._lock.release()

    def _close(self) -> None:
        t0 = time.monotonic()
        try:
            self._recorder.park()
        except Exception:
            log.warning("Closing the mic failed", exc_info=True)
        from voiceio import feedback   # imports sounddevice: lazily
        feedback.close_cue_stream()
        log.info("Audio closed while idle (keep_warm=%s, %.0f ms)",
                 self.mode, (time.monotonic() - t0) * 1000)

    def shutdown(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _schedule(self, delay: float) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(delay, self.tick)
        self._timer.daemon = True
        self._timer.start()

