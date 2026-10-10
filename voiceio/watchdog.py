"""Health watchdog: periodic checks of the daemon's backends + resume recovery."""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from voiceio import tray

if TYPE_CHECKING:
    from voiceio import config
    from voiceio.core.decoders.base import Decoder
    from voiceio.ibus.session import IBusSession
    from voiceio.recorder import AudioRecorder
    from voiceio.streaming import StreamingSession
    from voiceio.typers.manager import TyperManager

log = logging.getLogger("voiceio")

_HEALTH_CHECK_INTERVAL = 10  # seconds between health checks

# If the gap between health checks exceeds this, we probably resumed
# from suspend/hibernate and should re-probe everything.
_RESUME_THRESHOLD = 30  # seconds


class Watchdog:
    """Every ``_HEALTH_CHECK_INTERVAL``: transcriber worker, audio stream,
    tray, typer and IBus engine; a full re-probe after suspend/resume.

    It never mutates the daemon's state machine itself: ``is_idle`` /
    ``is_recording`` read it, ``on_transcriber_failed`` asks the daemon to
    enter ERROR, and typer/platform changes go through ``typers``' IDLE-gated
    swap helpers.
    """

    def __init__(
        self,
        cfg: config.Config,
        *,
        transcriber: Decoder,
        recorder: AudioRecorder,
        typers: TyperManager,
        ibus: IBusSession,
        session: Callable[[], StreamingSession | None],
        is_idle: Callable[[], bool],
        is_recording: Callable[[], bool],
        on_transcriber_failed: Callable[[], None],
        on_hotkey: Callable[[], None],
        shutdown: threading.Event,
    ):
        self._cfg = cfg
        self._transcriber = transcriber
        self._recorder = recorder
        self._typers = typers
        self._ibus = ibus
        self._session = session
        self._is_idle = is_idle
        self._is_recording = is_recording
        self._on_transcriber_failed = on_transcriber_failed
        self._on_hotkey = on_hotkey
        self._shutdown = shutdown

        # Audio stream recovery backoff
        self._stream_fail_count = 0
        self._next_stream_retry: float = 0

    def start(self) -> None:
        threading.Thread(
            target=self._loop, daemon=True, name="health-watchdog",
        ).start()

    def _loop(self) -> None:
        """Periodic health check: transcriber worker, IBus engine, audio stream."""
        last_check = time.monotonic()
        while not self._shutdown.is_set():
            self._shutdown.wait(_HEALTH_CHECK_INTERVAL)
            if self._shutdown.is_set():
                break

            now = time.monotonic()
            gap = now - last_check
            last_check = now

            try:
                if gap > _RESUME_THRESHOLD:
                    log.info("System resume detected (%.0fs gap), re-probing all backends", gap)
                    self.on_resume()
                self.check()
            except Exception:
                log.debug("Health check error", exc_info=True)

    def on_resume(self) -> None:
        """Re-probe all backends after system suspend/hibernate.

        Sleep/hibernate breaks connections to system services (IBus, audio,
        tray D-Bus, ydotoold) across all platforms. Instead of catching each
        failure individually, do a single sweep to restore everything.
        """
        self._typers.refresh_platform()

        # Audio: reopen stream (device may have changed or died)
        try:
            self._recorder.reopen_stream()
            log.info("Resume: audio stream reopened")
        except Exception:
            log.warning("Resume: audio stream reopen failed", exc_info=True)

        # Typer: re-probe if on a fallback backend (e.g. clipboard instead
        # of ibus) — the preferred backend may be available now.
        self._typers.try_upgrade(reason="resume")

        if self._typers.typer.name == "ibus":
            self._ibus.on_resume()

        # Tray: restart if subprocess died
        if self._cfg.tray.enabled and not tray.is_alive():
            log.info("Resume: restarting tray")
            tray.restart(self._on_hotkey)

        # Transcriber: ensure worker is alive
        if not self._transcriber.healthy():
            try:
                self._transcriber.recover()
                log.info("Resume: transcriber worker restarted")
            except RuntimeError:
                log.error("Resume: transcriber worker failed")

    def check(self) -> None:
        """Run one health check cycle."""
        # Check transcriber worker
        if not self._transcriber.healthy():
            if self._is_recording():
                log.warning("Transcriber worker died during recording")
            try:
                self._transcriber.recover()
                log.info("Transcriber worker recovered")
            except RuntimeError:
                log.error("Transcriber worker permanently failed")
                self._on_transcriber_failed()

        # Check audio stream (covers ALSA underrun, PulseAudio/PipeWire
        # restart, device disconnect, stale callback heartbeat)
        ok, reason = self._recorder.stream_health()
        if not ok:
            now = time.monotonic()
            if now < self._next_stream_retry:
                return  # backoff: skip this cycle
            log.warning("Audio stream unhealthy: %s — reopening (attempt %d)",
                        reason, self._stream_fail_count + 1)
            try:
                self._recorder.reopen_stream()
                self._stream_fail_count = 0
                self._next_stream_retry = 0
                tray.set_error(False)
                log.info("Audio stream recovered")
            except Exception:
                self._stream_fail_count += 1
                # Backoff: 10s, 20s, 40s, 80s, max 5min
                delay = min(10 * (2 ** (self._stream_fail_count - 1)), 300)
                self._next_stream_retry = now + delay
                tray.set_error(True)
                log.error("Audio stream recovery failed (retry in %ds)", delay)
        elif self._stream_fail_count > 0:
            # Stream recovered externally (e.g. device plugged back in)
            self._stream_fail_count = 0
            self._next_stream_retry = 0
            tray.set_error(False)

        # Check tray subprocess (restart if died / lost D-Bus registration)
        if self._cfg.tray.enabled and not tray.is_alive():
            log.warning("Tray subprocess died, restarting")
            tray.restart(self._on_hotkey)

        # Typer upkeep: only touch typer/platform while IDLE, and only through
        # the lock-guarded swap helpers (never hot-swap under a live recording).
        if self._is_idle():
            self._typers.health_upkeep()

        if self._typers.typer.name == "ibus":
            idle = self._is_idle()
            self._ibus.health_tick(idle)
            # Mid-recording: don't re-force the source; if the user switched
            # away from voiceio, fall back to clipboard for this session.
            if not idle and self._ibus.hijacked():
                self._fall_back_to_clipboard()

    def _fall_back_to_clipboard(self) -> None:
        clip = self._typers.clipboard_fallback()
        if clip is None:
            return
        session = self._session()
        if session is not None:
            session.set_typer(clip)
