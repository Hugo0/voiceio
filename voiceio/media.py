"""Pause media players while recording (opt-in: ``[audio] pause_media``).

Players that are Playing when recording starts are paused through playerctl
(MPRIS), and exactly those are resumed when it stops: a player the user had
paused stays paused. Without playerctl this is a no-op.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

log = logging.getLogger(__name__)

_TIMEOUT = 2


def _playerctl() -> str | None:
    return shutil.which("playerctl")


class MediaPauser:
    """``pause()`` and ``resume()`` return at once; the playerctl calls run in
    order on one worker thread, so a quick stop cannot resume before the
    pause it undoes has happened."""

    def __init__(self, enabled: bool, run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self._enabled = enabled
        self._run = run
        self._executor: ThreadPoolExecutor | None = None
        self._paused: list[str] = []   # touched only on the worker thread

    def _submit(self, fn: Callable[[], None]) -> None:
        if not self._enabled:
            return
        if _playerctl() is None:
            log.debug("pause_media is on but playerctl is not installed")
            self._enabled = False
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="media")
        self._executor.submit(fn)

    def pause(self) -> None:
        """Pause every player that is playing right now."""
        self._submit(self._pause)

    def resume(self) -> None:
        """Resume the players the last ``pause()`` paused."""
        self._submit(self._resume)

    def close(self) -> None:
        """Resume what is still paused, then wait for the worker (bounded by
        the playerctl timeouts)."""
        if self._executor is not None:
            self._executor.submit(self._resume)
            self._executor.shutdown(wait=True)
            self._executor = None

    def _playerctl_call(self, *args: str) -> subprocess.CompletedProcess | None:
        try:
            return self._run(["playerctl", *args], capture_output=True, text=True,
                             timeout=_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("playerctl %s failed: %s", " ".join(args), e)
            return None

    def _pause(self) -> None:
        status = self._playerctl_call("-a", "status", "--format", "{{playerInstance}}\t{{status}}")
        if status is None or status.returncode != 0:
            return   # no players
        for line in status.stdout.splitlines():
            name, _, state = line.partition("\t")
            if state.strip() != "Playing" or name in self._paused:
                continue
            done = self._playerctl_call("-p", name, "pause")
            if done is not None and done.returncode == 0:
                self._paused.append(name)
        if self._paused:
            log.debug("Paused %d media player(s) for recording", len(self._paused))

    def _resume(self) -> None:
        paused, self._paused = self._paused, []
        for name in paused:
            done = self._playerctl_call("-p", name, "play")
            if done is None or done.returncode != 0:
                log.debug("Could not resume %s (closed during recording?)", name)
