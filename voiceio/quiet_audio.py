"""Quiet the desktop while recording (opt-in, ``[audio]``).

``pause_media``: MPRIS players that are Playing when recording starts are
paused over the session bus (busctl, else playerctl), and exactly those resume
when it stops: a player the user had paused stays paused.

``duck_volume``: the default output sink is lowered to that fraction of its
level (wpctl, else pactl) and put back when recording stops, for sound no
player owns (games, calls, browser tabs). The sink is resolved to a concrete
device at duck time, so switching outputs mid-recording restores the right
one. A volume the user changed while recording is left alone, and mute is
never touched. Volumes go through the tools' own 2-decimal / whole-percent
precision, and every channel is set to the same level.

Each part is a no-op without its tools.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

log = logging.getLogger(__name__)

_TIMEOUT = 2
_MPRIS = "org.mpris.MediaPlayer2."
_PLAYER = ("/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player")
# Smaller differences are the tools' rounding, not the user turning the knob.
_VOLUME_SLACK = 0.015

Runner = Callable[..., subprocess.CompletedProcess]


def _which(name: str) -> str | None:
    return shutil.which(name)


_run: Runner = subprocess.run


class _Caller:
    def __init__(self, run: Runner | None):
        self._run = run

    def __call__(self, *argv: str) -> str | None:
        """stdout of a successful call; None when it failed or hung."""
        try:
            done = (self._run or _run)(list(argv), capture_output=True, text=True,
                                       timeout=_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("%s failed: %s", " ".join(argv[:3]), e)
            return None
        if done.returncode != 0:
            log.debug("%s exited %d: %s", " ".join(argv[:3]), done.returncode,
                      (done.stderr or "").strip())
            return None
        return done.stdout


# ── Players ─────────────────────────────────────────────────────────────

class _Busctl:
    tool = "busctl"

    def __init__(self, call: _Caller):
        self._call = call

    def playing(self) -> list[str]:
        out = self._call("busctl", "--user", "call", "org.freedesktop.DBus",
                         "/org/freedesktop/DBus", "org.freedesktop.DBus", "ListNames")
        names = [n for n in re.findall(r'"([^"]*)"', out or "") if n.startswith(_MPRIS)]
        return [n for n in names if self._status(n) == "Playing"]

    def _status(self, name: str) -> str | None:
        out = self._call("busctl", "--user", "get-property", name, *_PLAYER, "PlaybackStatus")
        m = re.match(r's "(\w+)"', out or "")   # s "Playing"
        return m[1] if m else None

    def pause(self, name: str) -> bool:
        return self._call("busctl", "--user", "call", name, *_PLAYER, "Pause") is not None

    def play(self, name: str) -> bool:
        return self._call("busctl", "--user", "call", name, *_PLAYER, "Play") is not None


class _Playerctl:
    tool = "playerctl"

    def __init__(self, call: _Caller):
        self._call = call

    def playing(self) -> list[str]:
        out = self._call("playerctl", "-a", "status", "--format",
                         "{{playerInstance}}\t{{status}}")
        rows = (line.partition("\t") for line in (out or "").splitlines())
        return [name for name, _, state in rows if state.strip() == "Playing"]

    def pause(self, name: str) -> bool:
        return self._call("playerctl", "-p", name, "pause") is not None

    def play(self, name: str) -> bool:
        return self._call("playerctl", "-p", name, "play") is not None


# ── Output volume ───────────────────────────────────────────────────────

class _Wpctl:
    tool = "wpctl"

    def __init__(self, call: _Caller):
        self._call = call

    def sink(self) -> str | None:
        m = re.match(r"id (\d+),", self._call("wpctl", "inspect", "@DEFAULT_AUDIO_SINK@") or "")
        return m[1] if m else None

    def get(self, sink: str) -> tuple[float, bool] | None:
        """(volume, muted); volume may exceed 1.0 when boosted."""
        out = self._call("wpctl", "get-volume", sink) or ""
        m = re.match(r"Volume: ([\d.]+)", out)
        return (float(m[1]), "[MUTED]" in out) if m else None

    def set(self, sink: str, volume: float) -> bool:
        return self._call("wpctl", "set-volume", sink, f"{volume:.2f}") is not None


class _Pactl:
    tool = "pactl"

    def __init__(self, call: _Caller):
        self._call = call

    def sink(self) -> str | None:
        return (self._call("pactl", "get-default-sink") or "").strip() or None

    def get(self, sink: str) -> tuple[float, bool] | None:
        # Volume: front-left: 26214 /  40% / -23.88 dB,   front-right: ...
        m = re.search(r"(\d+)%", self._call("pactl", "get-sink-volume", sink) or "")
        if not m:
            return None
        mute = self._call("pactl", "get-sink-mute", sink) or ""
        return int(m[1]) / 100, "yes" in mute

    def set(self, sink: str, volume: float) -> bool:
        return self._call("pactl", "set-sink-volume", sink, f"{round(volume * 100)}%") is not None


_PLAYER_BACKENDS = (_Busctl, _Playerctl)
_VOLUME_BACKENDS = (_Wpctl, _Pactl)


def _available(backends: tuple[type, ...]) -> type | None:
    return next((b for b in backends if _which(b.tool)), None)


def describe(pause_media: bool, duck_volume: float) -> str | None:
    """Doctor line, or None when neither part is enabled."""
    parts = []
    if pause_media:
        b = _available(_PLAYER_BACKENDS)
        parts.append(f"pause players ({b.tool})" if b
                     else "pause players: busctl/playerctl not found")
    if _ducks(duck_volume):
        b = _available(_VOLUME_BACKENDS)
        level = f"duck output to {round(duck_volume * 100)}%"
        parts.append(f"{level} ({b.tool})" if b else f"{level}: wpctl/pactl not found")
    return ", ".join(parts) or None


def _ducks(duck_volume: float) -> bool:
    return 0 <= duck_volume < 1


class _Lane:
    """One worker thread, so a quick stop never undoes a start that has not
    happened yet. The backend is picked on first use; without one the lane is
    a no-op."""

    def __init__(self, name: str, backends: tuple[type, ...], call: _Caller, enabled: bool):
        self._name = name
        self._backends = backends
        self._call = call
        self._executor: ThreadPoolExecutor | None = None
        self.backend = None
        self.enabled = enabled

    def submit(self, fn: Callable) -> bool:
        if not self.enabled:
            return False
        if self.backend is None:
            found = _available(self._backends)
            if found is None:
                log.debug("%s: none of %s installed", self._name,
                          "/".join(b.tool for b in self._backends))
                self.enabled = False
                return False
            self.backend = found(self._call)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._name)
        self._executor.submit(fn)
        return True

    def close(self, last: Callable) -> None:
        """Run ``last``, then wait for the worker (bounded by the timeouts)."""
        self.enabled = False
        if self._executor is not None:
            self._executor.submit(last)
            self._executor.shutdown(wait=True)
            self._executor = None


class QuietAudio:
    """``start()`` and ``stop()`` return at once; the tool calls run on one
    worker per part (players, volume), in order."""

    def __init__(self, pause_media: bool, duck_volume: float = 1.0, run: Runner | None = None):
        call = _Caller(run)
        self._players = _Lane("pause-media", _PLAYER_BACKENDS, call, pause_media)
        self._volume = _Lane("duck-volume", _VOLUME_BACKENDS, call, _ducks(duck_volume))
        self._factor = duck_volume
        self._paused: list[str] = []          # worker-only, as is _ducked
        self._ducked: tuple[str, float, float] | None = None   # sink, original, ours
        self._recording = 0

    def start(self, duck_after: float = 0.0) -> None:
        """Pause what is playing; duck the output ``duck_after`` seconds from
        now (once voiceio's own start chime has played out)."""
        self._players.submit(self._pause)
        self._recording += 1
        at, rid = time.monotonic() + duck_after, self._recording
        self._volume.submit(lambda: self._duck(at, rid))

    def stop(self, then: Callable[[], None] | None = None) -> None:
        """Restore the volume, then call ``then`` (on the volume worker, or
        right away when nothing ducks), then resume the paused players."""
        self._recording += 1

        def restore() -> None:
            try:
                self._restore()
            finally:
                if then is not None:
                    then()

        if not self._volume.submit(restore) and then is not None:
            then()
        self._players.submit(self._resume)

    def close(self) -> None:
        """Undo whatever is still in effect and stop the workers."""
        self._recording += 1
        self._volume.close(self._restore)
        self._players.close(self._resume)

    def _pause(self) -> None:
        backend = self._players.backend
        for name in backend.playing():
            if name not in self._paused and backend.pause(name):
                self._paused.append(name)
        if self._paused:
            log.debug("Paused %d media player(s) for recording", len(self._paused))

    def _resume(self) -> None:
        paused, self._paused = self._paused, []
        for name in paused:
            if not self._players.backend.play(name):
                log.debug("Could not resume %s (closed during recording?)", name)

    def _duck(self, at: float, rid: int) -> None:
        time.sleep(max(0.0, at - time.monotonic()))
        if rid != self._recording or self._ducked is not None:
            return   # already stopped, or still ducked from before
        backend = self._volume.backend
        sink = backend.sink()
        level = backend.get(sink) if sink else None
        if level is None or level[1]:
            return   # unreadable, or muted: nothing to lower
        original = level[0]
        ours = round(original * self._factor, 2)
        if ours < original and backend.set(sink, ours):
            self._ducked = (sink, original, ours)
            log.debug("Ducked output %s: %.2f -> %.2f", sink, original, ours)

    def _restore(self) -> None:
        if self._ducked is None:
            return
        (sink, original, ours), self._ducked = self._ducked, None
        backend = self._volume.backend
        level = backend.get(sink)
        if level is None:
            log.debug("Output %s gone before restore", sink)
        elif abs(level[0] - ours) > _VOLUME_SLACK:
            log.debug("Output %s changed during recording (%.2f); leaving it", sink, level[0])
        else:
            backend.set(sink, original)
