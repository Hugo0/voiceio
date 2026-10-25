"""Is someone using this computer right now? (for background work)

Background learning must never compete with the person at the keyboard, and
CPU priority alone does not stop that: an idle-weight cgroup still shares
the laptop's power budget, caches, memory bandwidth and SMT siblings, so
five busy cores slow the desktop even when they always yield. The only
reliable answer is to do heavy work while nobody is there.

Evidence, best first, all read-only over D-Bus (busctl, which ships with
systemd):

  1. Input idle time from the compositor: GNOME's
     org.gnome.Mutter.IdleMonitor (milliseconds since the last keyboard,
     mouse or touch event), else org.freedesktop.ScreenSaver's
     GetSessionIdleTime (KDE). The fast check, polled every few seconds.
  2. A locked or blanked screen (org.gnome.ScreenSaver, else logind's
     LockedHint): nobody is looking, whatever the idle time says.
  3. An idle inhibitor (org.gnome.SessionManager): a video or a call keeps
     the session "in use" with no input at all.
  4. logind's IdleHint for the user's graphical session, for desktops
     without an idle monitor (it is computed from tty activity on a
     console). No local session at all means nobody is at the screen.

When none of this can be read the answer is "in use": a machine that can't
tell is treated as busy, never as free.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TIMEOUT = 3
# No input for this long is away even when an app keeps the session awake:
# past a film's length, the inhibitor is a long-running job, not a viewer.
INHIBITED_MAX = 2 * 3600


@dataclass(frozen=True)
class Presence:
    here: bool
    why: str    # for the log: "away 42 min", "screen locked", …


@dataclass(frozen=True)
class SeatSession:
    """logind's view of the user's graphical (or console) session."""
    exists: bool
    idle: bool = False
    idle_secs: float = 0.0
    locked: bool = False


def _busctl(*args: str) -> str | None:
    """stdout of a successful busctl call; None when it failed or hung."""
    if shutil.which("busctl") is None:
        return None
    try:
        done = subprocess.run(["busctl", f"--timeout={_TIMEOUT}", *args],
                              capture_output=True, text=True, timeout=_TIMEOUT + 1)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("busctl %s failed: %s", " ".join(args[:3]), e)
        return None
    return done.stdout if done.returncode == 0 else None


def _values(out: str | None) -> list[str]:
    """busctl's 't 2527' / 'b true' lines → ['2527'] / ['true'], one per line."""
    return [m[1] for m in re.finditer(r"^[a-z]+ (.*)$", out or "", re.MULTILINE)]


class SessionProbe:
    """The real desktop, read over the session and system buses."""

    def __init__(self, call: Callable[..., str | None] = _busctl):
        self._call = call

    def idle_seconds(self) -> float | None:
        out = self._call("--user", "call", "org.gnome.Mutter.IdleMonitor",
                         "/org/gnome/Mutter/IdleMonitor/Core",
                         "org.gnome.Mutter.IdleMonitor", "GetIdletime")
        if v := _values(out):
            return int(v[0]) / 1000
        out = self._call("--user", "call", "org.freedesktop.ScreenSaver",
                         "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver",
                         "GetSessionIdleTime")
        if v := _values(out):
            return float(v[0])
        return None

    def screen_locked(self) -> bool:
        out = self._call("--user", "call", "org.gnome.ScreenSaver", "/org/gnome/ScreenSaver",
                         "org.gnome.ScreenSaver", "GetActive")
        if v := _values(out):
            return v[0] == "true"
        s = self.seat_session()
        return bool(s and s.locked)

    def idle_inhibited(self) -> bool:
        # GsmInhibitorFlag 8 = idle: set while a video plays or a call runs.
        out = self._call("--user", "call", "org.gnome.SessionManager",
                         "/org/gnome/SessionManager", "org.gnome.SessionManager",
                         "IsInhibited", "u", "8")
        v = _values(out)
        return bool(v) and v[0] == "true"

    def seat_session(self) -> SeatSession | None:
        out = self._call("get-property", "org.freedesktop.login1",
                         "/org/freedesktop/login1/user/self", "org.freedesktop.login1.User",
                         "Display")
        if out is None:
            return None
        m = re.search(r'"([^"]*)" "([^"]*)"', out)    # (so) "3" "/org/…/session/_33"
        if m is None:
            return None
        if not m[1]:
            return SeatSession(exists=False)
        props = _values(self._call("get-property", "org.freedesktop.login1", m[2],
                                   "org.freedesktop.login1.Session", "IdleHint",
                                   "IdleSinceHintMonotonic", "LockedHint"))
        if len(props) != 3:
            return None
        since_us = int(props[1])
        idle = props[0] == "true"
        idle_secs = max(0.0, time.monotonic() - since_us / 1e6) if idle and since_us else 0.0
        return SeatSession(exists=True, idle=idle, idle_secs=idle_secs,
                           locked=props[2] == "true")


def _minutes(secs: float) -> str:
    return f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.0f} min"


def check(away_after: float, probe: SessionProbe | None = None) -> Presence:
    """Is the user here? Away means no input for `away_after` seconds (and
    nothing keeping the session awake), or a locked screen."""
    probe = probe or SessionProbe()
    idle = probe.idle_seconds()
    if idle is not None:
        if idle < away_after:
            if probe.screen_locked():
                return Presence(False, "screen locked")
            return Presence(True, f"in use (last input {_minutes(idle)} ago)")
        if idle < INHIBITED_MAX and probe.idle_inhibited() and not probe.screen_locked():
            return Presence(True, "an app is keeping the session awake (a video or a call?)")
        return Presence(False, f"away {_minutes(idle)}")
    session = probe.seat_session()
    if session is None:
        return Presence(True, "can't tell whether anyone is at the computer "
                              "(no idle monitor or logind answered)")
    if not session.exists:
        return Presence(False, "nobody is logged in at the screen")
    if session.locked or probe.screen_locked():
        return Presence(False, "screen locked")
    if session.idle and session.idle_secs >= away_after:
        return Presence(False, f"away {_minutes(session.idle_secs)} (logind)")
    return Presence(True, "in use (logind)")
