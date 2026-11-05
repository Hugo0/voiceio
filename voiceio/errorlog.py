"""voiceio's most recent error, in a form safe to post, and a nudge to report
errors the service hits on its own.

Logs and crash tracebacks can quote dictated text (an exception message, an
argv), so nothing here ever copies a message. A report gets the exception
type, the innermost voiceio `file:line` and how long ago; a notification
says only that something failed.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
from pathlib import Path

MAX_AGE = dt.timedelta(days=7)
# The same error this many times within the window is worth one notification.
REPEATS = 3
REPEAT_WINDOW_SECS = 600

_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\S+) (\w+) ")
_CRASH_HEADER = re.compile(r"^===== (\S+) =====$")
_FRAME = re.compile(r'^\s+File "([^"]+)", line (\d+)')
# The final line of a traceback: a dotted exception name, alone or before ":".
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]{0,120})(?::|$)")
_NAME = re.compile(r"^[A-Za-z_][\w.]{0,120}$")
_PKG = Path(__file__).resolve().parent


def last_error(log_path: Path | None = None, crash_path: Path | None = None,
               now: dt.datetime | None = None) -> str:
    """'DecodeError at voiceio/streaming.py:488 (2 h ago)' for the newest
    error of the last week in the service log or crash.log; '' if none."""
    from voiceio.config import LOG_DIR, LOG_PATH
    now = now or dt.datetime.now()
    found = [e for path, parse in ((log_path or LOG_PATH, _log_errors),
                                   (crash_path or LOG_DIR / "crash.log", _crash_errors))
             for e in parse(_read(path))]
    recent = [e for e in found if now - e[0] <= MAX_AGE]
    if not recent:
        return ""
    when, exc, where = max(recent, key=lambda e: e[0])
    what = f"{exc} at {where}" if exc and where else exc or f"logged by {where}"
    return f"{what} ({_age(now - when)})"


def _read(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _log_errors(lines: list[str]):
    """(time, exception type, location) per ERROR/CRITICAL in the service log;
    without a traceback the location is the logger's name."""
    for i, line in enumerate(lines):
        m = _LOG_LINE.match(line)
        if not m or m.group(3) not in ("ERROR", "CRITICAL"):
            continue
        tail = []
        for nxt in lines[i + 1:]:
            if _LOG_LINE.match(nxt):
                break
            tail.append(nxt)
        exc, where = _traceback(tail)
        logger = m.group(2) if _NAME.match(m.group(2)) else ""
        yield _time(m.group(1)), exc, where or logger


def _crash_errors(lines: list[str]):
    starts = [(i, m.group(1)) for i, line in enumerate(lines)
              if (m := _CRASH_HEADER.match(line))]
    for n, (i, stamp) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        when = _time(stamp)
        if when is not None:
            yield (when, *_traceback(lines[i + 1:end]))


def _traceback(lines: list[str]) -> tuple[str, str]:
    """The last exception's type and its innermost voiceio frame."""
    exc, where, after_frames = "", "", False
    for line in lines:
        if f := _FRAME.match(line):
            after_frames = True
            if rel := _voiceio_path(f.group(1)):
                where = f"{rel}:{f.group(2)}"
        elif line.startswith("Traceback ("):
            exc, where = "", ""   # a chained traceback: the last one wins
        elif after_frames and not line.startswith((" ", "\t")):
            # Only the line right after the frames names the exception; the
            # message's own later lines could be anything, dictation included.
            after_frames = False
            if m := _EXC_LINE.match(line):
                exc = m.group(1).rsplit(".", 1)[-1]
    return exc, where


def _voiceio_path(path: str) -> str:
    """'voiceio/app.py' for a file of this package; '' for anything else."""
    try:
        return "voiceio/" + Path(path).resolve().relative_to(_PKG).as_posix()
    except (OSError, ValueError):
        pass
    if "site-packages/" in path:   # another install of voiceio, or a library
        rest = path.rsplit("site-packages/", 1)[1]
        return rest if rest.startswith("voiceio/") else ""
    return "voiceio/" + path.rsplit("/voiceio/", 1)[1] if "/voiceio/" in path else ""


def _time(stamp: str) -> dt.datetime | None:
    try:
        when = dt.datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return when.replace(tzinfo=None)


def _age(delta: dt.timedelta) -> str:
    secs = max(0, int(delta.total_seconds()))
    if secs < 3600:
        return f"{max(1, secs // 60)} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return f"{secs // 86400} d ago"


class RepeatedErrorNotifier(logging.Handler):
    """The service has no terminal to tell anyone, so when it logs the same
    error REPEATS times in REPEAT_WINDOW_SECS, one desktop notification a day
    suggests a report. Never posts anything itself."""

    def __init__(self, state_path: Path | None = None, notify=None):
        super().__init__(level=logging.ERROR)
        from voiceio.config import LOG_DIR
        self._path = state_path or LOG_DIR / "error-notify.json"
        self._notify = notify
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        # The format string, not the message: same error, different details.
        key = f"{record.name}:{record.msg}"[:200]
        now = time.time()
        today = dt.date.today().isoformat()
        try:
            with self._lock:
                state = self._load()
                seen = {k: [t for t in ts if now - t < REPEAT_WINDOW_SECS]
                        for k, ts in state["recent"].items() if isinstance(ts, list)}
                seen[key] = [*seen.get(key, []), now][-REPEATS:]
                state["recent"] = {k: ts for k, ts in seen.items() if ts}
                fire = len(seen[key]) >= REPEATS and state.get("notified") != today
                if fire:
                    state["notified"] = today
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(state))
            if fire:
                notify = self._notify
                if notify is None:
                    from voiceio.feedback import notify
                notify("voiceio hit an error",
                       "Run 'voiceio doctor --report' to tell us.")
        except Exception:  # noqa: BLE001 — logging must never fail the daemon
            pass

    def _load(self) -> dict:
        try:
            state = json.loads(self._path.read_text())
            if isinstance(state, dict) and isinstance(state.get("recent"), dict):
                return state
        except (OSError, ValueError):
            pass
        return {"recent": {}}
