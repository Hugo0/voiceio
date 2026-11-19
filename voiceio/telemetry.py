"""The opt-in weekly install ping: one anonymous count to voiceio.dev.

Off unless `[telemetry] ping = true`. Asked once, in an interactive
`voiceio setup` or `voiceio doctor`. When on, the daemon POSTs at most once
every 7 days exactly the fields of `payload()`: the voiceio version, distro,
desktop, session type and decoder family. No install ID, no hostname, no
model path, nothing else; the server keeps a count per week and those fields.
"""
from __future__ import annotations

import json
import logging
import platform as _py_platform
import re
import threading
import time

from voiceio import config

log = logging.getLogger(__name__)

PING_URL = "https://voiceio.dev/ping"
INTERVAL_SECS = 7 * 86400
CHECK_EVERY_SECS = 86400
# After daemon start: out of the way of the model load and the first dictation.
START_DELAY_SECS = 60.0
TIMEOUT_SECS = 5.0
FIELDS = ("v", "distro", "desktop", "session", "decoder")
QUESTION = ("Send an anonymous weekly count (voiceio version, distro, desktop, session type, "
            "speech engine) to voiceio.dev, so we know how many people use voiceio? "
            "Nothing else is sent. [y/N] ")
MAX_FIELD = 48
_UNSAFE = re.compile(r"[^a-z0-9._+ -]+")


def _state_path():
    return config.LOG_DIR / "ping.json"


def _clean(value: str) -> str:
    """Lowercase, only [a-z0-9._+ -], at most MAX_FIELD characters: what the
    server accepts, and nothing that could carry more than a label."""
    text = " ".join(_UNSAFE.sub("-", str(value).lower()).split()).strip(" -")
    return text[:MAX_FIELD].strip() or "unknown"


def _distro() -> str:
    """'<ID> <VERSION_ID>' from /etc/os-release, e.g. 'ubuntu 26.04'."""
    try:
        rel = _py_platform.freedesktop_os_release()
    except OSError:
        return "unknown"
    return f"{rel.get('ID', '')} {rel.get('VERSION_ID', '')}".strip()


def payload(model_name: str) -> dict[str, str]:
    """Exactly what is sent. The decoder is its family, never the model name."""
    from voiceio import __version__
    from voiceio import platform as plat
    from voiceio.core.decoders.catalog import family
    p = plat.detect()
    return {
        "v": _clean(__version__),
        "distro": _clean(_distro()),
        "desktop": _clean(p.desktop),
        "session": _clean(p.display_server),
        "decoder": _clean(family(model_name)),
    }


def setting() -> bool | None:
    """True, False, or None when never asked."""
    return config.load().telemetry.ping


def status_line(model_name: str | None = None) -> str:
    """Doctor's line: whether the count is on, and the exact payload if it is."""
    try:
        value = setting()
        state = "on" if value is True else "off" if value is False else "not asked"
        line = f"Weekly anonymous count: {state}"
        if value is True:
            if model_name is None:
                model_name = config.load().model.name
            line += f"\n  sends: {json.dumps(payload(model_name))}"
        return line
    except Exception as e:  # noqa: BLE001 — doctor must not fail over this line
        return f"Weekly anonymous count: could not read the setting ({type(e).__name__})"


def _last_sent() -> float:
    try:
        sent = json.loads(_state_path().read_text()).get("sent", 0)
        return float(sent) if isinstance(sent, (int, float)) else 0.0
    except (OSError, ValueError, AttributeError):
        return 0.0


def due(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    last = _last_sent()
    # A clock set back makes `last` lie in the future: send, don't wait years.
    return not (0 < last <= now and now - last < INTERVAL_SECS)


def maybe_send(model_name: str, *, now: float | None = None) -> bool:
    """Send the count if it is on and 7 days have passed. True when sent.
    Never raises; logs at debug level only."""
    try:
        if setting() is not True or not due(now):
            return False
        from voiceio import report
        body = json.dumps(payload(model_name)).encode("utf-8")
        with report._urlopen(PING_URL, timeout=TIMEOUT_SECS, json_body=body) as resp:
            resp.read()
        config.secure_write(_state_path(),
                            json.dumps({"sent": time.time() if now is None else now}))
        log.debug("Weekly anonymous count sent")
        return True
    except Exception as e:  # noqa: BLE001 — a count must never touch dictation
        log.debug("Weekly anonymous count not sent: %s", type(e).__name__)
        return False


def start(model_name: str, shutdown: threading.Event) -> threading.Thread:
    """The daemon's background check: shortly after start, then once a day."""
    def loop() -> None:
        if shutdown.wait(START_DELAY_SECS):
            return
        while True:
            maybe_send(model_name)
            if shutdown.wait(CHECK_EVERY_SECS):
                return

    thread = threading.Thread(target=loop, daemon=True, name="weekly-count")
    thread.start()
    return thread


def offer() -> None:
    """Ask once, in a terminal, and store the answer. Never asks again once
    answered (`voiceio config set telemetry.ping true|false` changes it)."""
    from voiceio.report import _ask, _interactive
    try:
        if not _interactive() or setting() is not None:
            return
        yes = _ask("\n  " + QUESTION).strip().lower() in ("y", "yes")
        config.set_value("telemetry", "ping", yes)
        print("  Weekly count " + ("on" if yes else "off")
              + ". Change it any time: voiceio config set telemetry.ping "
              + ("false" if yes else "true"))
    except Exception as e:  # noqa: BLE001 — never fail setup or doctor over this
        log.debug("Weekly count question skipped: %s", type(e).__name__)
