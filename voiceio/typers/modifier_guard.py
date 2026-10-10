"""Hold synthesized keystrokes until the user lets go of modifier keys.

Stopping dictation with a combo (ctrl+alt+v) leaves those modifiers down for a
moment, and a keystroke typer that types then turns the text into shortcuts.
EVIOCGKEY (python-evdev ``active_keys()``) snapshots each keyboard's pressed
keys without consuming its events, below the display server, so it works on
any compositor. Without readable keyboards in /dev/input this is a no-op.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

TIMEOUT_SECS = 0.6
_POLL_SECS = 0.015


# Closing an evdev fd waits out an RCU grace period (~15-50 ms per device),
# far longer than the check itself: keyboards stay open across calls, and
# other devices are closed off the typing path.
_keyboards: dict[str, object] = {}
_not_keyboards: set[str] = set()
_lock = threading.Lock()


def _open_keyboards() -> list:
    """Every readable keyboard device; [] without evdev or access.

    Re-listed on each call so hotplugged keyboards count; only new paths are
    opened, and devices that vanished are dropped.
    """
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        return []
    with _lock:
        paths = set(evdev.list_devices(writable=False))
        for path in set(_keyboards) - paths:
            try:
                _keyboards.pop(path).close()
            except OSError:
                pass
        _not_keyboards.intersection_update(paths)
        for path in sorted(paths - set(_keyboards) - _not_keyboards):
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            try:
                keys = dev.capabilities().get(ecodes.EV_KEY, ())
            except OSError:
                keys = ()
            if ecodes.KEY_A in keys and ecodes.KEY_Z in keys:
                _keyboards[path] = dev
            else:
                _not_keyboards.add(path)
                threading.Thread(target=dev.close, daemon=True).start()
        return list(_keyboards.values())


def _modifiers() -> frozenset[int]:
    from evdev import ecodes
    return frozenset({
        ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL,
        ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT,
        ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT,
        ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA,
    })


def _held(keyboards: list, modifiers: frozenset[int]) -> bool:
    for dev in keyboards:
        try:
            if modifiers.intersection(dev.active_keys()):
                return True
        except OSError:   # unplugged mid-wait
            continue
    return False


def wait_for_release(timeout: float = TIMEOUT_SECS) -> None:
    """Return once no modifier is held on any keyboard, or after ``timeout``
    (then typing goes ahead: a stuck key must not swallow the text)."""
    keyboards = _open_keyboards()
    if not keyboards:
        return
    modifiers = _modifiers()
    deadline = time.monotonic() + timeout
    while _held(keyboards, modifiers):
        if time.monotonic() >= deadline:
            log.debug("Modifier still held after %.1fs, typing anyway", timeout)
            return
        time.sleep(_POLL_SECS)
