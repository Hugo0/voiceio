"""Pynput hotkey backend for X11 and macOS."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

DEBOUNCE_SECS = 0.8
_AUTOREPEAT_SECS = 0.05


def _check_macos_accessibility() -> bool:
    """Check if the current process has macOS Accessibility permission."""
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke ""'],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Can't verify — assume it's fine and let pynput fail later
        return True


class PynputHotkey:
    """Hotkey detection via pynput (X11 + macOS)."""

    name = "pynput"

    def probe(self) -> ProbeResult:
        try:
            import pynput  # noqa: F401
        except ImportError:
            from voiceio.platform import desktop_dep_hint
            return ProbeResult(ok=False, reason="pynput package not installed",
                               fix_hint=desktop_dep_hint("pynput"))

        import os
        import sys
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland":
            return ProbeResult(ok=False, reason="pynput does not work on Wayland",
                               fix_hint="Use evdev or socket backend instead.")

        if sys.platform == "darwin":
            if not _check_macos_accessibility():
                return ProbeResult(
                    ok=False, reason="macOS Accessibility permission not granted",
                    fix_hint="System Settings → Privacy & Security → Accessibility → enable your terminal/Python",
                )

        return ProbeResult(ok=True)

    def start(self, combo: str, on_trigger: Callable[[], None],
              on_release: Callable[[], None] | None = None) -> None:
        from pynput import keyboard

        MOD_MAP = {
            "super": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
            "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
            "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r},
            "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
        }

        parts = [p.strip().lower() for p in combo.split("+")]
        required_mods = [parts[i] for i in range(len(parts) - 1)]
        key_name = parts[-1]
        if len(key_name) == 1:
            target_key = keyboard.KeyCode.from_char(key_name)
        else:
            target_key = getattr(keyboard.Key, key_name)

        pressed_mods: set = set()
        last_trigger = [0.0]
        # X11 autorepeat may report a held key as release+press pairs: a
        # release counts only once no press follows within _AUTOREPEAT_SECS.
        down = [False]
        fired = [False]
        pending: list[threading.Timer | None] = [None]
        lock = threading.RLock()

        def released() -> None:
            with lock:
                if pending[0] is None:   # cancelled by a press meanwhile
                    return
                pending[0] = None
                down[0] = False
                if fired[0]:
                    fired[0] = False
                    if on_release is not None:
                        on_release()

        def on_press(key):
            with lock:
                _on_press(key)

        def _on_press(key):
            for mod_keys in MOD_MAP.values():
                if key in mod_keys:
                    pressed_mods.add(key)
            if key != target_key:
                return
            if pending[0] is not None:
                pending[0].cancel()
                pending[0] = None
                return
            if down[0]:
                return
            down[0] = True
            for mod_name in required_mods:
                if not (MOD_MAP[mod_name] & pressed_mods):
                    return
            now = time.monotonic()
            if now - last_trigger[0] >= DEBOUNCE_SECS:
                last_trigger[0] = now
                fired[0] = True
                on_trigger()

        def on_release_key(key):
            with lock:
                _on_release_key(key)

        def _on_release_key(key):
            for mod_keys in MOD_MAP.values():
                if key in mod_keys:
                    pressed_mods.discard(key)
            if key == target_key and down[0] and pending[0] is None:
                pending[0] = threading.Timer(_AUTOREPEAT_SECS, released)
                pending[0].daemon = True
                pending[0].start()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release_key)
        self._listener.start()
        log.info("Pynput hotkey listener started")

    def stop(self) -> None:
        if hasattr(self, "_listener"):
            self._listener.stop()
