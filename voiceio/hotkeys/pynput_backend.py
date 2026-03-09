"""Pynput hotkey backend — works on X11 and macOS."""
from __future__ import annotations

import logging
import time
from typing import Callable

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

DEBOUNCE_SECS = 0.8


class PynputHotkey:
    """Hotkey detection via pynput (X11 + macOS)."""

    name = "pynput"

    def probe(self) -> ProbeResult:
        try:
            import pynput  # noqa: F401
        except ImportError:
            return ProbeResult(ok=False, reason="pynput package not installed",
                               fix_hint="pip install pynput")

        import os
        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland":
            return ProbeResult(ok=False, reason="pynput does not work on Wayland",
                               fix_hint="Use evdev or socket backend instead.")

        return ProbeResult(ok=True)

    def start(self, combo: str, on_trigger: Callable[[], None]) -> None:
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

        def on_press(key):
            for mod_keys in MOD_MAP.values():
                if key in mod_keys:
                    pressed_mods.add(key)
            if key == target_key:
                for mod_name in required_mods:
                    if not (MOD_MAP[mod_name] & pressed_mods):
                        return
                now = time.monotonic()
                if now - last_trigger[0] >= DEBOUNCE_SECS:
                    last_trigger[0] = now
                    on_trigger()

        def on_release(key):
            for mod_keys in MOD_MAP.values():
                if key in mod_keys:
                    pressed_mods.discard(key)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        log.info("Pynput hotkey listener started")

    def stop(self) -> None:
        if hasattr(self, "_listener"):
            self._listener.stop()
