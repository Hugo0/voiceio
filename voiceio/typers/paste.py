"""The paste keystroke: which chord, and pressing it with ydotool, wtype or xdotool.

Shared by the clipboard typer and the IBus typer's no-focus fallback.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time

log = logging.getLogger(__name__)

# name: (evdev code for ydotool, wtype name, xdotool name)
_MODIFIERS = {
    "ctrl": (29, "ctrl", "ctrl"),
    "shift": (42, "shift", "shift"),
    "alt": (56, "alt", "alt"),
    "super": (125, "logo", "super"),
}
_KEYS = {
    "v": (47, "v", "v"),
    "insert": (110, "Insert", "Insert"),
}
DEFAULT = (["ctrl"], "v")
# One focus query per dictation, not per streamed word.
_FOCUS_TTL = 1.0
_focus_lock = threading.Lock()
_focus_cache: tuple[float, bool] = (float("-inf"), False)


def parse_paste_keys(spec: str) -> tuple[list[str], str]:
    """``"ctrl+shift+v"`` → (["ctrl", "shift"], "v"); ctrl+v if not understood."""
    *mods, key = [part.strip().lower() for part in spec.split("+")]
    if key in _KEYS and len(set(mods)) == len(mods) \
            and all(m in _MODIFIERS for m in mods):
        return mods, key
    log.warning("output.paste_keys %r not understood (modifiers: %s; keys: %s), "
                "using ctrl+v", spec, ", ".join(_MODIFIERS), ", ".join(_KEYS))
    return list(DEFAULT[0]), DEFAULT[1]


def paste_command(tool: str, mods: list[str], key: str) -> list[str]:
    """argv that presses mods+key with ydotool, wtype or xdotool."""
    if tool == "ydotool":
        codes = [_MODIFIERS[m][0] for m in mods]
        k = _KEYS[key][0]
        return ["ydotool", "key", *(f"{c}:1" for c in codes), f"{k}:1", f"{k}:0",
                *(f"{c}:0" for c in reversed(codes))]
    if tool == "wtype":
        names = [_MODIFIERS[m][1] for m in mods]
        return ["wtype", *(a for n in names for a in ("-M", n)), "-k", _KEYS[key][1],
                *(a for n in reversed(names) for a in ("-m", n))]
    combo = [_MODIFIERS[m][2] for m in mods] + [_KEYS[key][2]]
    return ["xdotool", "key", "--clearmodifiers", "+".join(combo)]


def _terminal_focused() -> bool:
    global _focus_cache
    from voiceio import focus
    with _focus_lock:
        at, terminal = _focus_cache
        now = time.monotonic()
        if now - at > _FOCUS_TTL:
            window = focus.focused_window()
            terminal = focus.is_terminal(window)
            _focus_cache = (now, terminal)
            if terminal:
                log.debug("Focused window is a terminal (%s)", window.app_id)
        return terminal


class PasteKeys:
    """The configured chord, switched to ``terminal_keys`` while a terminal
    has focus — only when the configured chord is the default ctrl+v and the
    compositor can tell (Hyprland, sway, niri, X11)."""

    def __init__(self, keys: str = "ctrl+v", terminal_keys: str = ""):
        self.mods, self.key = parse_paste_keys(keys)
        self._terminal = (parse_paste_keys(terminal_keys)
                          if terminal_keys.strip() and (self.mods, self.key) == DEFAULT
                          else None)

    def for_focus(self) -> tuple[list[str], str]:
        if self._terminal is not None and _terminal_focused():
            return self._terminal
        return self.mods, self.key


def release_modifiers(tool: str, mods: list[str]) -> None:
    """Force-release modifiers a half-executed ydotool may have left down.

    `ydotool key 29:1 … 29:0` emits press and release as separate uinput
    events. Dying between them (timeout, ydotoold gone) leaves Ctrl held
    BELOW the compositor — every app, every keystroke, until the user taps
    Ctrl. Releasing an already-released key is a no-op, so this is always safe.
    """
    if tool != "ydotool" or not mods:
        return
    try:
        subprocess.run(["ydotool", "key", *(f"{_MODIFIERS[m][0]}:0" for m in mods)],
                       capture_output=True, timeout=2)
        log.debug("Released modifiers after failed ydotool paste")
    except (OSError, subprocess.TimeoutExpired):
        log.warning("Could not release modifiers after a failed ydotool "
                    "paste — if Ctrl seems stuck, tap it once to clear")


def press(tool: str, mods: list[str], key: str) -> None:
    """Press the chord. Raises CalledProcessError/TimeoutExpired/OSError."""
    try:
        subprocess.run(paste_command(tool, mods, key), check=True,
                       capture_output=True, timeout=3)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # A non-zero exit can still have emitted the press.
        release_modifiers(tool, mods)
        raise
