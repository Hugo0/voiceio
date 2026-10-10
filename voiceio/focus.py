"""Which window has keyboard focus, where the compositor will say.

Hyprland, sway and niri answer over their IPC; an X11 session through
xdotool. GNOME and KDE on Wayland expose nothing to other clients, so there
the answer is None. Never raises.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

_which = functools.lru_cache(maxsize=16)(shutil.which)
_TIMEOUT = 1.0

# Wayland app_id or X11 WM_CLASS, lowercased; reverse-DNS ids also match on
# their last component (org.wezfurlong.wezterm → wezterm).
TERMINALS = frozenset({
    "alacritty", "blackbox", "contour", "cool-retro-term", "deepin-terminal",
    "foot", "footclient", "ghostty", "gnome-terminal", "gnome-terminal-server",
    "guake", "hyper", "kgx", "kitty", "konsole", "lxterminal", "mate-terminal",
    "org.gnome.console", "ptyxis", "qterminal", "rio", "rxvt", "sakura", "st",
    "st-256color", "tabby", "terminator", "terminology", "termite", "tilix",
    "urxvt", "warp", "wezterm", "xfce4-terminal", "xterm", "yakuake",
})


@dataclass(frozen=True)
class FocusedWindow:
    app_id: str      # Wayland app_id, or the X11 WM_CLASS class
    title: str = ""


def _run(argv: list[str]) -> str | None:
    if not _which(argv[0]):
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout if out.returncode == 0 else None


def _json(argv: list[str]):
    out = _run(argv)
    try:
        return json.loads(out) if out else None
    except ValueError:
        return None


def _hyprland() -> FocusedWindow | None:
    w = _json(["hyprctl", "activewindow", "-j"])
    if not isinstance(w, dict) or not w.get("class"):
        return None
    return FocusedWindow(w["class"], w.get("title") or "")


def _sway() -> FocusedWindow | None:
    stack = [_json(["swaymsg", "-t", "get_tree"])]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("focused"):
            app = node.get("app_id") or (node.get("window_properties") or {}).get("class")
            return FocusedWindow(app, node.get("name") or "") if app else None
        stack.extend(node.get("nodes", []) + node.get("floating_nodes", []))
    return None


def _niri() -> FocusedWindow | None:
    w = _json(["niri", "msg", "-j", "focused-window"])
    if not isinstance(w, dict) or not w.get("app_id"):
        return None
    return FocusedWindow(w["app_id"], w.get("title") or "")


def _x11() -> FocusedWindow | None:
    title = (_run(["xdotool", "getactivewindow", "getwindowname"]) or "").strip()
    app = (_run(["xdotool", "getactivewindow", "getwindowclassname"]) or "").strip()
    if not app:   # xdotool before 3.2021 has no getwindowclassname
        wid = (_run(["xdotool", "getactivewindow"]) or "").strip()
        props = _run(["xprop", "-id", wid, "WM_CLASS"]) if wid else None
        # WM_CLASS(STRING) = "instance", "Class"
        names = re.findall(r'"([^"]*)"', props or "")
        app = names[-1] if names else ""
    if not (app or title):
        return None
    return FocusedWindow(app, title)


def focused_window(*, xwayland: bool = False) -> FocusedWindow | None:
    """The focused window, or None when this session cannot tell.

    ``xwayland`` also asks the X server on a Wayland session. Its answer is
    only right while an X11 app has focus, so use it for best-effort context
    (a title for history), never to decide what keys to press.
    """
    env = os.environ
    if env.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return _hyprland()
    if env.get("SWAYSOCK"):
        return _sway()
    if env.get("NIRI_SOCKET"):
        return _niri()
    wayland = env.get("XDG_SESSION_TYPE") == "wayland" or bool(env.get("WAYLAND_DISPLAY"))
    if env.get("DISPLAY") and (xwayland or not wayland):
        return _x11()
    return None


def is_terminal(window: FocusedWindow | None) -> bool:
    if window is None:
        return False
    app = window.app_id.lower().removesuffix(".desktop")
    return app in TERMINALS or app.rsplit(".", 1)[-1] in TERMINALS
