"""In-process pystray backend (macOS primary, Linux fallback).

Used when AppIndicator subprocess isn't available (macOS, or Linux without
system python3-gi).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_icon = None
_thread: threading.Thread | None = None
_anim_stop: threading.Event | None = None

# PIL Image objects, loaded from PNGs on disk
_idle_img = None
_recording_frames: list = []


def start(
    quit_callback: Callable[[], None],
    idle_path: Path,
    frame_paths: list[Path],
) -> bool:
    """Start pystray icon. Returns True on success."""
    global _icon, _thread, _idle_img, _recording_frames

    try:
        import pystray
        from PIL import Image
    except ImportError:
        log.warning("pystray/Pillow not installed, tray icon disabled")
        return False

    _idle_img = Image.open(idle_path)
    _recording_frames = [Image.open(p) for p in frame_paths]

    def _open_terminal(action: str):
        def _handler():
            from voiceio.tray import _MENU_COMMANDS
            from voiceio.platform import open_in_terminal
            cli_cmd = _MENU_COMMANDS.get(action)
            if cli_cmd:
                open_in_terminal(cli_cmd)
        return _handler

    _icon = pystray.Icon(
        "voiceio",
        icon=_idle_img,
        title="voiceio - idle",
        menu=pystray.Menu(
            pystray.MenuItem("Review corrections...", _open_terminal("correct")),
            pystray.MenuItem("View history...", _open_terminal("history")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Demo...", _open_terminal("demo")),
            pystray.MenuItem("Doctor...", _open_terminal("doctor")),
            pystray.MenuItem("View logs...", _open_terminal("logs")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: quit_callback()),
        ),
    )

    _thread = threading.Thread(target=_icon.run, daemon=True, name="tray-pystray")
    _thread.start()
    return True


def set_recording(recording: bool) -> None:
    global _anim_stop
    if _icon is None:
        return

    if recording:
        _icon.title = "voiceio - recording"
        if _anim_stop is None:
            _anim_stop = threading.Event()
            threading.Thread(
                target=_animate, daemon=True, name="tray-anim",
            ).start()
    else:
        if _anim_stop is not None:
            _anim_stop.set()
            _anim_stop = None
        if _idle_img is not None:
            _icon.icon = _idle_img
        _icon.title = "voiceio - idle"


def _animate() -> None:
    frame = 0
    interval = 1.0 / 6  # ~6fps
    stop_evt = _anim_stop
    while stop_evt is not None and not stop_evt.is_set():
        if _icon is not None and _recording_frames:
            try:
                _icon.icon = _recording_frames[frame % len(_recording_frames)]
            except Exception:
                pass
        frame += 1
        if stop_evt is not None:
            stop_evt.wait(interval)


def set_title(title: str) -> None:
    if _icon is not None:
        _icon.title = title


def stop() -> None:
    global _icon, _anim_stop
    if _anim_stop is not None:
        _anim_stop.set()
        _anim_stop = None
    if _icon is not None:
        try:
            _icon.stop()
        except Exception:
            pass
        _icon = None
