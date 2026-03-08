from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_icon = None
_thread = None


def _make_icon(color: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = {"idle": (120, 120, 120, 255), "recording": (220, 40, 40, 255)}[color]
    draw.ellipse([8, 8, 56, 56], fill=fill)
    return img


def start(quit_callback) -> None:
    global _icon, _thread

    try:
        import pystray
    except ImportError:
        log.warning("pystray not installed — tray icon disabled. Install with: pip install voiceio[tray]")
        return

    _icon = pystray.Icon(
        "voiceio",
        icon=_make_icon("idle"),
        title="voiceio — idle",
        menu=pystray.Menu(pystray.MenuItem("Quit", lambda: quit_callback())),
    )

    _thread = threading.Thread(target=_icon.run, daemon=True)
    _thread.start()


def set_recording(recording: bool) -> None:
    if _icon is None:
        return
    state = "recording" if recording else "idle"
    _icon.icon = _make_icon(state)
    _icon.title = f"voiceio — {state}"


def stop() -> None:
    global _icon
    if _icon:
        _icon.stop()
        _icon = None
