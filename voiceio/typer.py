from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


def type_text(text: str, method: str = "xdotool") -> None:
    if not text:
        return

    if method == "xclip":
        _type_via_clipboard(text)
    else:
        _type_via_xdotool(text)


def _type_via_xdotool(text: str) -> None:
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
            check=True,
        )
    except FileNotFoundError:
        log.error("xdotool not found — install it: sudo apt install xdotool")
    except subprocess.CalledProcessError as e:
        log.error("xdotool failed: %s", e)


def _type_via_clipboard(text: str) -> None:
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode(),
            check=True,
        )
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            check=True,
        )
    except FileNotFoundError:
        log.error("xclip/xdotool not found — install them: sudo apt install xclip xdotool")
    except subprocess.CalledProcessError as e:
        log.error("Clipboard paste failed: %s", e)
