"""Clipboard-based text injection, the universal fallback."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


class ClipboardTyper:
    """Type text by copying to clipboard and simulating Ctrl+V / Cmd+V."""

    name = "clipboard"

    def __init__(self, platform=None):
        self._copy_cmd: list[str] | None = None
        self._paste_tool: list[str] | None = None
        self._delete_tool: list[str] | None = None
        self._tools_resolved = False
        self._pynput_kb = None  # cached pynput Controller for Windows

    def _resolve_tools(self) -> None:
        """Detect available tools once and cache."""
        if self._tools_resolved:
            return
        self._tools_resolved = True

        if sys.platform == "darwin":
            if shutil.which("pbcopy"):
                self._copy_cmd = ["pbcopy"]
            return

        if sys.platform == "win32":
            self._copy_cmd = ["win32_pyperclip"]  # use pyperclip (ctypes Win32 API)
            self._paste_tool = ["win32_pynput"]
            self._delete_tool = ["win32_pynput"]
            log.debug("Clipboard typer: Windows mode (pyperclip + pynput)")
            return

        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("wl-copy"):
                self._copy_cmd = ["wl-copy", "--"]
                if shutil.which("ydotool"):
                    self._paste_tool = ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]
                    self._delete_tool = ["ydotool"]
                elif shutil.which("wtype"):
                    self._paste_tool = ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
                    self._delete_tool = ["wtype"]
        else:
            if shutil.which("xclip") and shutil.which("xdotool"):
                self._copy_cmd = ["xclip", "-selection", "clipboard"]
                self._paste_tool = ["xdotool", "key", "--clearmodifiers", "ctrl+v"]
                self._delete_tool = ["xdotool"]

    def _get_pynput_kb(self):
        """Return a cached pynput keyboard Controller (Windows/macOS)."""
        if self._pynput_kb is None:
            from pynput.keyboard import Controller
            self._pynput_kb = Controller()
        return self._pynput_kb

    def reset_tools(self) -> None:
        """Clear cached tool resolution so next probe re-detects."""
        self._copy_cmd = None
        self._paste_tool = None
        self._delete_tool = None
        self._tools_resolved = False

    def probe(self) -> ProbeResult:
        self._resolve_tools()
        if self._copy_cmd is None or (
            sys.platform not in ("darwin", "win32") and self._paste_tool is None
        ):
            return ProbeResult(
                ok=False,
                reason="No clipboard tool found",
                fix_hint="Install xclip (X11), wl-copy (Wayland), or pbcopy (macOS).",
            )
        if sys.platform == "win32":
            try:
                import pyperclip  # noqa: F401
            except ImportError:
                return ProbeResult(
                    ok=False,
                    reason="pyperclip not installed",
                    fix_hint="pip install pyperclip",
                )
        return ProbeResult(ok=True)

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._resolve_tools()

        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True, capture_output=True)
            subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
                check=True, capture_output=True,
            )
            return

        if sys.platform == "win32":
            import pyperclip
            pyperclip.copy(text)
            import time
            time.sleep(0.05)
            from pynput.keyboard import Key
            kb = self._get_pynput_kb()
            with kb.pressed(Key.ctrl):
                kb.tap("v")
            log.debug("Clipboard typed %d chars via pyperclip+pynput", len(text))
            return

        if self._copy_cmd is None:
            raise RuntimeError("No clipboard tools available")

        subprocess.run(self._copy_cmd, input=text.encode(), check=True, capture_output=True)
        if self._paste_tool:
            subprocess.run(self._paste_tool, check=True, capture_output=True)

    def delete_chars(self, n: int) -> None:
        if n <= 0:
            return
        self._resolve_tools()

        if self._delete_tool and self._delete_tool[0] == "xdotool":
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "--delay", "12"] + ["BackSpace"] * n,
                check=True, capture_output=True,
            )
        elif self._delete_tool and self._delete_tool[0] == "ydotool":
            # Batch all backspaces into one subprocess call
            keys = []
            for _ in range(n):
                keys.extend(["14:1", "14:0"])
            subprocess.run(["ydotool", "key"] + keys, check=True, capture_output=True)
        elif self._delete_tool and self._delete_tool[0] == "wtype":
            # Batch: -k BackSpace -k BackSpace ...
            args = ["wtype"]
            for _ in range(n):
                args.extend(["-k", "BackSpace"])
            subprocess.run(args, check=True, capture_output=True)
        elif self._delete_tool and self._delete_tool[0] == "win32_pynput":
            from pynput.keyboard import Key
            kb = self._get_pynput_kb()
            for _ in range(n):
                kb.tap(Key.backspace)
        elif sys.platform == "darwin":
            script = f'tell application "System Events" to repeat {n} times\nkey code 51\nend repeat'
            subprocess.run(
                ["osascript", "-e", script],
                check=True, capture_output=True,
            )
