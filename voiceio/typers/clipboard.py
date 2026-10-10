"""Clipboard-based text injection, the universal fallback."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading

from voiceio import clipboard_read
from voiceio.backends import ProbeResult
from voiceio.typers import paste
from voiceio.typers.base import KeystrokeTyper

log = logging.getLogger(__name__)

# Long enough for the target app to have read the clipboard after the paste
# keystroke; each further paste of a streaming burst restarts the wait.
RESTORE_DELAY_SECS = 0.5


class ClipboardTyper(KeystrokeTyper):
    """Type text by copying to clipboard and simulating Ctrl+V / Cmd+V.

    On Linux the paste keystroke is ``paste_keys``, or ``terminal_paste_keys``
    while a detectable terminal has focus (see typers/paste.py). With
    ``restore_clipboard`` the user's clipboard text comes back once the paste
    has been read, unless something else was copied since.
    """

    name = "clipboard"

    def __init__(self, platform=None, paste_keys: str = "ctrl+v",
                 terminal_paste_keys: str = "", restore_clipboard: bool = False):
        self._keys = paste.PasteKeys(paste_keys, terminal_paste_keys)
        self._paste_mods, self._paste_key = self._keys.mods, self._keys.key
        self._restore_clipboard = restore_clipboard
        self._restore_lock = threading.Lock()
        self._restore_timer: threading.Timer | None = None
        self._restore_seq = 0
        self._saved: str | None = None
        self._pasted = ""
        self._copy_cmd: list[str] | None = None
        self._paste_tool: str | None = None   # "ydotool" | "wtype" | "xdotool" | "win32_pynput"
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
            self._paste_tool = "win32_pynput"
            self._delete_tool = ["win32_pynput"]
            log.debug("Clipboard typer: Windows mode (pyperclip + pynput)")
            return

        session = os.environ.get("XDG_SESSION_TYPE", "")
        if session == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("wl-copy"):
                self._copy_cmd = ["wl-copy", "--"]
                for tool in ("ydotool", "wtype"):
                    if shutil.which(tool):
                        self._paste_tool = tool
                        self._delete_tool = [tool]
                        break
        else:
            if shutil.which("xclip") and shutil.which("xdotool"):
                self._copy_cmd = ["xclip", "-selection", "clipboard"]
                self._paste_tool = "xdotool"
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
                fix_hint="Install xclip + xdotool (X11), wl-copy + ydotool/wtype (Wayland), or pbcopy (macOS).",
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

    def _type(self, text: str) -> None:
        if not self._restore_clipboard:
            self._paste(text)
            return
        with self._restore_lock:
            if self._restore_timer is None:   # first paste of a burst
                self._saved = clipboard_read.read_clipboard()
            else:
                self._restore_timer.cancel()
            try:
                self._paste(text)
            finally:
                self._pasted = text
                self._restore_seq += 1
                self._restore_timer = threading.Timer(
                    RESTORE_DELAY_SECS, self._restore_saved, args=(self._restore_seq,))
                self._restore_timer.daemon = True
                self._restore_timer.start()

    def _restore_saved(self, seq: int) -> None:
        """Put the user's clipboard text back, if what we pasted is still there
        (they may have copied something since). Text only: an image or file
        that was on the clipboard is not restored."""
        with self._restore_lock:
            if seq != self._restore_seq:   # a later paste rescheduled this
                return
            self._restore_timer = None
            saved, self._saved = self._saved, None
            if saved is None:
                return
            current = clipboard_read.read_clipboard()
            if current is not None and current.strip() == self._pasted.strip():
                clipboard_read.copy_text(saved)
                log.debug("Restored the clipboard after pasting")

    def _paste(self, text: str) -> None:
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
            paste.press(self._paste_tool, *self._keys.for_focus())

    def _delete(self, n: int) -> None:
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
