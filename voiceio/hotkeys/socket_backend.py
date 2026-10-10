"""Socket-based hotkey backend: DE shortcut fires voiceio-toggle.

On Linux/macOS: uses a Unix datagram socket at $XDG_RUNTIME_DIR/voiceio.sock.
On Windows: uses a UDP socket on localhost (127.0.0.1:19384).
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_UDP_PORT = 19384
SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio.sock"
DEBOUNCE_SECS = 0.8


class SocketHotkey:
    """Listens on a Unix DGRAM socket (or UDP on Windows) for commands.

    "toggle" fires on_trigger (it is the hotkey: debounced against evdev
    seeing the same keypress); handle() adds others ("press", "release"...).
    A DE shortcut runs a command on key down only, so this backend reports
    no release unless the compositor binds one (`voiceio toggle --release`).
    """

    name = "socket"
    # Commands that may be the same keypress also seen by evdev/pynput.
    _DEBOUNCED = frozenset({"toggle", "press"})

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[], None]] = {}

    def handle(self, command: str, fn: Callable[[], None]) -> None:
        self._handlers[command] = fn

    def probe(self) -> ProbeResult:
        if _IS_WINDOWS:
            log.debug("Socket probe: Windows, using UDP localhost:%d", _UDP_PORT)
            return ProbeResult(ok=True)
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if not runtime_dir:
            return ProbeResult(ok=False, reason="XDG_RUNTIME_DIR not set",
                               fix_hint="Running under a normal user session should set this.")
        return ProbeResult(ok=True)

    def start(self, combo: str, on_trigger: Callable[[], None],
              on_release: Callable[[], None] | None = None) -> None:
        self._handlers["toggle"] = on_trigger
        self._running = True
        self._last_trigger: float = 0

        if _IS_WINDOWS:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("127.0.0.1", _UDP_PORT))
            self._sock.settimeout(1.0)
            log.debug("Socket listener started on UDP 127.0.0.1:%d", _UDP_PORT)
        else:
            SOCKET_PATH.unlink(missing_ok=True)
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._sock.bind(str(SOCKET_PATH))
            self._sock.settimeout(1.0)
            log.debug("Socket listener started at %s", SOCKET_PATH)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                command = self._sock.recv(64).decode(errors="replace")
                fn = self._handlers.get(command)
                if fn is None:
                    continue
                if command in self._DEBOUNCED:
                    now = time.monotonic()
                    if now - self._last_trigger < DEBOUNCE_SECS:
                        continue
                    self._last_trigger = now
                fn()
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_sock") and self._sock:
            self._sock.close()
        if not _IS_WINDOWS:
            SOCKET_PATH.unlink(missing_ok=True)


def send_command(command: str = "toggle") -> bool:
    """Send a command to the running daemon."""
    try:
        if _IS_WINDOWS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(command.encode(), ("127.0.0.1", _UDP_PORT))
            sock.close()
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.sendto(command.encode(), str(SOCKET_PATH))
            sock.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        log.error("Could not reach voiceio daemon: %s", e)
        return False
