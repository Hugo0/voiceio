"""Socket-based hotkey backend: DE shortcut fires voiceio-toggle."""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio.sock"
DEBOUNCE_SECS = 0.8


class SocketHotkey:
    """Listens on a Unix DGRAM socket for 'toggle' commands."""

    name = "socket"

    def probe(self) -> ProbeResult:
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if not runtime_dir:
            return ProbeResult(ok=False, reason="XDG_RUNTIME_DIR not set",
                               fix_hint="Running under a normal user session should set this.")
        return ProbeResult(ok=True)

    def start(self, combo: str, on_trigger: Callable[[], None]) -> None:
        SOCKET_PATH.unlink(missing_ok=True)

        self._on_trigger = on_trigger
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(str(SOCKET_PATH))
        self._sock.settimeout(1.0)
        self._running = True
        self._last_trigger: float = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.debug("Socket listener started at %s", SOCKET_PATH)

    def _loop(self) -> None:
        while self._running:
            try:
                data = self._sock.recv(64)
                if data != b"toggle":
                    continue
                now = time.monotonic()
                if now - self._last_trigger < DEBOUNCE_SECS:
                    continue
                self._last_trigger = now
                self._on_trigger()
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_sock") and self._sock:
            self._sock.close()
        SOCKET_PATH.unlink(missing_ok=True)


def send_toggle() -> bool:
    """Send a toggle command to the running daemon."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(b"toggle", str(SOCKET_PATH))
        sock.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        log.error("Could not reach voiceio daemon: %s", e)
        return False
