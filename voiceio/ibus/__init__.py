"""IBus input method engine for VoiceIO."""
from __future__ import annotations

import os
from pathlib import Path

SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.sock"
READY_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.ready"


def ping_engine(timeout: float = 2.0) -> bool:
    """True if a live engine answers a ping on SOCKET_PATH.

    Used by the daemon's health watchdog, and by the engine itself at
    startup to refuse to double-run: a second engine binding the socket
    path would silently starve the healthy instance (its listener stays
    blocked on a socket inode that no longer receives anything).
    """
    import socket

    if not SOCKET_PATH.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sock.bind("")  # Linux autobind: gives us an address for the reply
        sock.sendto(b"ping", str(SOCKET_PATH))
        data, _ = sock.recvfrom(64)
        return data == b"pong"
    except OSError:
        return False
    finally:
        sock.close()
