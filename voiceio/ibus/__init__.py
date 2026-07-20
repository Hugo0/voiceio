"""IBus input method engine for VoiceIO."""
from __future__ import annotations

import os
from pathlib import Path

SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.sock"
READY_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.ready"
# Held for the engine's lifetime so exactly one process ever owns the command
# socket — see acquire_singleton_lock().
LOCK_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.lock"


def acquire_singleton_lock() -> int | None:
    """Take an exclusive, lifetime-held lock on LOCK_PATH.

    Returns the open fd on success (the caller must keep it open — the lock is
    released when the fd closes, i.e. when the process dies), or None if another
    engine already holds it.

    This is the race-free gate against duplicate engines. ibus-daemon
    exec-spawns its own copy of the engine when something activates the voiceio
    source (GNOME's per-window source memory / Super+Space — voiceio sits first
    in mru-sources). A ping-then-bind check is check-then-act: two engines
    starting close together both see "no healthy peer", then both unlink+rebind
    the DGRAM socket in _socket_listener. The loser is left bound to an orphaned
    inode — alive but never receiving pings ("zombie") — which the daemon's
    health watchdog then kills and respawns, churning ~15x/day. flock is atomic:
    the loser never touches the socket, it just exits.
    """
    import fcntl

    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


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
