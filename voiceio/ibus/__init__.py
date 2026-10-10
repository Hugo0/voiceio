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


ENGINE_MODULE = "voiceio.ibus.engine"
LAUNCHER_NAME = "voiceio-ibus-engine"


def engine_pids() -> list[int]:
    """PIDs of this user's running engines (and their launcher shells).

    Matched on exact argv tokens, never a pattern: `pgrep -f
    voiceio.ibus.engine` is a regex whose dots also match
    "voiceio-ibus-engine.log", so it killed anything mentioning the log — a
    `tail -f` of it, an editor, the shell that ran the restart.
    """
    uid, me, pids = os.getuid(), os.getpid(), []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv if a]
        is_engine = any(a == "-m" and b == ENGINE_MODULE for a, b in zip(args, args[1:]))
        is_launcher = len(args) >= 2 and Path(args[0]).name in ("bash", "sh") \
            and Path(args[1]).name == LAUNCHER_NAME
        if is_engine or is_launcher:
            pids.append(int(entry.name))
    return pids
