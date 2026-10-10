"""Cross-platform PID file locking."""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def lock_pid_file(fd) -> None:
    """Acquire an exclusive, non-blocking lock on an open file descriptor.

    Raises BlockingIOError if the lock is already held by another process.
    """
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        log.debug("PID lock acquired (msvcrt)")
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log.debug("PID lock acquired (fcntl)")


def daemon_alive(path) -> bool:
    """Whether a voiceio daemon holds the PID lock at `path` (probes without
    keeping the lock, and without truncating the file)."""
    if sys.platform == "win32":
        return True     # no cheap probe; callers then trust their other signal
    import fcntl
    try:
        with open(path, "a") as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    except OSError:
        return False
