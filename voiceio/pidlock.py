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
