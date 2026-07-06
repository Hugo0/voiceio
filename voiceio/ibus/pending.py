"""Pre-ready command buffer for the IBus engine.

Kept free of any ``gi`` dependency so it can be unit-tested without the IBus
GObject bindings installed.
"""
from __future__ import annotations

import time

PENDING_MAX_AGE = 3.0  # seconds


class PendingBuffer:
    """Buffers commands that arrive before the engine instance exists.

    Replaying stale commands into a window that gains focus much later would
    inject text the user never intended. So buffered commands are dropped once
    they age past ``max_age``, and the whole buffer is cleared when a fresh
    engine instance is created (see the engine's ``do_create_engine``).
    """

    def __init__(self, max_age: float = PENDING_MAX_AGE):
        self._items: list[tuple[float, str]] = []
        self._max_age = max_age

    def add(self, msg: str, now: float | None = None) -> None:
        self._items.append((time.monotonic() if now is None else now, msg))

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def drain_fresh(self, now: float | None = None) -> list[str]:
        """Return the non-stale buffered commands in order, emptying the buffer.

        Anything older than ``max_age`` is discarded rather than replayed.
        """
        t = time.monotonic() if now is None else now
        fresh = [msg for ts, msg in self._items if t - ts <= self._max_age]
        self._items.clear()
        return fresh
