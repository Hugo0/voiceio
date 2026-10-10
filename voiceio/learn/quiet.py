"""Is the machine quiet enough for background learning?

Nightly learning is CPU-heavy for an hour or more. It runs at idle priority
regardless, but it should not even start while the machine is busy or on
battery: a laptop running on battery at 3am would just drain.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable

from voiceio.power import on_battery


def busy_reason(max_load: float) -> str | None:
    """Why learning should wait right now, or None when the machine is quiet."""
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if load > max_load:
        return f"busy (load {load:.2f} per CPU, waiting for {max_load:g})"
    if on_battery():
        return "on battery"
    return None


def wait_until_quiet(max_load: float, wait_hours: float, *, poll: float = 300,
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic,
                     say: Callable[[str], None] = print) -> bool:
    """Block until the machine is quiet. False when it stayed busy for wait_hours."""
    deadline = clock() + wait_hours * 3600
    last = None
    while (reason := busy_reason(max_load)) is not None:
        if reason != last:
            say(f"waiting: {reason}")
            last = reason
        if clock() + poll > deadline:
            say(f"still {reason} after {wait_hours:g} h; trying again at the next scheduled run")
            return False
        sleep(poll)
    return True
