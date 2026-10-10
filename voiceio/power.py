"""Is the machine running on battery? (Linux sysfs; False elsewhere.)"""
from __future__ import annotations

from pathlib import Path

POWER_SUPPLY = Path("/sys/class/power_supply")


def on_battery(root: Path | None = None) -> bool:
    """True when the machine has mains supplies and none is online. Two small
    sysfs reads: cheap enough to poll every few seconds."""
    mains = []
    for supply in (root or POWER_SUPPLY).glob("*"):
        try:
            if (supply / "type").read_text().strip() == "Mains":
                mains.append((supply / "online").read_text().strip() == "1")
        except OSError:
            continue
    return bool(mains) and not any(mains)
