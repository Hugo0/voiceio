"""Fallback chain for hotkey backends."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from voiceio.backends import ProbeResult
from voiceio.hotkeys.base import HotkeyBackend

if TYPE_CHECKING:
    from voiceio.platform import Platform

log = logging.getLogger(__name__)

# Preference order by platform
_CHAINS: dict[tuple[str, str], list[str]] = {
    # (display_server, desktop) -> backend list
    ("x11", "*"):       ["pynput", "evdev", "socket"],
    ("wayland", "*"):   ["evdev", "socket"],
    ("quartz", "*"):    ["pynput"],
}


def _get_chain(platform: Platform) -> list[str]:
    """Get the preference chain for this platform."""
    # Try exact match first
    key = (platform.display_server, platform.desktop)
    if key in _CHAINS:
        return _CHAINS[key]
    # Try wildcard desktop
    key = (platform.display_server, "*")
    if key in _CHAINS:
        return _CHAINS[key]
    # Fallback
    return ["socket"]


def resolve(platform: Platform, override: str | None = None) -> list[tuple[str, HotkeyBackend, ProbeResult]]:
    """Probe backends in preference order, return list of (name, backend, probe_result).

    If override is set (not "auto"/None), only probe that one backend.
    """
    from voiceio.hotkeys import create_hotkey_backend

    if override and override != "auto":
        backend = create_hotkey_backend(override, platform)
        result = backend.probe()
        return [(override, backend, result)]

    chain = _get_chain(platform)
    results = []
    for name in chain:
        try:
            backend = create_hotkey_backend(name, platform)
            result = backend.probe()
            results.append((name, backend, result))
        except Exception as e:
            log.debug("Failed to create backend '%s': %s", name, e)
            results.append((name, None, ProbeResult(ok=False, reason=str(e))))

    return results


def select(platform: Platform, override: str | None = None) -> HotkeyBackend:
    """Select the first working hotkey backend.

    Raises RuntimeError if none work.
    """
    results = resolve(platform, override)

    for name, backend, probe in results:
        if probe.ok and backend is not None:
            log.info("Selected hotkey backend: %s", name)
            return backend
        log.debug("Hotkey backend '%s' unavailable: %s", name, probe.reason)

    # Build error message
    reasons = [f"  {name}: {probe.reason}" for name, _, probe in results if not probe.ok]
    hints = [probe.fix_hint for _, _, probe in results if probe.fix_hint]
    msg = "No working hotkey backend found:\n" + "\n".join(reasons)
    if hints:
        msg += "\n\nTo fix:\n" + "\n".join(f"  - {h}" for h in hints)
    raise RuntimeError(msg)
