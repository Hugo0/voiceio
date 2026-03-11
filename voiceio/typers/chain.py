"""Fallback chain for typer backends."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from voiceio.backends import ProbeResult
from voiceio.typers.base import TyperBackend

if TYPE_CHECKING:
    from voiceio.platform import Platform

log = logging.getLogger(__name__)

# Preference order: (display_server, desktop) -> backend list
_CHAINS: dict[tuple[str, str], list[str]] = {
    ("x11", "*"):               ["ibus", "xdotool", "clipboard"],
    ("wayland", "gnome"):       ["ibus", "ydotool", "clipboard"],
    ("wayland", "kde"):         ["ibus", "ydotool", "clipboard"],
    ("wayland", "sway"):        ["ibus", "wtype", "ydotool", "clipboard"],
    ("wayland", "hyprland"):    ["ibus", "wtype", "ydotool", "clipboard"],
    ("wayland", "*"):           ["ibus", "ydotool", "wtype", "clipboard"],
    ("quartz", "*"):            ["pynput", "clipboard"],
    ("win32", "*"):             ["pynput", "clipboard"],
}


def _get_chain(platform: Platform) -> list[str]:
    """Get the preference chain for this platform."""
    key = (platform.display_server, platform.desktop)
    if key in _CHAINS:
        return _CHAINS[key]
    key = (platform.display_server, "*")
    if key in _CHAINS:
        return _CHAINS[key]
    return ["clipboard"]


def resolve(platform: Platform, override: str | None = None, **kwargs) -> list[tuple[str, TyperBackend | None, ProbeResult]]:
    """Probe backends in preference order."""
    from voiceio.typers import create_typer_backend

    if override and override != "auto":
        backend = create_typer_backend(override, platform, **kwargs)
        result = backend.probe()
        return [(override, backend, result)]

    chain = _get_chain(platform)
    results = []
    for name in chain:
        try:
            backend = create_typer_backend(name, platform, **kwargs)
            result = backend.probe()
            results.append((name, backend, result))
        except Exception as e:
            log.debug("Failed to create typer '%s': %s", name, e)
            results.append((name, None, ProbeResult(ok=False, reason=str(e))))

    return results


def select(platform: Platform, override: str | None = None, **kwargs) -> TyperBackend:
    """Select the first working typer backend.

    Raises RuntimeError if none work.
    """
    results = resolve(platform, override, **kwargs)

    for name, backend, probe in results:
        if probe.ok and backend is not None:
            log.info("Selected typer backend: %s", name)
            return backend
        log.debug("Typer backend '%s' unavailable: %s", name, probe.reason)

    reasons = [f"  {name}: {probe.reason}" for name, _, probe in results if not probe.ok]
    hints = [probe.fix_hint for _, _, probe in results if probe.fix_hint]
    msg = "No working typer backend found:\n" + "\n".join(reasons)
    if hints:
        msg += "\n\nTo fix:\n" + "\n".join(f"  - {h}" for h in hints)
    raise RuntimeError(msg)
