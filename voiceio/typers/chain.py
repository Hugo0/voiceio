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
    # fcitx-bridge only probes OK when fcitx5 runs with the bridge addon, in
    # which case IBus is absent and it beats simulated keystrokes.
    ("x11", "*"):               ["ibus", "fcitx-bridge", "xdotool", "clipboard"],
    ("wayland", "gnome"):       ["ibus", "fcitx-bridge", "ydotool", "clipboard"],
    ("wayland", "kde"):         ["ibus", "fcitx-bridge", "ydotool", "clipboard"],
    ("wayland", "sway"):        ["ibus", "fcitx-bridge", "wtype", "ydotool", "clipboard"],
    ("wayland", "hyprland"):    ["ibus", "fcitx-bridge", "wtype", "ydotool", "clipboard"],
    ("wayland", "*"):           ["ibus", "fcitx-bridge", "ydotool", "wtype", "clipboard"],
    ("quartz", "*"):            ["pynput", "clipboard"],
    ("win32", "*"):             ["pynput", "clipboard"],
}


def ibus_in_use(display_server: str, input_method: str) -> bool:
    """Whether apps in this session hand their text input to IBus.

    On X11 apps only reach an input method through XMODIFIERS/GTK_IM_MODULE/
    QT_IM_MODULE (KDE on X11 often sets none), so IBus commits would reach
    nothing. Wayland compositors wire IBus in themselves.
    """
    return display_server != "x11" or input_method == "ibus"


def _get_chain(platform: Platform) -> list[str]:
    """Get the preference chain for this platform."""
    chain = (_CHAINS.get((platform.display_server, platform.desktop))
             or _CHAINS.get((platform.display_server, "*"))
             or ["clipboard"])
    if getattr(platform, "input_method", "") != "fcitx":
        # Only reachable through a running fcitx5; elsewhere it only adds a
        # failed probe and a red line in `voiceio doctor`.
        chain = [name for name in chain if name != "fcitx-bridge"]
    if not ibus_in_use(platform.display_server, getattr(platform, "input_method", "")):
        chain = [name for name in chain if name != "ibus"]
    return chain


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
