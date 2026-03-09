"""Hotkey detection backends."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.hotkeys.base import HotkeyBackend
    from voiceio.platform import Platform


def create_hotkey_backend(name: str, platform: Platform) -> HotkeyBackend:
    """Create a hotkey backend by name."""
    if name == "evdev":
        from voiceio.hotkeys.evdev import EvdevHotkey
        return EvdevHotkey()
    if name == "pynput":
        from voiceio.hotkeys.pynput_backend import PynputHotkey
        return PynputHotkey()
    if name == "socket":
        from voiceio.hotkeys.socket_backend import SocketHotkey
        return SocketHotkey()
    raise ValueError(f"Unknown hotkey backend: {name}")
