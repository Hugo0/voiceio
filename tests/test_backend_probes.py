"""Smoke tests: call every backend's probe() on the real system.

These don't assert OK/FAIL (that depends on installed tools),
they just verify probe() doesn't crash or raise.
"""
from __future__ import annotations

import pytest

from voiceio.backends import ProbeResult


class TestHotkeyProbes:
    def test_socket_probe(self):
        from voiceio.hotkeys.socket_backend import SocketHotkey
        result = SocketHotkey().probe()
        assert isinstance(result, ProbeResult)

    def test_evdev_probe(self):
        from voiceio.hotkeys.evdev import EvdevHotkey
        result = EvdevHotkey().probe()
        assert isinstance(result, ProbeResult)

    def test_pynput_probe(self):
        from voiceio.hotkeys.pynput_backend import PynputHotkey
        result = PynputHotkey().probe()
        assert isinstance(result, ProbeResult)


class TestTyperProbes:
    def test_xdotool_probe(self):
        from voiceio.typers.xdotool import XdotoolTyper
        result = XdotoolTyper().probe()
        assert isinstance(result, ProbeResult)

    def test_ydotool_probe(self):
        from voiceio.typers.ydotool import YdotoolTyper
        result = YdotoolTyper().probe()
        assert isinstance(result, ProbeResult)

    def test_wtype_probe(self):
        from voiceio.typers.wtype import WtypeTyper
        result = WtypeTyper().probe()
        assert isinstance(result, ProbeResult)

    def test_clipboard_probe(self):
        from voiceio.typers.clipboard import ClipboardTyper
        result = ClipboardTyper().probe()
        assert isinstance(result, ProbeResult)

    def test_pynput_probe(self):
        from voiceio.typers.pynput_type import PynputTyper
        result = PynputTyper().probe()
        assert isinstance(result, ProbeResult)


class TestChainResolution:
    """Verify chain resolution doesn't crash for any platform combo."""

    @pytest.mark.parametrize("display,desktop", [
        ("x11", "gnome"), ("x11", "kde"), ("x11", "unknown"),
        ("wayland", "gnome"), ("wayland", "kde"), ("wayland", "sway"),
        ("wayland", "hyprland"), ("wayland", "cosmic"), ("wayland", "unknown"),
        ("quartz", "macos"),
    ])
    def test_hotkey_chain_resolves(self, display, desktop):
        from voiceio.platform import Platform
        from voiceio.hotkeys import chain
        p = Platform(os="linux" if display != "quartz" else "darwin",
                     display_server=display, desktop=desktop)
        results = chain.resolve(p)
        assert len(results) > 0
        for name, backend, probe in results:
            assert isinstance(probe, ProbeResult)

    @pytest.mark.parametrize("display,desktop", [
        ("x11", "gnome"), ("x11", "unknown"),
        ("wayland", "gnome"), ("wayland", "kde"), ("wayland", "sway"),
        ("wayland", "hyprland"), ("wayland", "unknown"),
        ("quartz", "macos"),
    ])
    def test_typer_chain_resolves(self, display, desktop):
        from voiceio.platform import Platform
        from voiceio.typers import chain
        p = Platform(os="linux" if display != "quartz" else "darwin",
                     display_server=display, desktop=desktop)
        results = chain.resolve(p)
        assert len(results) > 0
        for name, backend, probe in results:
            assert isinstance(probe, ProbeResult)
