"""Tests for fallback chain resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voiceio.backends import ProbeResult
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.typers import chain as typer_chain
from voiceio.platform import Platform


def _mock_hotkey_backend(name: str, ok: bool):
    b = MagicMock()
    b.name = name
    b.probe.return_value = ProbeResult(ok=ok, reason="" if ok else f"{name} unavailable")
    return b


def _mock_typer_backend(name: str, ok: bool):
    b = MagicMock()
    b.name = name
    b.probe.return_value = ProbeResult(ok=ok, reason="" if ok else f"{name} unavailable")
    return b


class TestHotkeyChain:
    def test_x11_prefers_pynput(self, linux_x11):
        chain = hotkey_chain._get_chain(linux_x11)
        assert chain[0] == "pynput"

    def test_wayland_prefers_evdev(self, linux_wayland_gnome):
        chain = hotkey_chain._get_chain(linux_wayland_gnome)
        assert chain[0] == "evdev"

    def test_macos_uses_pynput(self, macos):
        chain = hotkey_chain._get_chain(macos)
        assert chain == ["pynput"]

    def test_select_first_ok(self, linux_x11):
        backends = {
            "pynput": _mock_hotkey_backend("pynput", ok=False),
            "evdev": _mock_hotkey_backend("evdev", ok=True),
            "socket": _mock_hotkey_backend("socket", ok=True),
        }
        with patch("voiceio.hotkeys.create_hotkey_backend", side_effect=lambda n, p: backends[n]):
            result = hotkey_chain.select(linux_x11)
            assert result.name == "evdev"

    def test_select_raises_when_none_work(self, linux_x11):
        backends = {
            "pynput": _mock_hotkey_backend("pynput", ok=False),
            "evdev": _mock_hotkey_backend("evdev", ok=False),
            "socket": _mock_hotkey_backend("socket", ok=False),
        }
        with patch("voiceio.hotkeys.create_hotkey_backend", side_effect=lambda n, p: backends[n]):
            with pytest.raises(RuntimeError, match="No working hotkey backend"):
                hotkey_chain.select(linux_x11)

    def test_override_bypasses_chain(self, linux_x11):
        backend = _mock_hotkey_backend("socket", ok=True)
        with patch("voiceio.hotkeys.create_hotkey_backend", return_value=backend):
            result = hotkey_chain.select(linux_x11, override="socket")
            assert result.name == "socket"


class TestTyperChain:
    def test_x11_prefers_ibus(self, linux_x11):
        chain = typer_chain._get_chain(linux_x11)
        assert chain[0] == "ibus"
        assert chain[1] == "xdotool"

    def test_wayland_gnome_prefers_ibus(self, linux_wayland_gnome):
        chain = typer_chain._get_chain(linux_wayland_gnome)
        assert chain[0] == "ibus"
        assert chain[1] == "ydotool"

    def test_wayland_sway_prefers_ibus(self, linux_wayland_sway):
        chain = typer_chain._get_chain(linux_wayland_sway)
        assert chain[0] == "ibus"
        assert chain[1] == "wtype"

    def test_macos_prefers_pynput(self, macos):
        chain = typer_chain._get_chain(macos)
        assert chain[0] == "pynput"

    def test_select_falls_back(self, linux_wayland_gnome):
        backends = {
            "ydotool": _mock_typer_backend("ydotool", ok=False),
            "clipboard": _mock_typer_backend("clipboard", ok=True),
        }
        with patch("voiceio.typers.create_typer_backend", side_effect=lambda n, p: backends[n]):
            result = typer_chain.select(linux_wayland_gnome)
            assert result.name == "clipboard"

    def test_wildcard_fallback(self):
        """Unknown desktop on wayland should use wildcard chain."""
        p = Platform(os="linux", display_server="wayland", desktop="cosmic")
        chain = typer_chain._get_chain(p)
        assert "ydotool" in chain
