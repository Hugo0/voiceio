"""Tests for platform detection."""
from __future__ import annotations

from unittest.mock import patch

from voiceio.platform import Platform, _detect_os, _detect_display_server, _detect_desktop


def test_detect_os_linux():
    with patch("voiceio.platform.sys") as mock_sys:
        mock_sys.platform = "linux"
        assert _detect_os() == "linux"


def test_detect_os_darwin():
    with patch("voiceio.platform.sys") as mock_sys:
        mock_sys.platform = "darwin"
        assert _detect_os() == "darwin"


def test_detect_display_wayland():
    with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_display_server() == "wayland"


def test_detect_display_x11():
    with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_display_server() == "x11"


def test_detect_display_fallback_wayland():
    with patch.dict("os.environ", {"XDG_SESSION_TYPE": "", "WAYLAND_DISPLAY": "wayland-0"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_display_server() == "wayland"


def test_detect_desktop_gnome():
    with patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": "GNOME"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_desktop() == "gnome"


def test_detect_desktop_kde():
    with patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": "KDE"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_desktop() == "kde"


def test_detect_desktop_sway():
    with patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": "sway"}, clear=False):
        with patch("voiceio.platform._detect_os", return_value="linux"):
            assert _detect_desktop() == "sway"


def test_detect_desktop_macos():
    with patch("voiceio.platform._detect_os", return_value="darwin"):
        assert _detect_desktop() == "macos"


def test_platform_properties():
    p = Platform(os="linux", display_server="wayland", desktop="gnome")
    assert p.is_linux
    assert not p.is_mac
    assert p.is_wayland
    assert not p.is_x11


def test_platform_frozen():
    p = Platform(os="linux", display_server="x11", desktop="unknown")
    import pytest
    with pytest.raises(AttributeError):
        p.os = "darwin"


class TestOpenInTerminal:
    """The tray menu launches print-and-exit commands (history/doctor/logs);
    hold=True must keep the window open instead of flashing it shut."""

    def _capture(self, monkeypatch, *, hold):
        import voiceio.platform as plat
        monkeypatch.setattr(plat, "_detect_os", lambda: "linux")
        monkeypatch.setenv("TERMINAL", "myterm")
        monkeypatch.setattr(
            plat.shutil, "which",
            lambda b: "/usr/bin/myterm" if b == "myterm" else None,
        )
        calls = []
        with patch("subprocess.Popen", side_effect=lambda args, *a, **k: calls.append(args)):
            ok = plat.open_in_terminal(["voiceio", "history"], hold=hold)
        assert ok is True
        return calls[0]

    def test_hold_wraps_in_shell_that_pauses(self, monkeypatch):
        args = self._capture(monkeypatch, hold=True)
        assert args[:2] == ["myterm", "-e"]
        assert "sh" in args and "-c" in args
        script = args[-1]
        assert "voiceio history" in script
        assert "read" in script  # pauses so the window survives

    def test_no_hold_runs_command_directly(self, monkeypatch):
        args = self._capture(monkeypatch, hold=False)
        assert args == ["myterm", "-e", "voiceio", "history"]
