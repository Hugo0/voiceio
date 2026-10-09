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


# ── Distro portability ───────────────────────────────────────────────────

import os  # noqa: E402
import stat  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

import voiceio.platform as plat  # noqa: E402


@pytest.mark.parametrize("env,expected", [
    ({"XMODIFIERS": "@im=fcitx"}, "fcitx"),
    ({"GTK_IM_MODULE": "fcitx5"}, "fcitx"),
    ({"XMODIFIERS": "@im=ibus"}, "ibus"),
    ({}, ""),
])
def test_detect_input_method(env, expected):
    with patch.dict(os.environ, env, clear=True):
        assert plat._detect_input_method() == expected


@pytest.fixture
def fake_pythons(tmp_path, monkeypatch):
    """Two executables: one whose probe fails, one that passes."""
    def make(name, rc):
        p = tmp_path / name
        p.write_text(f"#!/bin/sh\nexit {rc}\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return str(p)
    bad, good = make("bad-python", 1), make("good-python", 0)
    monkeypatch.setattr(plat, "_system_python_hits", {})
    monkeypatch.setattr(plat, "_SYSTEM_PYTHON_PATHS", (bad, good))
    monkeypatch.setattr(plat.shutil, "which", lambda _: None)
    monkeypatch.delenv("VOICEIO_SYSTEM_PYTHON", raising=False)
    return bad, good


def test_find_system_python_skips_failing_probe(fake_pythons):
    _, good = fake_pythons
    assert plat.find_system_python("probe") == good


def test_find_system_python_env_override_wins(fake_pythons, monkeypatch):
    bad, good = fake_pythons
    monkeypatch.setattr(plat, "_SYSTEM_PYTHON_PATHS", (good,))
    monkeypatch.setenv("VOICEIO_SYSTEM_PYTHON", good)
    assert plat.find_system_python("probe") == good


def test_find_system_python_falls_back_to_running_interpreter(fake_pythons, monkeypatch):
    bad, _ = fake_pythons
    monkeypatch.setattr(plat, "_SYSTEM_PYTHON_PATHS", (bad,))
    assert plat.find_system_python("import sys") == sys.executable


def test_find_system_python_does_not_cache_misses(fake_pythons, monkeypatch):
    bad, good = fake_pythons
    monkeypatch.setattr(plat, "_SYSTEM_PYTHON_PATHS", (bad,))
    monkeypatch.setattr(plat.sys, "executable", bad)
    assert plat.find_system_python("probe") is None
    monkeypatch.setattr(plat, "_SYSTEM_PYTHON_PATHS", (good,))
    assert plat.find_system_python("probe") == good


def test_pkg_manager_nixos_wins(monkeypatch):
    plat._detect_pkg_manager.cache_clear()
    monkeypatch.setattr(plat, "is_nixos", lambda: True)
    try:
        assert plat._detect_pkg_manager() == "nix"
        assert plat.pkg_install("python3-gi") == (
            "add to environment.systemPackages: python3Packages.pygobject3"
        )
    finally:
        plat._detect_pkg_manager.cache_clear()
