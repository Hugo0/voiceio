"""Focused-window detection and the terminal paste chord it selects."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from voiceio import focus
from voiceio.focus import FocusedWindow
from voiceio.typers import paste
from voiceio.typers.ibus import IBusTyper

_SESSION_VARS = ("HYPRLAND_INSTANCE_SIGNATURE", "SWAYSOCK", "NIRI_SOCKET",
                 "WAYLAND_DISPLAY", "DISPLAY", "XDG_SESSION_TYPE")


@pytest.fixture
def session(monkeypatch):
    """Clear the session env, make every tool 'installed', and answer each
    command from `replies` (argv tuple → stdout; missing → exit 1)."""
    for var in _SESSION_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(focus, "_which", lambda name: f"/usr/bin/{name}")
    replies: dict[tuple, str] = {}
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        out = replies.get(tuple(argv))
        return MagicMock(returncode=0 if out is not None else 1, stdout=out or "")

    monkeypatch.setattr(focus.subprocess, "run", run)
    return monkeypatch, replies, calls


def test_hyprland(session):
    mp, replies, _ = session
    mp.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc")
    replies[("hyprctl", "activewindow", "-j")] = json.dumps({"class": "kitty", "title": "zsh"})
    assert focus.focused_window() == FocusedWindow("kitty", "zsh")


def test_sway_finds_the_focused_node(session):
    mp, replies, _ = session
    mp.setenv("SWAYSOCK", "/run/sway.sock")
    tree = {"nodes": [{"nodes": [
        {"focused": False, "app_id": "firefox", "name": "web"},
        {"focused": True, "app_id": None, "name": "term",
         "window_properties": {"class": "Alacritty"}},
    ]}], "floating_nodes": []}
    replies[("swaymsg", "-t", "get_tree")] = json.dumps(tree)
    assert focus.focused_window() == FocusedWindow("Alacritty", "term")


def test_niri(session):
    mp, replies, _ = session
    mp.setenv("NIRI_SOCKET", "/run/niri.sock")
    replies[("niri", "msg", "-j", "focused-window")] = json.dumps(
        {"app_id": "com.mitchellh.ghostty", "title": "~"})
    assert focus.is_terminal(focus.focused_window())


def test_x11_falls_back_to_xprop_for_the_class(session):
    mp, replies, _ = session
    mp.setenv("DISPLAY", ":0")
    mp.setenv("XDG_SESSION_TYPE", "x11")
    replies[("xdotool", "getactivewindow", "getwindowname")] = "vim\n"
    replies[("xdotool", "getactivewindow")] = "4242\n"
    replies[("xprop", "-id", "4242", "WM_CLASS")] = 'WM_CLASS(STRING) = "xterm", "XTerm"\n'
    assert focus.focused_window() == FocusedWindow("XTerm", "vim")


def test_gnome_wayland_is_undetectable_without_asking_xwayland(session):
    mp, _, calls = session
    mp.setenv("WAYLAND_DISPLAY", "wayland-0")
    mp.setenv("DISPLAY", ":0")
    assert focus.focused_window() is None
    assert calls == []      # XWayland's answer is stale for native windows


def test_xwayland_title_for_context_only(session):
    mp, replies, _ = session
    mp.setenv("WAYLAND_DISPLAY", "wayland-0")
    mp.setenv("DISPLAY", ":0")
    replies[("xdotool", "getactivewindow", "getwindowname")] = "Doc - Writer\n"
    replies[("xdotool", "getactivewindow", "getwindowclassname")] = "libreoffice\n"
    from voiceio import retention
    assert retention.active_window_title() == "Doc - Writer"


def test_broken_ipc_is_none(session):
    mp, replies, _ = session
    mp.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc")
    replies[("hyprctl", "activewindow", "-j")] = "not json"
    assert focus.focused_window() is None


@pytest.mark.parametrize("app_id,expected", [
    ("kitty", True), ("org.wezfurlong.wezterm", True), ("Gnome-terminal", True),
    ("org.gnome.Console", True), ("org.gnome.Ptyxis", True), ("foot", True),
    ("URxvt", True), ("dev.warp.Warp", True), ("firefox", False),
    ("code", False), ("org.gnome.TextEditor", False),
])
def test_is_terminal(app_id, expected):
    assert focus.is_terminal(FocusedWindow(app_id)) is expected


# ── chord selection ──────────────────────────────────────────────────────

def _focus_on(monkeypatch, app_id):
    monkeypatch.setattr(focus, "focused_window",
                        lambda **k: FocusedWindow(app_id) if app_id else None)


def test_terminal_gets_ctrl_shift_v(monkeypatch):
    _focus_on(monkeypatch, "kitty")
    assert paste.PasteKeys("ctrl+v", "ctrl+shift+v").for_focus() == (["ctrl", "shift"], "v")


def test_other_apps_and_undetectable_keep_ctrl_v(monkeypatch):
    _focus_on(monkeypatch, "firefox")
    assert paste.PasteKeys("ctrl+v", "ctrl+shift+v").for_focus() == (["ctrl"], "v")
    monkeypatch.setattr(paste, "_focus_cache", (float("-inf"), False))
    _focus_on(monkeypatch, None)
    assert paste.PasteKeys("ctrl+v", "ctrl+shift+v").for_focus() == (["ctrl"], "v")


def test_explicit_paste_keys_are_never_switched(monkeypatch):
    queried = []
    monkeypatch.setattr(focus, "focused_window", lambda **k: queried.append(1))
    assert paste.PasteKeys("shift+insert", "ctrl+shift+v").for_focus() == (["shift"], "insert")
    assert paste.PasteKeys("ctrl+v", "").for_focus() == (["ctrl"], "v")
    assert queried == []


def test_focus_is_queried_once_per_burst(monkeypatch):
    queried = []
    monkeypatch.setattr(focus, "focused_window",
                        lambda **k: queried.append(1) or FocusedWindow("foot"))
    keys = paste.PasteKeys("ctrl+v", "ctrl+shift+v")
    for _ in range(5):
        keys.for_focus()
    assert len(queried) == 1


# ── IBus no-focus fallback ───────────────────────────────────────────────

def _ibus_typer(monkeypatch, tools, wayland=True, **kwargs) -> IBusTyper:
    from voiceio.typers import ibus as ibus_mod
    monkeypatch.setattr(ibus_mod.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name in tools else None)
    monkeypatch.setattr(ibus_mod, "ydotoold_running", lambda: True)
    if wayland:
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    else:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    typer = IBusTyper(**kwargs)
    monkeypatch.setattr(typer, "_send", lambda msg: None)
    monkeypatch.setattr(typer, "_copy_to_clipboard", lambda text: None)
    monkeypatch.setattr(typer, "_engine_has_focus", lambda: False)
    return typer


@pytest.mark.parametrize("tools,wayland,expected", [
    ({"ydotool"}, True, ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]),
    ({"wtype"}, True, ["wtype", "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]),
    ({"xdotool"}, False, ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"]),
])
def test_ibus_no_focus_paste_uses_paste_keys(monkeypatch, tools, wayland, expected):
    typer = _ibus_typer(monkeypatch, tools, wayland, paste_keys="ctrl+shift+v")
    with patch.object(paste.subprocess, "run") as run:
        typer.commit_text("hello")
    assert run.call_args.args[0] == expected


def test_ibus_no_focus_paste_in_a_terminal(monkeypatch):
    _focus_on(monkeypatch, "Alacritty")
    typer = _ibus_typer(monkeypatch, {"wtype"}, paste_keys="ctrl+v",
                        terminal_paste_keys="ctrl+shift+v")
    with patch.object(paste.subprocess, "run") as run:
        typer.type_text("hello")
    assert run.call_args.args[0][:5] == ["wtype", "-M", "ctrl", "-M", "shift"]


def test_ibus_focused_does_not_paste(monkeypatch):
    typer = _ibus_typer(monkeypatch, {"wtype"})
    monkeypatch.setattr(typer, "_engine_has_focus", lambda: True)
    with patch.object(paste.subprocess, "run") as run:
        typer.commit_text("hello")
    run.assert_not_called()


def test_ibus_failed_ydotool_paste_releases_modifiers(monkeypatch):
    typer = _ibus_typer(monkeypatch, {"ydotool"}, paste_keys="ctrl+shift+v")

    def run(cmd, **kwargs):
        if cmd[2].endswith(":1"):
            raise subprocess.TimeoutExpired(cmd, 3)
        return MagicMock(returncode=0)

    with patch.object(paste.subprocess, "run", side_effect=run) as mock_run:
        typer.commit_text("hello")     # logged, never raised into the session
    assert mock_run.call_args.args[0] == ["ydotool", "key", "29:0", "42:0"]


def test_options_reach_the_ibus_typer(linux_wayland_gnome):
    from voiceio import config
    from voiceio.typers import create_typer_backend, typer_options
    cfg = config.Config()
    cfg.output.paste_keys = "shift+insert"
    typer = create_typer_backend("ibus", linux_wayland_gnome, **typer_options(cfg))
    assert (typer._keys.mods, typer._keys.key) == (["shift"], "insert")
