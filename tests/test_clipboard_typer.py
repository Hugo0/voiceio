"""Clipboard typer: configurable paste keystroke and clipboard restore."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from voiceio import config
from voiceio.typers import clipboard as clip_mod, typer_options
from voiceio.typers.clipboard import ClipboardTyper
from voiceio.typers.paste import parse_paste_keys, paste_command


@pytest.mark.parametrize("spec,expected", [
    ("ctrl+v", (["ctrl"], "v")),
    ("Ctrl + Shift + V", (["ctrl", "shift"], "v")),
    ("shift+insert", (["shift"], "insert")),
])
def test_parse_paste_keys(spec, expected):
    assert parse_paste_keys(spec) == expected


@pytest.mark.parametrize("spec", ["ctrl+q", "hyper+v", "ctrl+ctrl+v", ""])
def test_unknown_paste_keys_fall_back_to_ctrl_v(spec, caplog):
    assert parse_paste_keys(spec) == (["ctrl"], "v")
    assert "paste_keys" in caplog.text


def test_paste_command_per_tool():
    mods, key = ["ctrl", "shift"], "v"
    assert paste_command("ydotool", mods, key) == \
        ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]
    assert paste_command("wtype", mods, key) == \
        ["wtype", "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]
    assert paste_command("xdotool", mods, key) == \
        ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"]
    assert paste_command("xdotool", ["shift"], "insert") == \
        ["xdotool", "key", "--clearmodifiers", "shift+Insert"]


def _wayland_typer(monkeypatch, tool="ydotool", **kwargs) -> ClipboardTyper:
    monkeypatch.setattr(clip_mod.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    available = {"wl-copy", tool}
    monkeypatch.setattr(clip_mod.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name in available else None)
    return ClipboardTyper(**kwargs)


@pytest.mark.parametrize("tool,expected", [
    ("ydotool", ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]),
    ("wtype", ["wtype", "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]),
])
def test_configured_keys_are_what_gets_pressed(monkeypatch, tool, expected):
    typer = _wayland_typer(monkeypatch, tool, paste_keys="ctrl+shift+v")
    with patch.object(clip_mod.subprocess, "run") as run:
        typer.type_text("hello")
    assert run.call_args_list[0].args[0] == ["wl-copy", "--"]
    assert run.call_args_list[1].args[0] == expected


def test_default_paste_is_ctrl_v(monkeypatch):
    typer = _wayland_typer(monkeypatch)
    with patch.object(clip_mod.subprocess, "run") as run:
        typer.type_text("hello")
    assert run.call_args_list[1].args[0] == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


def test_failed_paste_releases_the_configured_modifiers(monkeypatch):
    typer = _wayland_typer(monkeypatch, paste_keys="ctrl+shift+v")

    def run(cmd, **kwargs):
        if cmd[:2] == ["ydotool", "key"] and cmd[2].endswith(":1"):
            raise subprocess.TimeoutExpired(cmd, 3)
        return MagicMock(returncode=0)

    with patch.object(clip_mod.subprocess, "run", side_effect=run) as mock_run, \
         pytest.raises(subprocess.TimeoutExpired):
        typer.type_text("hello")
    assert mock_run.call_args.args[0] == ["ydotool", "key", "29:0", "42:0"]


class FakeClipboard:
    def __init__(self, text):
        self.text = text
        self.reads = 0
        self.writes = []

    def read(self):
        self.reads += 1
        return self.text

    def copy(self, text):
        self.writes.append(text)
        self.text = text
        return True


@pytest.fixture
def clipboard(monkeypatch):
    board = FakeClipboard("user's own text\n")
    monkeypatch.setattr(clip_mod.clipboard_read, "read_clipboard", board.read)
    monkeypatch.setattr(clip_mod.clipboard_read, "copy_text", board.copy)
    monkeypatch.setattr(clip_mod, "RESTORE_DELAY_SECS", 0.01)
    return board


def _copying_run(board):
    """subprocess.run stand-in: wl-copy puts its input on the fake board."""
    def run(cmd, input=None, **kwargs):
        if cmd[0] == "wl-copy":
            board.text = input.decode()
        return MagicMock(returncode=0)
    return run


def _settle(typer):
    """Wait for a scheduled restore (None: it already ran)."""
    timer = typer._restore_timer
    if timer is not None:
        timer.join(timeout=2)


def _paste(typer, board, *texts):
    """Paste each text, then let the restore timer run."""
    with patch.object(clip_mod.subprocess, "run", side_effect=_copying_run(board)):
        for text in texts:
            typer.type_text(text)
    _settle(typer)


def test_restores_the_users_clipboard_after_pasting(monkeypatch, clipboard):
    typer = _wayland_typer(monkeypatch, restore_clipboard=True)
    _paste(typer, clipboard, "dictated")
    assert clipboard.writes == ["user's own text\n"]


def test_streaming_burst_saves_once_and_restores_once(monkeypatch, clipboard):
    typer = _wayland_typer(monkeypatch, restore_clipboard=True)
    monkeypatch.setattr(clip_mod, "RESTORE_DELAY_SECS", 5)
    with patch.object(clip_mod.subprocess, "run", side_effect=_copying_run(clipboard)):
        typer.type_text("one ")
        typer.type_text("two")
    assert clipboard.reads == 1           # saved before the first paste only
    typer._restore_timer.cancel()
    typer._restore_saved(typer._restore_seq)
    assert clipboard.writes == ["user's own text\n"]


def test_superseded_restore_does_nothing(monkeypatch, clipboard):
    typer = _wayland_typer(monkeypatch, restore_clipboard=True)
    typer._saved, typer._pasted, typer._restore_seq = "old", "x", 2
    clipboard.text = "x"
    typer._restore_saved(1)
    assert clipboard.writes == []


def test_no_restore_over_something_copied_since(monkeypatch, clipboard):
    typer = _wayland_typer(monkeypatch, restore_clipboard=True)

    def run(cmd, input=None, **kwargs):
        clipboard.text = "copied by the user meanwhile"
        return MagicMock(returncode=0)

    with patch.object(clip_mod.subprocess, "run", side_effect=run):
        typer.type_text("dictated")
    _settle(typer)
    assert clipboard.writes == []


def test_empty_clipboard_is_left_with_the_transcript(monkeypatch, clipboard):
    clipboard.text = None
    typer = _wayland_typer(monkeypatch, restore_clipboard=True)
    _paste(typer, clipboard, "dictated")
    assert clipboard.writes == []


def test_restore_disabled_never_reads_the_clipboard(monkeypatch, clipboard):
    typer = _wayland_typer(monkeypatch)
    with patch.object(clip_mod.subprocess, "run"):
        typer.type_text("dictated")
    assert clipboard.reads == 0
    assert typer._restore_timer is None


@pytest.mark.parametrize("copy,notify,expected", [
    ("off", False, True),
    ("final", False, False),     # the transcript is meant to stay on the clipboard
    ("live", False, False),
    ("off", True, False),        # the notification says "copied to clipboard"
])
def test_restore_only_when_nothing_keeps_the_transcript(copy, notify, expected):
    cfg = config.Config()
    cfg.output.copy_to_clipboard = copy
    cfg.feedback.notify_clipboard = notify
    assert typer_options(cfg)["restore_clipboard"] is expected


def test_options_reach_the_clipboard_typer(linux_wayland_gnome):
    from voiceio.typers import create_typer_backend
    typer = create_typer_backend("clipboard", linux_wayland_gnome,
                                 paste_keys="shift+insert", restore_clipboard=True)
    assert (typer._paste_mods, typer._paste_key) == (["shift"], "insert")
    assert typer._restore_clipboard is True


def test_terminal_focus_switches_the_default_chord(monkeypatch):
    from voiceio import focus
    monkeypatch.setattr(focus, "focused_window", lambda **k: focus.FocusedWindow("kitty"))
    typer = _wayland_typer(monkeypatch, terminal_paste_keys="ctrl+shift+v")
    with patch.object(clip_mod.subprocess, "run") as run:
        typer.type_text("hello")
    assert run.call_args_list[1].args[0] == \
        ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]
