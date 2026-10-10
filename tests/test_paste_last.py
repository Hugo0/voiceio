"""voiceio paste-last / copy-last: the last dictation, again."""
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from voiceio import history
from voiceio.app import _State
from voiceio.config import Config, HistoryConfig
from voiceio.typers.base import TyperBackend


def _dictate(*texts):
    for text in texts:
        history.append(text, cfg=HistoryConfig())


def test_last_text():
    assert history.last_text() is None
    _dictate("first one", "second one")
    assert history.last_text() == "second one"


def _make_app(typer_name="wtype"):
    hotkey = MagicMock()
    hotkey.name = "socket"
    typer = MagicMock(spec=TyperBackend)
    typer.name = typer_name
    with patch("voiceio.app.hotkey_chain.select", return_value=hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        from voiceio.app import VoiceIO
        vio = VoiceIO(Config())
    vio._typers = MagicMock()
    vio._typers.typer.name = typer_name
    vio.typed = threading.Event()
    return vio


def test_paste_last_types_through_the_active_typer():
    _dictate("hello again")
    vio = _make_app("wtype")
    vio._typers.type_with_fallback.side_effect = lambda text: vio.typed.set()
    vio.paste_last()
    assert vio.typed.wait(5)
    vio._typers.type_with_fallback.assert_called_once_with("hello again")


def test_paste_last_with_ibus_pastes_from_the_clipboard():
    """Idle, voiceio's IBus engine holds no input source to commit through."""
    _dictate("hello again")
    vio = _make_app("ibus")
    clip = MagicMock()
    clip.type_text.side_effect = lambda text: vio.typed.set()
    vio._typers.clipboard_fallback.return_value = clip
    vio.paste_last()
    assert vio.typed.wait(5)
    clip.type_text.assert_called_once_with("hello again")
    vio._typers.type_with_fallback.assert_not_called()


@pytest.mark.parametrize("state", [_State.RECORDING, _State.FINALIZING])
def test_paste_last_waits_out_a_dictation(state):
    _dictate("hello again")
    vio = _make_app()
    vio._state = state
    with patch.object(vio, "_output_thread") as spawn:
        vio.paste_last()
    spawn.assert_not_called()


def test_paste_last_without_history_does_nothing():
    vio = _make_app()
    with patch.object(vio, "_output_thread") as spawn:
        vio.paste_last()
    spawn.assert_not_called()


def test_paste_last_is_a_control_socket_command():
    vio = _make_app()
    vio._start_hotkeys()
    vio._hotkey.handle.assert_any_call("paste-last", vio.paste_last)


def _cli(monkeypatch, *argv):
    from voiceio import cli
    monkeypatch.setattr(sys, "argv", ["voiceio", *argv])
    cli.main()


def test_cli_paste_last_asks_the_daemon(monkeypatch):
    _dictate("hello again")
    sent = []
    monkeypatch.setattr("voiceio.hotkeys.socket_backend.send_command",
                        lambda c: sent.append(c) or True)
    _cli(monkeypatch, "paste-last")
    assert sent == ["paste-last"]


def test_cli_copy_last(monkeypatch, capsys):
    _dictate("hello again")
    copied = []
    monkeypatch.setattr("voiceio.clipboard_read.copy_text", lambda t: copied.append(t) or True)
    _cli(monkeypatch, "copy-last")
    assert copied == ["hello again"]
    assert "2 words" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["paste-last", "copy-last"])
def test_cli_without_history_fails(monkeypatch, capsys, command):
    sent = []
    monkeypatch.setattr("voiceio.hotkeys.socket_backend.send_command",
                        lambda c: sent.append(c) or True)
    with pytest.raises(SystemExit) as exc:
        _cli(monkeypatch, command)
    assert exc.value.code == 1 and sent == []
    assert "No dictation" in capsys.readouterr().err
