"""One socket listener per daemon: `voiceio toggle` dictates, `voiceio speak`
reads aloud. A second SocketHotkey for the TTS key used to steal the socket
path, so the toggle shortcut read the clipboard aloud instead of recording."""
from __future__ import annotations

import socket
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from voiceio.config import Config
from voiceio.hotkeys import socket_backend as sb
from voiceio.typers.base import TyperBackend

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix socket path")


@pytest.fixture
def sock_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "SOCKET_PATH", tmp_path / "v.sock")
    return tmp_path / "v.sock"


def _make_app(tts_engine):
    """The hotkey chain resolves to the socket backend (no evdev, no pynput)."""
    typer = MagicMock(spec=TyperBackend)
    typer.name = "clipboard"
    with patch("voiceio.app.hotkey_chain.select", side_effect=lambda *a, **k: sb.SocketHotkey()), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.tts.select", return_value=tts_engine), \
         patch("voiceio.tts.player.TTSPlayer", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        from voiceio.app import VoiceIO
        vio = VoiceIO(Config())
    return vio


def _start(vio):
    """What run() does with the hotkeys."""
    vio._start_hotkeys()
    if vio._tts_hotkey:
        vio._tts_hotkey.start(vio.cfg.tts.hotkey, vio.on_tts_hotkey)


def _stop(vio):
    vio._hotkey.stop()
    if vio._socket is not None:
        vio._socket.stop()
    if vio._tts_hotkey:
        vio._tts_hotkey.stop()


def _wait_for(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.02)
    return cond()


def test_toggle_dictates_when_tts_is_on_and_the_chain_is_socket(sock_path):
    engine = MagicMock()
    engine.name = "espeak"
    vio = _make_app(engine)
    assert vio._hotkey.name == "socket"
    assert vio._socket is None and vio._tts_hotkey is None

    vio.on_hotkey = MagicMock()
    vio.on_tts_hotkey = MagicMock()
    binds = []
    real_start = sb.SocketHotkey.start
    with patch.object(sb.SocketHotkey, "start", autospec=True,
                      side_effect=lambda self, *a, **k: (binds.append(self),
                                                         real_start(self, *a, **k))):
        _start(vio)
    try:
        assert len(binds) == 1
        assert sb.send_command("toggle")
        assert _wait_for(lambda: vio.on_hotkey.called)
        time.sleep(0.1)
        vio.on_tts_hotkey.assert_not_called()

        assert sb.send_command("speak")
        assert _wait_for(lambda: vio.on_tts_hotkey.called)
        vio.on_hotkey.assert_called_once()
    finally:
        _stop(vio)


def test_speak_is_ignored_without_a_tts_engine(sock_path):
    vio = _make_app(None)
    vio.on_hotkey = MagicMock()
    vio.on_tts_hotkey = MagicMock()
    _start(vio)
    try:
        assert sb.send_command("speak")
        assert sb.send_command("toggle")
        assert _wait_for(lambda: vio.on_hotkey.called)
        vio.on_tts_hotkey.assert_not_called()
    finally:
        _stop(vio)


def test_non_socket_tts_backend_still_gets_its_own_key():
    key_backend = MagicMock()
    key_backend.name = "evdev"
    typer = MagicMock(spec=TyperBackend)
    typer.name = "clipboard"
    engine = MagicMock()
    engine.name = "espeak"
    with patch("voiceio.app.hotkey_chain.select", return_value=key_backend), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.tts.select", return_value=engine), \
         patch("voiceio.tts.player.TTSPlayer", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        from voiceio.app import VoiceIO
        vio = VoiceIO(Config())
    assert vio._tts_hotkey is key_backend
    vio._socket = MagicMock()
    vio._start_hotkeys()
    vio._socket.handle.assert_any_call("speak", vio.on_tts_hotkey)


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.parametrize("udp", [False, True], ids=["unix", "windows-udp"])
def test_debounce_is_per_command(sock_path, monkeypatch, udp):
    if udp:
        monkeypatch.setattr(sb, "_IS_WINDOWS", True)
        monkeypatch.setattr(sb, "_UDP_PORT", _free_udp_port())
    got = []
    done = threading.Event()
    hk = sb.SocketHotkey()
    hk.start("x", lambda: got.append("toggle"))
    hk.handle("speak", lambda: got.append("speak"))
    hk.handle("cancel", lambda: done.set())
    try:
        for command in ("toggle", "speak", "toggle", "speak", "cancel"):
            assert sb.send_command(command)
        assert done.wait(3)
        # A toggle doesn't suppress a speak; repeats of each are debounced.
        assert got == ["toggle", "speak"]
    finally:
        hk.stop()
