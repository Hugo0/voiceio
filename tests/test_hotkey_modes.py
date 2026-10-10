"""Hotkey modes: toggle, push-to-talk (hold) and auto (tap toggles, hold talks)."""
from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from voiceio.app import _State
from voiceio.config import Config
from voiceio.typers.base import TyperBackend


def _make_app(mode="auto", tail_ms=0, hold_ms=350):
    cfg = Config()
    cfg.hotkey.mode = mode
    cfg.hotkey.release_tail_ms = tail_ms
    cfg.hotkey.hold_ms = hold_ms
    hotkey = MagicMock()
    hotkey.name = "evdev"
    typer = MagicMock(spec=TyperBackend)
    typer.name = "clipboard"
    with patch("voiceio.app.hotkey_chain.select", return_value=hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        from voiceio.app import VoiceIO
        vio = VoiceIO(cfg)
    vio.stops = []

    def start():
        vio._state = _State.RECORDING
        vio._recording += 1
        vio._record_start = time.monotonic()

    def stop():
        vio.stops.append(vio._recording)
        vio._generation += 1
        vio._state = _State.FINALIZING

    vio._do_start = start
    vio._do_stop = stop
    return vio


def _held_for(vio, secs):
    """Pretend the key went down `secs` ago."""
    pressed_at, rid = vio._held
    vio._held = (pressed_at - secs, rid)


def _after_debounce(vio):
    vio._last_hotkey -= 1.0


def _wait_stopped(vio, n=1, timeout=2.0):
    deadline = time.monotonic() + timeout
    while len(vio.stops) < n and time.monotonic() < deadline:
        time.sleep(0.005)
    return len(vio.stops) >= n


def test_auto_tap_toggles():
    vio = _make_app("auto")
    vio.on_hotkey_press()
    vio.on_hotkey_release()                  # quick tap: latched
    time.sleep(0.05)
    assert vio._state == _State.RECORDING and vio.stops == []
    _after_debounce(vio)
    vio.on_hotkey_press()                    # next tap stops
    assert vio.stops == [1]
    vio.on_hotkey_release()                  # ...and its release does nothing
    assert vio.stops == [1]


def test_auto_hold_is_push_to_talk():
    vio = _make_app("auto")
    vio.on_hotkey_press()
    _held_for(vio, 0.5)
    vio.on_hotkey_release()
    assert _wait_stopped(vio)
    assert vio.stops == [1]


def test_hold_mode_stops_on_any_release():
    vio = _make_app("hold")
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    assert _wait_stopped(vio)


def test_toggle_mode_ignores_release():
    vio = _make_app("toggle")
    vio.on_hotkey_press()
    assert vio._held is None
    vio.on_hotkey_release()
    time.sleep(0.05)
    assert vio._state == _State.RECORDING


def test_unknown_mode_falls_back_to_toggle():
    assert _make_app("sometimes")._hotkey_mode == "toggle"


def test_release_tail_keeps_recording_briefly():
    vio = _make_app("hold", tail_ms=150)
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    time.sleep(0.05)
    assert vio._state == _State.RECORDING    # still capturing the last word
    assert _wait_stopped(vio)


def test_press_during_tail_stops_once():
    vio = _make_app("hold", tail_ms=100)
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    _after_debounce(vio)
    vio.on_hotkey_press()                    # stops now
    time.sleep(0.2)                          # tail timer fires: stale
    assert vio.stops == [1]


def test_release_after_auto_stop_does_nothing():
    vio = _make_app("hold")
    vio.on_hotkey_press()
    vio._state = _State.FINALIZING           # silence auto-stop meanwhile
    vio.on_hotkey_release()
    time.sleep(0.05)
    assert vio.stops == []


def test_release_never_stops_a_newer_recording():
    vio = _make_app("hold", tail_ms=100)
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    vio._do_stop()                           # recording 1 ends some other way
    vio._do_start()                          # recording 2 starts within the tail
    time.sleep(0.2)
    assert vio.stops == [1]
    assert vio._state == _State.RECORDING


def test_release_while_start_is_slow_is_still_a_tap():
    """The release waits on the hotkey lock while the start runs (stream
    reopen can take 0.4s): the tap's length is when the key came up."""
    vio = _make_app("auto", hold_ms=350)
    started = threading.Event()
    fast_start = vio._do_start

    def slow_start():
        started.set()
        time.sleep(0.5)
        fast_start()

    vio._do_start = slow_start
    press = threading.Thread(target=vio.on_hotkey_press)
    press.start()
    started.wait(1)
    time.sleep(0.05)
    vio.on_hotkey_release()                  # blocks until the start is done
    press.join(2)
    time.sleep(0.05)
    assert vio._state == _State.RECORDING and vio.stops == []


def test_release_of_the_press_the_socket_already_handled():
    """DE shortcut (socket toggle) and evdev both see one keypress: the
    debounced evdev press still arms push-to-talk."""
    vio = _make_app("auto")
    vio.on_hotkey()                          # socket: starts
    vio.on_hotkey_press()                    # evdev: debounced, arms
    _held_for(vio, 1.0)
    vio.on_hotkey_release()
    assert _wait_stopped(vio)


def test_press_that_stops_a_toggled_recording_is_not_armed():
    vio = _make_app("hold")
    vio.on_hotkey()                          # tray / voiceio toggle
    _after_debounce(vio)
    vio.on_hotkey_press()                    # stops it
    assert vio._held is None and vio.stops == [1]


def test_default_mode_is_auto():
    assert Config().hotkey.mode == "auto"


# ── backends ─────────────────────────────────────────────────────────────

def test_evdev_combo_edges():
    from voiceio.hotkeys.evdev import _Combo
    CTRL, ALT, V, X = 29, 56, 47, 45
    combo = _Combo(V, [{CTRL}, {ALT}])
    assert combo.feed(V, 1, 0.0) is None              # no modifiers
    assert combo.feed(V, 0, 0.1) is None              # release of a press that never fired
    combo.feed(CTRL, 1, 1.0)
    combo.feed(ALT, 1, 1.0)
    assert combo.feed(V, 1, 1.1) == "press"
    assert combo.feed(V, 2, 1.5) is None              # autorepeat
    combo.feed(CTRL, 0, 1.6)                          # modifiers up first
    assert combo.feed(X, 0, 1.7) is None
    assert combo.feed(V, 0, 1.8) == "release"
    combo.feed(CTRL, 1, 1.82)
    assert combo.feed(V, 1, 1.85) is None             # debounced
    assert combo.feed(V, 0, 1.9) is None


@pytest.fixture
def pynput_handlers(monkeypatch):
    pytest.importorskip("pynput")
    from pynput import keyboard

    from voiceio.hotkeys import pynput_backend
    monkeypatch.setattr(pynput_backend, "_AUTOREPEAT_SECS", 0.02)
    captured = {}

    class FakeListener:
        def __init__(self, on_press, on_release):
            captured["press"], captured["release"] = on_press, on_release

        def start(self):
            pass

    monkeypatch.setattr(keyboard, "Listener", FakeListener)
    events = []
    pynput_backend.PynputHotkey().start(
        "ctrl+v", lambda: events.append("press"), lambda: events.append("release"))
    v = keyboard.KeyCode.from_char("v")
    return captured, events, keyboard.Key.ctrl, v


def test_pynput_press_and_release(pynput_handlers):
    h, events, ctrl, v = pynput_handlers
    h["press"](ctrl)
    h["press"](v)
    h["press"](v)                            # repeated press while held
    h["release"](v)
    time.sleep(0.1)
    assert events == ["press", "release"]


def test_pynput_autorepeat_pairs_are_not_a_release(pynput_handlers):
    h, events, ctrl, v = pynput_handlers
    h["press"](ctrl)
    h["press"](v)
    for _ in range(3):                       # X11 autorepeat: release+press
        h["release"](v)
        h["press"](v)
    time.sleep(0.1)
    assert events == ["press"]
    h["release"](v)
    time.sleep(0.1)
    assert events == ["press", "release"]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket")
def test_socket_press_release_commands(monkeypatch):
    import tempfile
    from pathlib import Path

    from voiceio.hotkeys import socket_backend
    path = Path(tempfile.mkdtemp()) / "v.sock"
    monkeypatch.setattr(socket_backend, "SOCKET_PATH", path)
    got = []
    done = threading.Event()
    hk = socket_backend.SocketHotkey()
    hk.start("ctrl+alt+v", lambda: got.append("toggle"))
    hk.handle("press", lambda: got.append("press"))
    hk.handle("release", lambda: (got.append("release"), done.set()))
    try:
        for command in ("toggle", "press", "release", "bogus"):
            assert socket_backend.send_command(command)
        done.wait(2)
        assert got == ["toggle", "release"]  # press: same keypress as the toggle
    finally:
        hk.stop()


@pytest.mark.parametrize("mode,with_release", [("auto", True), ("hold", True), ("toggle", False)])
def test_key_backend_gets_releases_unless_toggling(mode, with_release):
    vio = _make_app(mode)
    vio._socket = MagicMock()
    vio._start_hotkeys()
    args = vio._hotkey.start.call_args.args
    if with_release:
        assert args[1:] == (vio.on_hotkey_press, vio.on_hotkey_release)
    else:
        assert args[1:] == (vio.on_hotkey,)
    vio._socket.start.assert_called_once_with(vio.cfg.hotkey.key, vio.on_hotkey)
    vio._socket.handle.assert_any_call("release", vio.on_hotkey_release)


def test_socket_only_setups_still_toggle():
    vio = _make_app("auto")
    vio._hotkey.name = "socket"
    vio._socket = None
    vio._start_hotkeys()
    vio._hotkey.start.assert_called_once_with(vio.cfg.hotkey.key, vio.on_hotkey)
