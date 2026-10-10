"""Modifier guard: keystroke typers wait for the user to let go of modifiers."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from voiceio.typers import modifier_guard
from voiceio.typers.base import KeystrokeTyper

CTRL, SHIFT, A, Z, V = 29, 42, 30, 44, 47
# conftest stubs this out for every test; these ones exercise the real thing.
_open_keyboards = modifier_guard._open_keyboards
EV_KEY = 1


@pytest.fixture
def fake_evdev(monkeypatch):
    """A python-evdev stand-in: only the names the guard touches."""
    ecodes = types.SimpleNamespace(
        EV_KEY=EV_KEY, KEY_A=A, KEY_Z=Z,
        KEY_LEFTCTRL=CTRL, KEY_RIGHTCTRL=97, KEY_LEFTSHIFT=SHIFT, KEY_RIGHTSHIFT=54,
        KEY_LEFTALT=56, KEY_RIGHTALT=100, KEY_LEFTMETA=125, KEY_RIGHTMETA=126,
    )
    mod = types.ModuleType("evdev")
    mod.ecodes = ecodes
    monkeypatch.setitem(sys.modules, "evdev", mod)
    monkeypatch.setitem(sys.modules, "evdev.ecodes", ecodes)
    return mod


class FakeKeyboard:
    def __init__(self, *frames):
        """Each active_keys() call returns the next frame; the last repeats."""
        self.frames = list(frames) or [[]]
        self.closed = False

    def capabilities(self):
        return {EV_KEY: [A, Z, V]}

    def active_keys(self):
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

    def close(self):
        self.closed = True


def _guard_with(monkeypatch, *keyboards):
    monkeypatch.setattr(modifier_guard, "_open_keyboards", lambda: list(keyboards))


def test_no_keyboards_returns_immediately(monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(modifier_guard.time, "sleep", sleep)
    modifier_guard.wait_for_release()
    sleep.assert_not_called()


def test_waits_until_modifiers_released(monkeypatch, fake_evdev):
    kb = FakeKeyboard([CTRL, V], [CTRL], [])
    _guard_with(monkeypatch, kb)
    sleep = MagicMock()
    monkeypatch.setattr(modifier_guard.time, "sleep", sleep)
    modifier_guard.wait_for_release()
    assert sleep.call_count == 2


def test_letter_keys_alone_do_not_block(monkeypatch, fake_evdev):
    _guard_with(monkeypatch, FakeKeyboard([V]))
    sleep = MagicMock()
    monkeypatch.setattr(modifier_guard.time, "sleep", sleep)
    modifier_guard.wait_for_release()
    sleep.assert_not_called()


def test_any_keyboard_holding_a_modifier_blocks(monkeypatch, fake_evdev):
    _guard_with(monkeypatch, FakeKeyboard([]), FakeKeyboard([SHIFT], []))
    sleep = MagicMock()
    monkeypatch.setattr(modifier_guard.time, "sleep", sleep)
    modifier_guard.wait_for_release()
    assert sleep.call_count == 1


def test_gives_up_after_timeout(monkeypatch, fake_evdev):
    """A stuck modifier delays typing, never swallows it."""
    kb = FakeKeyboard([CTRL])
    _guard_with(monkeypatch, kb)
    clock = iter(x * 0.1 for x in range(100))
    monkeypatch.setattr(modifier_guard.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(modifier_guard.time, "sleep", lambda s: None)
    modifier_guard.wait_for_release(timeout=0.5)


def test_unplugged_keyboard_is_ignored(monkeypatch, fake_evdev):
    kb = FakeKeyboard()
    kb.active_keys = MagicMock(side_effect=OSError("No such device"))
    _guard_with(monkeypatch, kb)
    modifier_guard.wait_for_release()


def test_open_keyboards_without_evdev(monkeypatch):
    monkeypatch.setitem(sys.modules, "evdev", None)
    assert _open_keyboards() == []


@pytest.fixture
def device_cache(monkeypatch):
    monkeypatch.setattr(modifier_guard, "_keyboards", {})
    monkeypatch.setattr(modifier_guard, "_not_keyboards", set())


def test_open_keyboards_keeps_only_readable_keyboards(fake_evdev, device_cache):
    keyboard, mouse = FakeKeyboard(), FakeKeyboard()
    mouse.capabilities = lambda: {EV_KEY: [272]}   # BTN_LEFT only
    devices = {"/dev/input/event0": keyboard, "/dev/input/event1": mouse}

    def open_device(path):
        if path == "/dev/input/event2":
            raise PermissionError(path)
        return devices[path]

    fake_evdev.list_devices = MagicMock(return_value=[*devices, "/dev/input/event2"])
    fake_evdev.InputDevice = open_device
    with patch.object(modifier_guard.threading, "Thread") as thread:
        assert _open_keyboards() == [keyboard]
    fake_evdev.list_devices.assert_called_once_with(writable=False)
    thread.assert_called_once_with(target=mouse.close, daemon=True)
    assert not keyboard.closed


def test_open_keyboards_reuses_devices_and_follows_hotplug(fake_evdev, device_cache):
    """Closing an evdev fd is slow, so devices stay open; unplugged ones go."""
    first, second = FakeKeyboard(), FakeKeyboard()
    opened = []

    def open_device(path):
        opened.append(path)
        return {"/dev/input/event0": first, "/dev/input/event1": second}[path]

    fake_evdev.InputDevice = open_device
    fake_evdev.list_devices = MagicMock(return_value=["/dev/input/event0"])
    assert _open_keyboards() == [first]
    fake_evdev.list_devices.return_value = ["/dev/input/event0", "/dev/input/event1"]
    assert _open_keyboards() == [first, second]
    assert opened == ["/dev/input/event0", "/dev/input/event1"]
    fake_evdev.list_devices.return_value = ["/dev/input/event1"]
    assert _open_keyboards() == [second]
    assert first.closed and not second.closed


class _Recorder(KeystrokeTyper):
    name = "fake"

    def __init__(self):
        self.calls = []

    def _type(self, text):
        self.calls.append(("type", text))

    def _delete(self, n):
        self.calls.append(("delete", n))


def test_keystroke_typer_waits_before_typing_and_deleting():
    typer = _Recorder()
    order = []
    with patch.object(modifier_guard, "wait_for_release",
                      side_effect=lambda: order.append("wait")):
        typer.type_text("hi")
        typer.delete_chars(2)
    assert order == ["wait", "wait"]
    assert typer.calls == [("type", "hi"), ("delete", 2)]


def test_keystroke_typer_skips_empty_work():
    typer = _Recorder()
    with patch.object(modifier_guard, "wait_for_release") as wait:
        typer.type_text("")
        typer.delete_chars(0)
    wait.assert_not_called()
    assert typer.calls == []


@pytest.mark.parametrize("module,cls", [
    ("voiceio.typers.ydotool", "YdotoolTyper"),
    ("voiceio.typers.wtype", "WtypeTyper"),
    ("voiceio.typers.xdotool", "XdotoolTyper"),
    ("voiceio.typers.pynput_type", "PynputTyper"),
    ("voiceio.typers.clipboard", "ClipboardTyper"),
])
def test_keystroke_backends_are_guarded(module, cls):
    import importlib
    assert issubclass(getattr(importlib.import_module(module), cls), KeystrokeTyper)


@pytest.mark.parametrize("module,cls", [
    ("voiceio.typers.ibus", "IBusTyper"),
    ("voiceio.typers.fcitx_bridge", "FcitxBridgeTyper"),
])
def test_text_committing_backends_are_not_guarded(module, cls):
    import importlib
    assert not issubclass(getattr(importlib.import_module(module), cls), KeystrokeTyper)
