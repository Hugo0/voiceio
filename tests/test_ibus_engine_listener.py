"""The engine's command socket must stay up from bind until shutdown.

The listener used to loop `while mainloop.is_running()`, but it starts before
`mainloop.run()`: on a loaded CPU it saw "not running", exited and unlinked
its own socket — an engine alive but deaf, reaped as a zombie in a loop.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def engine_module(monkeypatch):
    repository = types.ModuleType("gi.repository")
    repository.IBus = types.SimpleNamespace(Engine=type("Engine", (), {}),
                                            Factory=type("Factory", (), {}))
    repository.GLib = types.SimpleNamespace(idle_add=lambda fn, *a: None)
    repository.GObject = types.SimpleNamespace()
    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    gi.repository = repository
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    sys.modules.pop("voiceio.ibus.engine", None)
    module = importlib.import_module("voiceio.ibus.engine")
    yield module
    sys.modules.pop("voiceio.ibus.engine", None)


def _ping(path: Path) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        client.bind(os.path.join(d, "c"))
        client.settimeout(2)
        client.sendto(b"ping", str(path))
        reply = client.recv(16)
        client.close()
        return reply


def test_listener_serves_before_the_mainloop_runs(engine_module, monkeypatch, tmp_path):
    sock_path = tmp_path / "engine.sock"
    monkeypatch.setattr(engine_module, "SOCKET_PATH", sock_path)
    not_running_yet = types.SimpleNamespace(is_running=lambda: False, quit=lambda: None)
    stopping = threading.Event()
    t = threading.Thread(target=engine_module._socket_listener,
                         args=(not_running_yet, stopping), daemon=True)
    t.start()
    for _ in range(100):
        if sock_path.exists():
            break
        threading.Event().wait(0.02)
    assert _ping(sock_path) == b"pong"
    assert t.is_alive() and sock_path.exists()

    stopping.set()
    t.join(timeout=3)
    assert not t.is_alive()
    assert not sock_path.exists()  # its own socket, so it cleans up


def test_bind_failure_stops_the_engine(engine_module, monkeypatch, tmp_path):
    quits = []
    engine_module.GLib.idle_add = lambda fn, *a: quits.append(fn)
    monkeypatch.setattr(engine_module, "SOCKET_PATH", tmp_path / "missing-dir" / "s.sock")
    loop = types.SimpleNamespace(is_running=lambda: True, quit=lambda: None)
    engine_module._socket_listener(loop, threading.Event())   # returns, doesn't hang
    assert quits == [loop.quit]
