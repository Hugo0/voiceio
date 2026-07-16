"""Tests for the shared engine ping used by the daemon health watchdog and
the engine's own refuse-to-double-run guard (a duplicate engine binding the
socket path silently starves the healthy instance)."""
from __future__ import annotations

import socket
import threading

import voiceio.ibus as ibus_mod


def test_no_socket_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(ibus_mod, "SOCKET_PATH", tmp_path / "nope.sock")
    assert ibus_mod.ping_engine(timeout=0.2) is False


def test_healthy_peer_answers_pong(tmp_path, monkeypatch):
    sock_path = tmp_path / "engine.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(sock_path))
    server.settimeout(2.0)

    def _serve():
        try:
            data, addr = server.recvfrom(64)
            if data == b"ping" and addr:
                server.sendto(b"pong", addr)
        except OSError:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    monkeypatch.setattr(ibus_mod, "SOCKET_PATH", sock_path)
    try:
        assert ibus_mod.ping_engine(timeout=1.0) is True
    finally:
        t.join(timeout=2.0)
        server.close()


def test_unresponsive_socket_returns_false(tmp_path, monkeypatch):
    """A bound-but-silent socket (starved listener) must read as dead."""
    sock_path = tmp_path / "engine.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(sock_path))
    monkeypatch.setattr(ibus_mod, "SOCKET_PATH", sock_path)
    try:
        assert ibus_mod.ping_engine(timeout=0.2) is False
    finally:
        server.close()
