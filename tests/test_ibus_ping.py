"""Tests for the shared engine ping used by the daemon health watchdog and
the engine's own refuse-to-double-run guard (a duplicate engine binding the
socket path silently starves the healthy instance)."""
from __future__ import annotations

import os
import socket
import sys
import threading

import pytest

import voiceio.ibus as ibus_mod

# IBus (and AF_UNIX DGRAM + Linux abstract autobind) is Linux-only.
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="IBus is Linux-only")


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


class TestSingletonLock:
    """The atomic gate that stops a duplicate engine from stealing the socket.

    ibus-daemon exec-spawns its own engine copy when the voiceio source is
    activated; only the flock holder may bind the command socket, so the loser
    exits instead of unlink+rebind-ing it into a zombie.
    """

    def test_second_acquire_fails_while_held(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ibus_mod, "LOCK_PATH", tmp_path / "e.lock")
        fd = ibus_mod.acquire_singleton_lock()
        assert fd is not None
        try:
            # A concurrent engine (same process here, standing in for a second
            # process on the same file) must be refused.
            assert ibus_mod.acquire_singleton_lock() is None
        finally:
            os.close(fd)

    def test_reacquire_succeeds_after_holder_closes(self, tmp_path, monkeypatch):
        """Releasing the fd (process death) frees the lock for the next engine."""
        monkeypatch.setattr(ibus_mod, "LOCK_PATH", tmp_path / "e.lock")
        fd = ibus_mod.acquire_singleton_lock()
        assert fd is not None
        os.close(fd)  # simulate the holder dying
        fd2 = ibus_mod.acquire_singleton_lock()
        assert fd2 is not None
        os.close(fd2)

    def test_lock_is_cross_process(self, tmp_path, monkeypatch):
        """flock must exclude a genuinely separate process, not just this one."""
        import subprocess
        import textwrap

        lock_path = tmp_path / "e.lock"
        monkeypatch.setattr(ibus_mod, "LOCK_PATH", lock_path)
        fd = ibus_mod.acquire_singleton_lock()
        assert fd is not None
        try:
            # A child process using the same flock path must be refused while we
            # hold it. Exit code 7 = "got the lock" (bad), 0 = "refused" (good).
            code = textwrap.dedent(f"""
                import fcntl, os, sys
                fd = os.open({str(lock_path)!r}, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    sys.exit(7)
                except OSError:
                    sys.exit(0)
            """)
            r = subprocess.run([sys.executable, "-c", code], timeout=5)
            assert r.returncode == 0
        finally:
            os.close(fd)
