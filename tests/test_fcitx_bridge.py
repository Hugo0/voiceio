import socket
import struct

import pytest

from voiceio.typers import fcitx_bridge
from voiceio.typers.base import TyperBackend


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(str(fcitx_bridge.socket_path()))
    srv.settimeout(2)
    yield srv
    srv.close()


def test_probe_needs_the_addon_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert not fcitx_bridge.FcitxBridgeTyper().probe().ok


def test_commit_and_delete_wire_format(bridge):
    t = fcitx_bridge.FcitxBridgeTyper()
    assert isinstance(t, TyperBackend) and t.probe().ok
    t.type_text("héllo")
    assert bridge.recv(100) == b"\x01\x01" + "héllo".encode()
    t.delete_chars(3)
    assert bridge.recv(100) == b"\x01\x02" + struct.pack("<I", 3)


def test_long_text_is_split_on_utf8_boundaries(bridge, monkeypatch):
    monkeypatch.setattr(fcitx_bridge, "_MAX_PAYLOAD", 5)
    fcitx_bridge.FcitxBridgeTyper().type_text("aaaéé")  # é = 2 bytes
    parts = [bridge.recv(100)[2:] for _ in range(2)]
    assert b"".join(parts).decode() == "aaaéé"
    assert all(p.decode() for p in parts)  # each part is valid UTF-8
