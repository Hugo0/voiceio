"""fcitx5 text injection via the fcitx5-text-bridge addon.

Under fcitx5 (Omarchy's default, and most CJK setups) IBus can't be used, and
keystroke simulation drops characters on fast corrections. The
fcitx5-text-bridge addon (MIT, https://github.com/981213/fcitx5-text-bridge)
commits text through fcitx5's own InputContext instead. Protocol: datagrams
to $XDG_RUNTIME_DIR/fcitx5-text-bridge.sock, each [version=1][op] + payload:
op 1 = UTF-8 text to commit, op 2 = u32 little-endian count of characters to
delete before the cursor. One-way; no preedit, so streaming uses the
append-and-correct path like wtype/ydotool.
"""
from __future__ import annotations

import os
import socket
import struct
from pathlib import Path

from voiceio.backends import ProbeResult
# Same per-commit budget as IBus: fcitx5 relays a commit to the app as one
# Wayland message, and Wayland drops the client past 4096 bytes.
from voiceio.ibus.textlimit import MAX_TEXT_BYTES as _MAX_PAYLOAD

_VERSION = 1
_COMMIT, _DELETE = 1, 2


def socket_path() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) \
        / "fcitx5-text-bridge.sock"


class FcitxBridgeTyper:
    name = "fcitx-bridge"

    def __init__(self) -> None:
        self._sock: socket.socket | None = None

    def probe(self) -> ProbeResult:
        # connect() on a datagram socket fails fast if nobody is bound — a
        # stale socket file left by a crashed fcitx5 doesn't count.
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            probe.connect(str(socket_path()))
        except OSError:
            return ProbeResult(
                ok=False, reason="fcitx5-text-bridge addon not running",
                fix_hint="install https://github.com/981213/fcitx5-text-bridge and restart fcitx5",
            )
        finally:
            probe.close()
        return ProbeResult(ok=True)

    def _send(self, op: int, payload: bytes) -> None:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.sendto(bytes((_VERSION, op)) + payload, str(socket_path()))

    def type_text(self, text: str) -> None:
        data = text.encode("utf-8")
        while data:
            cut = min(len(data), _MAX_PAYLOAD)
            while cut < len(data) and (data[cut] & 0xC0) == 0x80:
                cut -= 1  # never split a UTF-8 sequence across datagrams
            self._send(_COMMIT, data[:cut])
            data = data[cut:]

    def delete_chars(self, n: int) -> None:
        if n > 0:
            self._send(_DELETE, struct.pack("<I", n))
