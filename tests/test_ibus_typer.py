"""Tests for IBus typer backend (IBus preedit + commit, clipboard backup)."""
from __future__ import annotations

import socket
import sys
import threading
from unittest.mock import patch, MagicMock

import pytest

from voiceio.typers.ibus import IBusTyper, SOCKET_PATH

_needs_unix = pytest.mark.skipif(
    sys.platform == "win32", reason="AF_UNIX not available on Windows",
)


@pytest.fixture
def mock_socket(tmp_path):
    """Create a mock IBus engine socket that records received messages."""
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX not available")
    # Use /tmp for short path - macOS has 104-char limit on AF_UNIX paths
    import tempfile
    sock_dir = tempfile.mkdtemp()
    sock_path = type(tmp_path)(sock_dir) / "vi.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(sock_path))
    sock.settimeout(2.0)

    messages = []

    def listener():
        while True:
            try:
                data, addr = sock.recvfrom(65536)
                msg = data.decode("utf-8")
                messages.append(msg)
                if msg == "ping" and addr:
                    try:
                        sock.sendto(b"pong", addr)
                    except OSError:
                        pass
            except socket.timeout:
                break
            except OSError:
                break

    thread = threading.Thread(target=listener, daemon=True)
    thread.start()

    with patch("voiceio.typers.ibus.SOCKET_PATH", sock_path):
        yield messages, sock_path

    sock.close()


class TestIBusTyper:
    def test_type_text_sends_commit(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.type_text("Hello world")
        import time; time.sleep(0.1)
        assert "commit:Hello world" in messages

    def test_type_text_empty_skipped(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.type_text("")
        import time; time.sleep(0.1)
        assert len(messages) == 0

    def test_update_preedit(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.update_preedit("Hello")
        import time; time.sleep(0.1)
        assert "preedit:Hello" in messages

    def test_commit_text_sends_commit(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.commit_text("Final text")
        import time; time.sleep(0.1)
        assert "commit:Final text" in messages

    def test_commit_empty_sends_clear(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.commit_text("")
        import time; time.sleep(0.1)
        assert "clear" in messages

    def test_clear_preedit(self, mock_socket):
        messages, _ = mock_socket
        typer = IBusTyper()
        typer.clear_preedit()
        import time; time.sleep(0.1)
        assert "clear" in messages

    def test_delete_chars_is_noop(self):
        typer = IBusTyper()
        typer.delete_chars(10)  # should not raise

    def test_commit_copies_to_clipboard(self, mock_socket):
        """commit_text also copies text to clipboard as backup."""
        typer = IBusTyper()
        typer._wl_copy = "/usr/bin/wl-copy"
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        with patch("voiceio.typers.ibus.subprocess.Popen", return_value=mock_proc):
            typer.commit_text("Hello")
            mock_proc.stdin.write.assert_called_once_with(b"Hello")
            mock_proc.stdin.close.assert_called_once()

    def test_probe_no_ibus(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value=None):
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "not installed" in result.reason

    def test_probe_no_gi(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=False):
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "bindings not available" in result.reason

    def test_probe_no_daemon(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=True), \
             patch("voiceio.typers.ibus._ibus_daemon_running", return_value=False):
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "daemon not running" in result.reason

    def test_probe_success(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=True), \
             patch("voiceio.typers.ibus._ibus_daemon_running", return_value=True), \
             patch("voiceio.typers.ibus._component_installed", return_value=True), \
             patch("voiceio.typers.ibus._gnome_source_configured", return_value=True):
            typer = IBusTyper()
            result = typer.probe()
            assert result.ok


class TestStreamingTyperProtocol:
    def test_ibus_is_streaming_typer(self):
        from voiceio.typers.base import StreamingTyper
        typer = IBusTyper()
        assert isinstance(typer, StreamingTyper)

    def test_ibus_is_typer_backend(self):
        from voiceio.typers.base import TyperBackend
        typer = IBusTyper()
        assert isinstance(typer, TyperBackend)
