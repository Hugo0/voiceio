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
        """commit_text also copies text to clipboard as backup (via the
        focus-safe clipboard_read.copy_text, never wl-copy directly)."""
        typer = IBusTyper()
        with patch("voiceio.clipboard_read.copy_text", return_value=True) as mock_copy, \
             patch("voiceio.typers.ibus.subprocess.run"):
            typer.commit_text("Hello")
            mock_copy.assert_called_once_with("Hello")

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


class TestProbeReadOnly:
    """Fix #6: probe() must be side-effect-free (no install / IBus restart)
    and must cache a successful result to avoid subprocess storms."""

    def test_missing_component_does_not_install(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=True), \
             patch("voiceio.typers.ibus._ibus_daemon_running", return_value=True), \
             patch("voiceio.typers.ibus._component_installed", return_value=False), \
             patch("voiceio.typers.ibus.install_component") as mock_install:
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "not installed" in result.reason
            mock_install.assert_not_called()  # read-only!

    def test_ok_result_is_cached(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=True), \
             patch("voiceio.typers.ibus._ibus_daemon_running", return_value=True) as mock_daemon, \
             patch("voiceio.typers.ibus._component_installed", return_value=True), \
             patch("voiceio.typers.ibus._gnome_source_configured", return_value=True):
            typer = IBusTyper()
            assert typer.probe().ok
            assert typer.probe().ok
            # Second call served from cache — no repeated subprocess probing.
            assert mock_daemon.call_count == 1

    def test_failure_is_not_cached(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.typers.ibus._has_ibus_gi", return_value=True), \
             patch("voiceio.typers.ibus._ibus_daemon_running", return_value=False) as mock_daemon:
            typer = IBusTyper()
            assert not typer.probe().ok
            assert not typer.probe().ok
            # Failures re-check every time (recover immediately when fixed).
            assert mock_daemon.call_count == 2

    def test_ensure_installed_installs_when_missing(self):
        with patch("voiceio.typers.ibus._component_installed", return_value=False), \
             patch("voiceio.typers.ibus.install_component", return_value=True) as mock_install, \
             patch("voiceio.typers.ibus._ensure_gnome_input_source") as mock_src:
            typer = IBusTyper()
            assert typer.ensure_installed() is True
            mock_install.assert_called_once()
            mock_src.assert_called_once()

    def test_ensure_installed_skips_install_when_present(self):
        with patch("voiceio.typers.ibus._component_installed", return_value=True), \
             patch("voiceio.typers.ibus.install_component") as mock_install, \
             patch("voiceio.typers.ibus._ensure_gnome_input_source") as mock_src:
            typer = IBusTyper()
            assert typer.ensure_installed() is True
            mock_install.assert_not_called()
            mock_src.assert_called_once()


class TestStreamingTyperProtocol:
    def test_ibus_is_streaming_typer(self):
        from voiceio.typers.base import StreamingTyper
        typer = IBusTyper()
        assert isinstance(typer, StreamingTyper)

    def test_ibus_is_typer_backend(self):
        from voiceio.typers.base import TyperBackend
        typer = IBusTyper()
        assert isinstance(typer, TyperBackend)
