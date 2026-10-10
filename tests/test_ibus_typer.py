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
             patch("voiceio.ibus.install.has_ibus_gi", return_value=False):
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "bindings not available" in result.reason

    def test_probe_no_daemon(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.ibus.install.has_ibus_gi", return_value=True), \
             patch("voiceio.ibus.install.daemon_running", return_value=False):
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "daemon not running" in result.reason

    def test_probe_success(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.ibus.install.has_ibus_gi", return_value=True), \
             patch("voiceio.ibus.install.daemon_running", return_value=True), \
             patch("voiceio.ibus.install.component_installed", return_value=True), \
             patch("voiceio.ibus.install.gnome_source_configured", return_value=True):
            typer = IBusTyper()
            result = typer.probe()
            assert result.ok


class TestProbeReadOnly:
    """Fix #6: probe() must be side-effect-free (no install / IBus restart)
    and must cache a successful result to avoid subprocess storms."""

    def test_missing_component_does_not_install(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.ibus.install.has_ibus_gi", return_value=True), \
             patch("voiceio.ibus.install.daemon_running", return_value=True), \
             patch("voiceio.ibus.install.component_installed", return_value=False), \
             patch("voiceio.ibus.install.install_component") as mock_install:
            typer = IBusTyper()
            result = typer.probe()
            assert not result.ok
            assert "not installed" in result.reason
            mock_install.assert_not_called()  # read-only!

    def test_ok_result_is_cached(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.ibus.install.has_ibus_gi", return_value=True), \
             patch("voiceio.ibus.install.daemon_running", return_value=True) as mock_daemon, \
             patch("voiceio.ibus.install.component_installed", return_value=True), \
             patch("voiceio.ibus.install.gnome_source_configured", return_value=True):
            typer = IBusTyper()
            assert typer.probe().ok
            assert typer.probe().ok
            # Second call served from cache — no repeated subprocess probing.
            assert mock_daemon.call_count == 1

    def test_failure_is_not_cached(self):
        with patch("voiceio.typers.ibus.shutil.which", return_value="/usr/bin/ibus"), \
             patch("voiceio.ibus.install.has_ibus_gi", return_value=True), \
             patch("voiceio.ibus.install.daemon_running", return_value=False) as mock_daemon:
            typer = IBusTyper()
            assert not typer.probe().ok
            assert not typer.probe().ok
            # Failures re-check every time (recover immediately when fixed).
            assert mock_daemon.call_count == 2

    def test_ensure_installed_installs_when_missing(self):
        with patch("voiceio.ibus.install.component_installed", return_value=False), \
             patch("voiceio.ibus.install.install_component", return_value=True) as mock_install, \
             patch("voiceio.ibus.install.ensure_gnome_input_source") as mock_src:
            typer = IBusTyper()
            assert typer.ensure_installed() is True
            mock_install.assert_called_once()
            mock_src.assert_called_once()

    def test_ensure_installed_skips_install_when_present(self):
        with patch("voiceio.ibus.install.component_installed", return_value=True), \
             patch("voiceio.ibus.install.install_component") as mock_install, \
             patch("voiceio.ibus.install.ensure_gnome_input_source") as mock_src:
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


# ── Distro portability ───────────────────────────────────────────────────

def test_component_path_derives_system_dirs(monkeypatch):
    """NixOS keeps engines under the profile, not /usr/share; dropping them
    from IBUS_COMPONENT_PATH would hide every other input method."""
    from voiceio.ibus import install as ibus_mod
    monkeypatch.setattr(ibus_mod.shutil, "which",
                        lambda _: "/run/current-system/sw/bin/ibus")
    monkeypatch.setenv("XDG_DATA_DIRS", "/run/current-system/sw/share:/etc/profiles/per-user/u/share")
    dirs = ibus_mod._ibus_component_path().split(":")
    assert dirs[0] == str(ibus_mod.COMPONENT_DIR)
    assert "/run/current-system/sw/share/ibus/component" in dirs
    assert "/etc/profiles/per-user/u/share/ibus/component" in dirs
    assert len(dirs) == len(set(dirs))


def test_probe_reports_fcitx_instead_of_starting_ibus(monkeypatch):
    from voiceio.platform import Platform
    from voiceio.typers import ibus as ibus_mod
    monkeypatch.setattr(ibus_mod.shutil, "which", lambda _: "/usr/bin/ibus")
    monkeypatch.setattr("voiceio.ibus.install.has_ibus_gi", lambda: True)
    monkeypatch.setattr("voiceio.ibus.install.daemon_running", lambda: False)
    plat = Platform(os="linux", display_server="wayland", desktop="hyprland",
                    input_method="fcitx")
    monkeypatch.setattr("voiceio.platform.detect", lambda: plat)
    result = ibus_mod.IBusTyper()._probe_full()
    assert not result.ok
    assert "fcitx5" in result.reason
    assert not result.fix_hint


def test_stale_launcher_counts_as_not_installed(tmp_path, monkeypatch):
    """A launcher baking a dead interpreter path (moved venv, GC'd Nix store
    path) must be rewritten, not trusted because the file exists."""
    from voiceio.ibus import install as ibus_mod
    monkeypatch.setattr(ibus_mod, "LAUNCHER_PATH", tmp_path / "launcher")
    monkeypatch.setattr(ibus_mod, "COMPONENT_DEST", tmp_path / "voiceio.xml")
    monkeypatch.setattr(ibus_mod, "system_python", lambda: "/usr/bin/python3")
    ibus_mod.LAUNCHER_PATH.write_text(ibus_mod._launcher_script())
    ibus_mod.COMPONENT_DEST.write_text(ibus_mod._component_xml())
    assert ibus_mod.component_installed()
    ibus_mod.LAUNCHER_PATH.write_text(
        ibus_mod._launcher_script().replace("/usr/bin/python3", "/nix/store/gone/bin/python3"))
    assert not ibus_mod.component_installed()


def test_refresh_stale_launcher_rewrites_only_stale_paths(tmp_path, monkeypatch):
    from voiceio.ibus import install as m
    monkeypatch.setattr(m, "LAUNCHER_DIR", tmp_path)
    monkeypatch.setattr(m, "LAUNCHER_PATH", tmp_path / "voiceio-ibus-engine")
    monkeypatch.setattr(m, "COMPONENT_DEST", tmp_path / "voiceio.xml")
    assert not m.refresh_stale_launcher()            # nothing installed: no-op
    m.COMPONENT_DEST.write_text(m._component_xml())
    m.LAUNCHER_PATH.write_text('INSTALL_PY="/old/venv/bin/python3"\n')  # pre-upgrade
    assert m.refresh_stale_launcher()
    assert m.component_installed()
    assert not m.refresh_stale_launcher()            # current: no rewrite
