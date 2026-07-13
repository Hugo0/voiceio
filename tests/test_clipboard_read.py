"""Tests for clipboard_read module."""
from unittest.mock import patch, MagicMock

import pytest


def _mock_platform(os="linux", display_server="x11", desktop="gnome"):
    """Create a mock Platform object."""
    p = MagicMock()
    p.os = os
    p.display_server = display_server
    p.desktop = desktop
    p.is_wayland = display_server == "wayland"
    p.is_x11 = display_server == "x11"
    p.is_windows = os == "windows"
    p.is_mac = os == "darwin"
    return p


def test_read_text_linux_x11_primary():
    """Primary selection is tried first on X11."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", return_value="/usr/bin/xclip"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="selected text")
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result == "selected text"
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "primary" in args


def test_read_text_linux_x11_clipboard_fallback():
    """Falls back to clipboard if primary is empty."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", return_value="/usr/bin/xclip"), \
         patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="clipboard text"),
        ]
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result == "clipboard text"
        assert mock_run.call_count == 2


def test_read_text_returns_none_when_empty():
    """Returns None when no text available."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", return_value="/usr/bin/xclip"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result is None


def test_read_text_macos():
    """macOS uses pbpaste."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(os="darwin", display_server="quartz")), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="mac text")
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result == "mac text"
        args = mock_run.call_args[0][0]
        assert args == ["pbpaste"]


def test_read_text_wayland():
    """Wayland uses wl-paste."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="wayland")), \
         patch("shutil.which", return_value="/usr/bin/wl-paste"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="wayland text")
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result == "wayland text"
        args = mock_run.call_args[0][0]
        assert "wl-paste" in args


def test_read_text_no_tools():
    """Returns None when no clipboard tools available."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", return_value=None):
        from voiceio.clipboard_read import read_text
        result = read_text()
        assert result is None


# --- copy_text ---

def test_copy_text_wayland():
    """Wayland without xclip copies via wl-copy with text on stdin."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="wayland")), \
         patch("shutil.which", lambda t: "/usr/bin/wl-copy" if t == "wl-copy" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is True
        args = mock_run.call_args[0][0]
        assert "wl-copy" in args
        assert mock_run.call_args.kwargs["input"] == b"hello"


def test_copy_text_x11_xclip():
    """X11 copies to the CLIPBOARD selection via xclip."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", lambda t: "/usr/bin/xclip" if t == "xclip" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is True
        args = mock_run.call_args[0][0]
        assert args[:1] == ["xclip"]
        assert "clipboard" in args


def test_copy_text_no_tools():
    """Returns False (no raise) when no clipboard tools available."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="x11")), \
         patch("shutil.which", return_value=None):
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is False


def test_copy_text_empty():
    """Empty text is never copied."""
    with patch("subprocess.run") as mock_run:
        from voiceio.clipboard_read import copy_text
        assert copy_text("") is False
        mock_run.assert_not_called()


def test_copy_text_command_failure():
    """Nonzero exit from the copy tool reports False."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="wayland")), \
         patch("shutil.which", return_value="/usr/bin/wl-copy"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is False


def test_copy_text_wayland_prefers_xclip_bridge():
    """On Wayland prefer xclip (XWayland bridge): wl-copy can steal focus
    on GNOME, and a focus-out discards a visible IBus preedit."""
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="wayland")), \
         patch("shutil.which", lambda t: f"/usr/bin/{t}" if t in ("xclip", "wl-copy") else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is True
        assert mock_run.call_args[0][0][0] == "xclip"


def test_copy_text_wayland_falls_back_to_wl_copy():
    """xclip failing (no DISPLAY etc.) falls through to wl-copy."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[0])
        return MagicMock(returncode=1 if cmd[0] == "xclip" else 0)

    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(display_server="wayland")), \
         patch("shutil.which", lambda t: f"/usr/bin/{t}" if t in ("xclip", "wl-copy") else None), \
         patch("subprocess.run", side_effect=run):
        from voiceio.clipboard_read import copy_text
        assert copy_text("hello") is True
    assert calls == ["xclip", "wl-copy"]
