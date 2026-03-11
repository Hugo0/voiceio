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
    with patch("voiceio.clipboard_read.detect", return_value=_mock_platform(os="macos", display_server="quartz")), \
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
