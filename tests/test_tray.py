"""Tray backend probing: what doctor/setup promise must match what start() does."""
from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from voiceio import tray
from voiceio.tray import _pystray

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="linux probe")


@linux
def test_appindicator_wins():
    with patch.object(tray, "_find_system_python", return_value="/usr/bin/python3"):
        assert tray.probe_availability() == (True, "AppIndicator3 available", "")


@linux
def test_pystray_xembed_fallback_counts_as_a_tray():
    # KDE/LXQt users without python3-gi got "no tray" though start() falls back to pystray.
    with patch.object(tray, "_find_system_python", return_value=None), \
         patch.object(tray, "_pystray_backend", return_value="xorg"):
        ok, reason, hint = tray.probe_availability()
    assert ok and "no menu" in reason and "for the full menu" in hint


@linux
def test_no_backend_at_all():
    with patch.object(tray, "_find_system_python", return_value=None), \
         patch.object(tray, "_pystray_backend", return_value=None):
        ok, reason, hint = tray.probe_availability()
    assert not ok and "AppIndicator3" in reason and hint.startswith("install:")


def _import_raising(exc):
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "pystray":
            raise exc
        return real(name, *a, **kw)
    return fake


class _DisplayNameError(Exception):
    """Stands in for Xlib.error.DisplayNameError (no DISPLAY): not an ImportError."""


def test_pystray_without_x_display_does_not_raise(monkeypatch):
    monkeypatch.delitem(sys.modules, "pystray", raising=False)
    with patch("builtins.__import__", _import_raising(_DisplayNameError(":0"))):
        assert tray._pystray_backend() is None
        assert _pystray.start(lambda: None, Path("idle.png"), []) is False
