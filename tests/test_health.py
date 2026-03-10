"""Tests for health report generation."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from voiceio.health import check_health, format_report
from voiceio.backends import ProbeResult
from voiceio.platform import Platform


def test_health_report_structure():
    p = Platform(os="linux", display_server="wayland", desktop="gnome",
                 has_ydotool=True, has_input_group=True)

    def mock_hotkey_resolve(platform):
        return [("evdev", MagicMock(), ProbeResult(ok=True))]

    def mock_typer_resolve(platform):
        return [("ydotool", MagicMock(), ProbeResult(ok=True))]

    with patch("voiceio.health.hotkey_chain.resolve", mock_hotkey_resolve), \
         patch("voiceio.health.typer_chain.resolve", mock_typer_resolve), \
         patch.dict("sys.modules", {"sounddevice": MagicMock()}):
        report = check_health(p)

    assert report.platform == p
    assert len(report.hotkey_backends) == 1
    assert report.hotkey_backends[0].ok is True
    assert len(report.typer_backends) == 1


def test_health_report_format():
    p = Platform(os="linux", display_server="wayland", desktop="gnome")

    def mock_hotkey_resolve(platform):
        return [
            ("evdev", MagicMock(), ProbeResult(ok=True)),
            ("socket", MagicMock(), ProbeResult(ok=True)),
        ]

    def mock_typer_resolve(platform):
        return [
            ("ydotool", MagicMock(), ProbeResult(ok=False, reason="not installed", fix_hint="sudo apt install ydotool")),
            ("clipboard", MagicMock(), ProbeResult(ok=True)),
        ]

    with patch("voiceio.health.hotkey_chain.resolve", mock_hotkey_resolve), \
         patch("voiceio.health.typer_chain.resolve", mock_typer_resolve), \
         patch.dict("sys.modules", {"sounddevice": MagicMock()}):
        report = check_health(p)

    text = format_report(report)
    assert "wayland" in text
    assert "✓" in text and "evdev" in text
    assert "✗" in text and "ydotool" in text
    assert "install ydotool" in text
    assert "clipboard" in text
    assert "◀ active" in text


def test_all_ok_property():
    p = Platform(os="linux", display_server="x11", desktop="unknown")
    from voiceio.health import HealthReport, BackendStatus
    report = HealthReport(
        platform=p,
        hotkey_backends=[BackendStatus(name="pynput", ok=True)],
        typer_backends=[BackendStatus(name="xdotool", ok=True)],
        audio_ok=True,
    )
    assert report.all_ok is True

    report.audio_ok = False
    assert report.all_ok is False
