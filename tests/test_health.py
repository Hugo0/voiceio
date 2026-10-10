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


def test_usage_latency_judges_the_current_release(monkeypatch):
    import json

    from voiceio import __version__, config
    from voiceio.health import usage_findings
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"text": "x", "latency": {"finalize_total": 20.0}, "v": "0.9.11"}] * 30 \
        + [{"text": "x", "latency": {"finalize_total": 2.0}, "v": __version__}] * 12
    config.HISTORY_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    finding = next(f for ok, f, _ in usage_findings(config.Config()) if f.startswith("stop→text"))
    assert "median 2.0s" in finding and f"on {__version__}" in finding


def test_usage_decoder_counts_incidents_on_the_current_release(tmp_path, monkeypatch):
    import time

    from voiceio import __version__, config
    from voiceio.health import usage_findings
    day = time.strftime("%Y-%m-%d")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    (config.LOG_DIR / "voiceio.log.1").write_text(
        f"{day} 01:00 voiceio INFO === voiceio v0.9.11 startup ===\n"
        f"{day} 01:01 voiceio.transcriber WARNING Transcription timed out after 30s, killing worker\n"
        f"{day} 01:01 voiceio.transcriber WARNING Worker died, restarting (attempt 1/3)\n")
    (config.LOG_DIR / "voiceio.log").write_text(
        f"{day} 02:00 voiceio INFO === voiceio v{__version__} startup ===\n"
        f"{day} 02:05 voiceio.transcriber WARNING Worker died, restarting (attempt 1/3)\n")
    ok, finding = next((ok, f) for ok, f, _ in usage_findings(config.Config())
                       if f.startswith("decoder"))
    assert ok
    assert finding.startswith(f"decoder: 0 hangs, 1 crashes in the last 7 days on {__version__}")
    assert f"(1 more before {__version__})" in finding
