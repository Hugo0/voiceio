"""The last error in a report, and the service's repeated-error nudge."""
from __future__ import annotations

import datetime as dt
import logging

from voiceio import errorlog

NOW = dt.datetime(2026, 11, 5, 15, 0, 0)
SECRET = "please wire the money to my sister on Friday"
PKG = errorlog._PKG


def _log(tmp_path, text):
    path = tmp_path / "voiceio.log"
    path.write_text(text)
    return path


SERVICE_LOG = f"""\
2026-11-05 12:58:10,961 voiceio INFO Recording started
2026-11-05 12:58:11,000 voiceio.streaming ERROR Streaming failed on: {SECRET}
Traceback (most recent call last):
  File "{PKG}/core/decoders/sherpa.py", line 280, in decode
    stream = self._rec.create_stream(hotwords)
AttributeError: 'NoneType' object has no attribute 'create_stream'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "{PKG}/streaming.py", line 488, in _transcribe_and_apply
    result = self._decoder.decode(tail)
  File "/usr/lib/python3/dist-packages/numpy/core.py", line 9, in f
    pass
voiceio.core.decoders.base.DecodeError: decode failed: {SECRET}
Word
{SECRET}
2026-11-05 12:58:12,000 voiceio INFO voiceio stopped
"""


def test_last_error_names_only_type_frame_and_age(tmp_path):
    got = errorlog.last_error(_log(tmp_path, SERVICE_LOG), tmp_path / "none", now=NOW)
    assert got == "DecodeError at voiceio/streaming.py:488 (2 h ago)"


def test_dictated_text_in_a_log_never_reaches_the_report(tmp_path):
    from voiceio.health import BackendStatus, HealthReport
    from voiceio.platform import Platform
    from voiceio.report import build_report
    from voiceio.config import Config

    err = errorlog.last_error(_log(tmp_path, SERVICE_LOG), tmp_path / "none", now=NOW)
    health = HealthReport(
        platform=Platform(os="linux", display_server="wayland", desktop="gnome"),
        hotkey_backends=[BackendStatus("evdev", True)],
        typer_backends=[BackendStatus("ibus", True)], audio_ok=True)
    text = build_report(health, last_error=err, cfg=Config())
    assert "last error: DecodeError at voiceio/streaming.py:488" in text
    for word in ("money", "sister", "Friday", "Word", "NoneType", "create_stream"):
        assert word not in text


def test_an_error_without_traceback_names_its_logger(tmp_path):
    log = f"2026-11-05 14:30:00,000 voiceio.recorder ERROR mic gone: {SECRET}\n"
    got = errorlog.last_error(_log(tmp_path, log), tmp_path / "none", now=NOW)
    assert got == "logged by voiceio.recorder (30 min ago)"


def test_errors_older_than_a_week_are_left_out(tmp_path):
    old = SERVICE_LOG.replace("2026-11-05 12:58:11", "2026-10-20 12:58:11")
    assert errorlog.last_error(_log(tmp_path, old), tmp_path / "none", now=NOW) == ""


def test_crash_log_keeps_the_frame_not_the_argv(tmp_path):
    crash = tmp_path / "crash.log"
    crash.write_text(f"""
===== 2026-11-04T14:00:00.123456 =====
argv: ['voiceio', 'say', '{SECRET}']
Traceback (most recent call last):
  File "/home/someone/src/voiceio/voiceio/cli/__init__.py", line 205, in _entry_point
    main()
  File "/home/someone/.venv/lib/python3.13/site-packages/numpy/__init__.py", line 1, in <module>
    x
ImportError: cannot import {SECRET}
""")
    got = errorlog.last_error(tmp_path / "none", crash, now=NOW)
    assert got == "ImportError at voiceio/cli/__init__.py:205 (1 d ago)"


def test_the_newest_of_service_and_crash_errors_wins(tmp_path):
    crash = tmp_path / "crash.log"
    crash.write_text("===== 2026-11-05T14:59:00 =====\nargv: []\n"
                     "Traceback (most recent call last):\n"
                     f'  File "{PKG}/app.py", line 7, in run\n    x\nKeyError: 1\n')
    got = errorlog.last_error(_log(tmp_path, SERVICE_LOG), crash, now=NOW)
    assert got == "KeyError at voiceio/app.py:7 (1 min ago)"


def _record(msg, *args):
    return logging.LogRecord("voiceio.streaming", logging.ERROR, __file__, 1, msg, args, None)


def test_the_same_error_three_times_notifies_once_a_day(tmp_path):
    sent = []
    h = errorlog.RepeatedErrorNotifier(tmp_path / "state.json",
                                       notify=lambda title, body: sent.append(body))
    for n in range(2):
        h.emit(_record("decode failed: %s", f"words {n}"))
    h.emit(_record("something else"))
    assert sent == []
    h.emit(_record("decode failed: %s", "more words"))
    assert sent == ["Run 'voiceio doctor --report' to tell us."]
    # A restarted daemon reads the same state: no second nudge today.
    h2 = errorlog.RepeatedErrorNotifier(tmp_path / "state.json",
                                        notify=lambda title, body: sent.append(body))
    for _ in range(3):
        h2.emit(_record("decode failed: %s", "x"))
    assert len(sent) == 1


def test_doctor_offers_a_report_only_after_a_failure_in_a_terminal(monkeypatch):
    import argparse

    from voiceio import report
    from voiceio.cli.doctor import _offer_report

    asked = []
    monkeypatch.setattr(report, "_ask", lambda prompt: asked.append(prompt) or "y")
    monkeypatch.setattr(report, "_interactive", lambda: True)
    args = argparse.Namespace(yes=False)
    assert not _offer_report(args, healthy=True)
    assert not _offer_report(argparse.Namespace(yes=True), healthy=False)
    assert _offer_report(args, healthy=False) and "Something isn't working" in asked[0]
    monkeypatch.setattr(report, "_interactive", lambda: False)
    assert not _offer_report(args, healthy=False)
