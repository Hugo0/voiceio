"""The opt-in weekly install ping (voiceio/telemetry.py)."""
from __future__ import annotations

import argparse
import contextlib
import json
import threading

import pytest

from voiceio import config, report, telemetry
from voiceio.platform import Platform

SHIPPED_LIVE = telemetry.ENDPOINT_LIVE   # before any fixture patches it
DAY = 86400.0
T0 = 1_800_000_000.0


@pytest.fixture(autouse=True)
def _live(monkeypatch):
    """These tests run as if workers/ping were deployed, unless they say otherwise."""
    monkeypatch.setattr(telemetry, "ENDPOINT_LIVE", True)


@pytest.fixture(autouse=True)
def _platform(monkeypatch):
    from voiceio import platform as plat
    monkeypatch.setattr(plat, "detect", lambda: Platform(
        os="linux", display_server="wayland", desktop="gnome"))
    monkeypatch.setattr(telemetry._py_platform, "freedesktop_os_release",
                        lambda: {"ID": "ubuntu", "VERSION_ID": "26.04", "NAME": "Ubuntu",
                                 "PRETTY_NAME": "Ubuntu 26.04 LTS"})


@pytest.fixture
def sent(monkeypatch):
    """Requests that reached the network helper: (url, body)."""
    calls: list[tuple[str, dict]] = []

    def fake(url, *, timeout, json_body=None):
        calls.append((url, json.loads(json_body)))
        return contextlib.nullcontext(type("R", (), {"read": lambda self: b""})())

    monkeypatch.setattr(report, "_urlopen", fake)
    return calls


def _turn(value: bool) -> None:
    config.set_value("telemetry", "ping", value)


def test_payload_has_exactly_the_allowed_keys():
    from voiceio import __version__
    p = telemetry.payload("small")
    assert tuple(p) == telemetry.FIELDS
    assert p == {"v": __version__, "distro": "ubuntu 26.04", "desktop": "gnome",
                 "session": "wayland", "decoder": "whisper"}


@pytest.mark.parametrize("name,family", [
    ("parakeet-tdt-0.6b-v2", "parakeet"), ("parakeet-tdt-0.6b-v3", "parakeet"),
    ("small", "whisper"), ("whispercpp:http://10.0.0.5:8080", "whisper.cpp"),
    ("sherpa:/home/alice/models/parakeet-custom", "sherpa-onnx"),
])
def test_decoder_is_the_family_never_the_name(name, family):
    assert telemetry.payload(name)["decoder"] == family


def test_a_fine_tune_path_never_leaks(sent):
    from voiceio.learn.store import learn_dir
    model = str(learn_dir() / "models" / "v0003-small-20261101-0300" / "ct2")
    _turn(True)
    assert telemetry.maybe_send(model, now=T0)
    (_, body), = sent
    text = json.dumps(body)
    assert body["decoder"] == "whisper"
    for leak in (model, str(learn_dir()), "v0003", "ct2", "/"):
        assert leak not in text


def test_fields_are_cleaned_to_short_labels(monkeypatch):
    from voiceio import platform as plat
    monkeypatch.setattr(plat, "detect", lambda: Platform(
        os="linux", display_server="wayland", desktop='x-"Cinnamon"/' + "a" * 80))
    p = telemetry.payload("small")
    assert p["desktop"].startswith("x--cinnamon")
    assert len(p["desktop"]) <= telemetry.MAX_FIELD
    assert all(not set(v) - set("abcdefghijklmnopqrstuvwxyz0123456789._+ -") for v in p.values())


def test_sends_once_per_seven_days(sent):
    _turn(True)
    assert telemetry.maybe_send("small", now=T0)
    assert not telemetry.maybe_send("small", now=T0 + DAY)
    assert not telemetry.maybe_send("small", now=T0 + 6.9 * DAY)
    assert telemetry.maybe_send("small", now=T0 + 7 * DAY)
    assert [url for url, _ in sent] == [telemetry.PING_URL] * 2
    assert (config.LOG_DIR / "ping.json").exists()


def test_a_clock_set_back_does_not_block_it_for_years(sent):
    _turn(True)
    assert telemetry.maybe_send("small", now=T0)
    assert telemetry.maybe_send("small", now=T0 - 30 * DAY)


@pytest.mark.parametrize("value", [None, False])
def test_off_or_never_asked_sends_nothing(monkeypatch, value):
    monkeypatch.setattr(report, "_urlopen", lambda *a, **k: pytest.fail("sent"))
    if value is not None:
        _turn(value)
    assert telemetry.setting() is value
    assert not telemetry.maybe_send("small", now=T0)
    assert not (config.LOG_DIR / "ping.json").exists()


def test_offline_never_raises_and_retries_later():
    _turn(True)   # conftest's network block refuses the request
    assert not telemetry.maybe_send("small", now=T0)
    assert not (config.LOG_DIR / "ping.json").exists()
    assert telemetry.due(T0 + 60)


def test_a_broken_config_never_raises(monkeypatch):
    monkeypatch.setattr(config, "load", lambda *a, **k: 1 / 0)
    assert not telemetry.maybe_send("small", now=T0)


def test_daemon_thread_stops_with_the_daemon(monkeypatch, sent):
    monkeypatch.setattr(telemetry, "START_DELAY_SECS", 0)
    _turn(True)
    shutdown = threading.Event()
    checked = threading.Event()
    real = telemetry.maybe_send
    monkeypatch.setattr(telemetry, "maybe_send",
                        lambda name: (real(name), checked.set()))
    thread = telemetry.start("small", shutdown)
    assert checked.wait(5)
    shutdown.set()
    thread.join(5)
    assert not thread.is_alive()
    assert len(sent) == 1


def _tty(monkeypatch, answers: list[str]) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(report, "_interactive", lambda: True)
    monkeypatch.setattr(report, "_ask", lambda prompt: asked.append(prompt) or answers.pop(0))
    return asked


@pytest.mark.parametrize("answer,stored", [("y", True), ("yes", True), ("n", False), ("", False)])
def test_asked_once_and_the_answer_is_stored(monkeypatch, answer, stored):
    asked = _tty(monkeypatch, [answer])
    telemetry.offer()
    assert telemetry.setting() is stored
    assert asked == ["\n  " + telemetry.QUESTION]
    telemetry.offer()   # answered: never again
    assert len(asked) == 1


def test_never_asked_without_a_terminal(monkeypatch):
    monkeypatch.setattr(report, "_interactive", lambda: False)
    monkeypatch.setattr(report, "_ask", lambda prompt: pytest.fail("asked"))
    telemetry.offer()
    assert telemetry.setting() is None


def test_setup_and_doctor_ask_the_same_question(monkeypatch):
    """Doctor asks an install that was never asked; it calls the same offer()."""
    import inspect

    from voiceio.cli import doctor
    from voiceio.setup import steps
    assert "telemetry.offer()" in inspect.getsource(doctor._cmd_doctor)
    assert "telemetry.offer()" in inspect.getsource(steps._step_finalize)
    from voiceio.setup import noninteractive
    assert "telemetry" not in inspect.getsource(noninteractive)


def test_question_names_every_field_sent():
    q = telemetry.QUESTION.lower()
    for word in ("version", "distro", "desktop", "session", "speech engine"):
        assert word in q
    assert q.endswith("[y/n] ")


def test_doctor_line(sent):
    assert telemetry.status_line("small") == "Weekly anonymous count: not asked"
    _turn(False)
    assert telemetry.status_line("small") == "Weekly anonymous count: off"
    _turn(True)
    line = telemetry.status_line("small")
    assert line.startswith("Weekly anonymous count: on\n  sends: {")
    assert json.loads(line.split("sends: ", 1)[1]) == telemetry.payload("small")


def test_config_set_turns_it_off(capsys):
    from voiceio.cli import configcmd
    _turn(True)
    configcmd.cmd_set(argparse.Namespace(key="telemetry.ping", value="false"))
    assert telemetry.setting() is False
    configcmd.cmd_set(argparse.Namespace(key="telemetry.ping", value="true"))
    assert telemetry.setting() is True


def test_shipped_with_the_endpoint_off():
    """Flip ENDPOINT_LIVE only once workers/ping is deployed."""
    assert SHIPPED_LIVE is False


def test_before_the_endpoint_is_live_nothing_is_asked_or_sent(monkeypatch):
    monkeypatch.setattr(telemetry, "ENDPOINT_LIVE", False)
    monkeypatch.setattr(report, "_interactive", lambda: True)
    monkeypatch.setattr(report, "_ask", lambda prompt: pytest.fail("asked"))
    monkeypatch.setattr(report, "_urlopen", lambda *a, **k: pytest.fail("sent"))
    telemetry.offer()
    assert telemetry.setting() is None
    _turn(True)   # even when set by hand
    assert not telemetry.maybe_send("small", now=T0)
    assert telemetry.start("small", threading.Event()) is None
    assert telemetry.status_line("small") == ""
    assert not (config.LOG_DIR / "ping.json").exists()
