"""`voiceio doctor --report` and the post-setup feedback line: what goes public."""
from __future__ import annotations

import io
import json
import sys

import pytest

from voiceio import report
from voiceio.config import Config
from voiceio.health import BackendStatus, HealthReport
from voiceio.platform import Platform

FAKE_HOME = "/home/zelda-tester"
FAKE_USER = "zelda-tester"
FAKE_HOST = "zeldas-thinkpad"


@pytest.fixture
def identity(monkeypatch):
    """A fake user, home and hostname that must never reach a report."""
    monkeypatch.setenv("HOME", FAKE_HOME)
    monkeypatch.setenv("USER", FAKE_USER)
    monkeypatch.setattr(report.getpass, "getuser", lambda: FAKE_USER)
    monkeypatch.setattr(report.socket, "gethostname", lambda: FAKE_HOST)
    monkeypatch.setattr(report._py_platform, "node", lambda: FAKE_HOST)
    monkeypatch.setattr(report, "_homes", lambda: [FAKE_HOME])


def _health(**kw) -> HealthReport:
    defaults = dict(
        platform=Platform(os="linux", display_server="wayland", desktop="gnome"),
        hotkey_backends=[BackendStatus("evdev", True)],
        typer_backends=[BackendStatus("ibus", True), BackendStatus("ydotool", False, "not installed")],
        audio_ok=True, cli_in_path=True, tray_ok=True,
    )
    defaults.update(kw)
    return HealthReport(**defaults)


def _assert_private(text: str) -> None:
    for secret in (FAKE_HOME, FAKE_USER, FAKE_HOST, "zelda"):
        assert secret not in text.lower(), (secret, text)


# ── scrubbing ──

def test_scrub_removes_home_user_host_and_addresses(identity):
    raw = (f"cannot open {FAKE_HOME}/.config/voiceio/x.toml as {FAKE_USER}@{FAKE_HOST}; "
           "also /home/someoneelse/.cache, 192.168.1.40, fe80:0:0:0:1:2:3:4, "
           "aa:bb:cc:dd:ee:ff, /dev/input/by-id/usb-Logitech_K120_SN12345-event-kbd, "
           "ZELDA-TESTER shouted")
    out = report.scrub(raw)
    _assert_private(out)
    for leaked in ("someoneelse", "192.168", "fe80", "aa:bb", "SN12345", "Logitech"):
        assert leaked not in out
    assert "~/.config/voiceio/x.toml" in out


def test_scrub_keeps_versions_and_backend_names(identity):
    out = report.scrub("voiceio 1.10.0.2 · ibus · python 3.12.3")
    assert out == "voiceio 1.10.0.2 · ibus · python 3.12.3"


def test_report_scrubs_check_messages_and_comment(identity):
    health = _health(
        hotkey_backends=[BackendStatus("evdev", False, f"{FAKE_HOME}/dev denied for {FAKE_USER}")],
        audio_ok=False, audio_reason=f"device on {FAKE_HOST} busy",
        warnings=[f"see {FAKE_HOME}/.local/state/voiceio/voiceio.log"],
    )
    text = report.build_report(health, comment=f"broke in {FAKE_HOME}/src, I'm {FAKE_USER}",
                               cfg=Config())
    _assert_private(text)
    assert "failed: hotkey evdev: ~/dev denied" in text
    assert "audio: device on <host> busy" in text
    assert "comment: broke in ~/src" in text


def test_report_contains_only_backend_and_model_names(identity):
    cfg = Config()
    cfg.model.name = f"{FAKE_HOME}/models/my-tune"
    cfg.model.vocabulary_file = f"{FAKE_HOME}/vocab.txt"
    cfg.cloud.api_key = "sk-secret"
    cfg.hotkey.key = "ctrl+alt+zelda"
    text = report.build_report(_health(), cfg=cfg)
    _assert_private(text)
    assert "sk-secret" not in text and "vocab" not in text and "ctrl+alt" not in text
    assert "model: custom (local path)" in text
    assert "hotkey evdev" in text and "typer ibus" in text
    assert "fallbacks unavailable: ydotool" in text
    assert "failed: none" in text
    assert text.splitlines()[0].startswith("voiceio ")


def test_report_lists_failures_and_warnings():
    health = _health(typer_backends=[BackendStatus("ibus", False, "daemon down")],
                     cli_in_path=False, tray_ok=False, tray_reason="no AppIndicator")
    text = report.build_report(health, missing_deps=["portaudio"], perm_issues=2, cfg=Config())
    assert "typer none" in text
    assert "typer ibus: daemon down" in text
    assert "missing system deps" in text
    assert "cli not in PATH" in text and "tray: no AppIndicator" in text
    assert "insecure permissions on 2 file(s)" in text


def test_comment_is_one_bounded_line():
    text = report.build_report(_health(), comment="a\nb\tc " + "x" * 1000, cfg=Config())
    line = next(ln for ln in text.splitlines() if ln.startswith("comment:"))
    assert line.startswith("comment: a b c x")
    assert len(line) <= len("comment: ") + report.MAX_COMMENT


# ── posting ──

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def outside_pytest(monkeypatch):
    """post() refuses under pytest; these tests check what it does past that.
    (pytest sets PYTEST_CURRENT_TEST again after fixtures, so patch the check.)"""
    monkeypatch.setattr(report, "_test_artifact", lambda text: "")


def test_post_refuses_test_runs_and_mock_text(monkeypatch):
    monkeypatch.setattr(report, "_urlopen", lambda url, timeout: pytest.fail("posted"))
    ok, why = report.post("voiceio 1.13.0 report")
    assert not ok and "pytest" in why
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    ok, why = report.post("system: Ubuntu · <MagicMock name='mock.platform.desktop'>")
    assert not ok and "mock" in why


def test_post_success_reads_receipt(monkeypatch, outside_pytest):
    seen = []

    def fake(url, timeout):
        seen.append((url, timeout))
        return _Resp(json.dumps({"ok": True, "receipt": {"id": "r123"}}).encode())

    monkeypatch.setattr(report, "_urlopen", fake)
    assert report.post("hello world") == (True, "r123")
    url, timeout = seen[0]
    assert url.startswith(report.POST_URL + "?")
    assert "text=hello+world" in url and "format=json" in url and "request_id=voiceio-" in url
    assert timeout <= 10


@pytest.mark.parametrize("body", [b'{"ok": false, "error": "rate limited"}', b"[]", b"not json",
                                  b'{"ok": true}'])
def test_post_bad_response_is_failure(monkeypatch, outside_pytest, body):
    monkeypatch.setattr(report, "_urlopen", lambda url, timeout: _Resp(body))
    ok, why = report.post("x")
    assert not ok and why


def test_post_network_error_never_raises(outside_pytest):
    # conftest refuses every connection: this is the offline path.
    ok, why = report.post("x")
    assert not ok and "ConnectionRefused" in why


# ── confirm / --yes / no TTY ──

@pytest.fixture
def posted(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(report, "post", lambda text: (calls.append(text), (True, "r1"))[1])
    return calls


def _tty(monkeypatch, tty: bool):
    monkeypatch.setattr(report, "_interactive", lambda: tty)


def test_share_asks_and_posts_on_yes(monkeypatch, capsys, posted):
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert report.share("the report") is True
    out = capsys.readouterr().out
    assert "PUBLICLY" in out and "the report" in out and report.BOARD_URL in out
    assert posted == ["the report"]


@pytest.mark.parametrize("answer", ["", "n", "no", "whatever"])
def test_share_defaults_to_no(monkeypatch, capsys, posted, answer):
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert report.share("r") is False
    assert posted == []


def test_share_eof_is_no(monkeypatch, posted):
    _tty(monkeypatch, True)

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert report.share("r") is False and posted == []


def test_share_without_tty_prints_but_never_posts(monkeypatch, capsys, posted):
    _tty(monkeypatch, False)
    monkeypatch.setattr("builtins.input", lambda p: pytest.fail("must not prompt"))
    assert report.share("the report") is False
    out = capsys.readouterr().out
    assert "the report" in out and "--yes" in out
    assert posted == []


def test_share_yes_skips_prompt(monkeypatch, posted):
    _tty(monkeypatch, False)
    monkeypatch.setattr("builtins.input", lambda p: pytest.fail("must not prompt"))
    assert report.share("r", yes=True) is True and posted == ["r"]


def test_share_failure_is_reported_not_raised(monkeypatch, capsys):
    assert report.share("r", yes=True) is False
    assert "Could not confirm the post" in capsys.readouterr().out


# ── CLI wiring ──

@pytest.fixture
def doctor_env(monkeypatch, identity):
    from voiceio import config, health
    from voiceio import platform as plat
    monkeypatch.setattr(health, "check_health", lambda: _health(
        warnings=[f"something at {FAKE_HOME}/.config"]))
    monkeypatch.setattr(health, "format_report", lambda r: "DOCTOR OUTPUT")
    monkeypatch.setattr(health, "usage_findings", lambda: [])
    monkeypatch.setattr(plat, "check_system_deps", lambda: [])
    monkeypatch.setattr(config, "check_permissions", lambda: [])
    monkeypatch.setattr("voiceio.hints.hint", lambda *a, **k: None)


def _doctor(monkeypatch, capsys, *argv):
    from voiceio.cli import main
    monkeypatch.setattr(sys, "argv", ["voiceio", "doctor", *argv])
    with pytest.raises(SystemExit):
        main()
    return capsys.readouterr().out


def test_doctor_report_yes_posts_scrubbed_comment(monkeypatch, capsys, doctor_env, posted):
    out = _doctor(monkeypatch, capsys, "--report", "--yes",
                  "--comment", f"worked, {FAKE_USER} here")
    assert len(posted) == 1
    _assert_private(posted[0])
    _assert_private(out)
    assert "comment: worked, <user> here" in posted[0]
    assert report.BOARD_URL in out


def test_doctor_report_without_tty_or_yes_does_not_post(monkeypatch, capsys, doctor_env, posted):
    _tty(monkeypatch, False)
    out = _doctor(monkeypatch, capsys, "--report")
    assert posted == []
    assert "voiceio " in out and "Not posted" in out


def test_doctor_without_report_never_posts(monkeypatch, capsys, doctor_env, posted):
    out = _doctor(monkeypatch, capsys)
    assert posted == [] and "PUBLICLY" not in out


# ── after setup ──

def test_after_setup_enter_skips_silently(monkeypatch, capsys, posted):
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda p: "")
    monkeypatch.setattr(report, "gather", lambda c: pytest.fail("no checks on skip"))
    report.offer_after_setup()
    assert posted == [] and "PUBLICLY" not in capsys.readouterr().out


def test_after_setup_comment_is_shown_confirmed_and_posted(monkeypatch, capsys, posted):
    _tty(monkeypatch, True)
    answers = iter(["went fine", "y"])
    monkeypatch.setattr("builtins.input", lambda p: next(answers))
    monkeypatch.setattr(report, "gather", lambda c: (None, f"voiceio x report\ncomment: {c}"))
    report.offer_after_setup()
    assert posted == ["voiceio x report\ncomment: went fine"]


def test_after_setup_never_raises(monkeypatch, capsys):
    _tty(monkeypatch, True)
    monkeypatch.setattr("builtins.input", lambda p: "it broke")

    def boom(c):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(report, "gather", boom)
    report.offer_after_setup()  # must not raise
    assert "Skipped" in capsys.readouterr().out


def test_after_setup_needs_a_terminal(monkeypatch, posted):
    _tty(monkeypatch, False)
    monkeypatch.setattr("builtins.input", lambda p: pytest.fail("must not prompt"))
    report.offer_after_setup()
    assert posted == []


def test_setup_wizard_offers_the_report(monkeypatch):
    from voiceio.cli import main
    calls = []
    monkeypatch.setattr("voiceio.setup.run_wizard", lambda: calls.append("wizard"))
    monkeypatch.setattr(report, "offer_after_setup", lambda: calls.append("offer"))
    monkeypatch.setattr("voiceio.hints.hint", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["voiceio", "setup"])
    main()
    assert calls == ["wizard", "offer"]


def test_network_guard_blocks_real_hosts():
    import socket
    s = socket.socket()
    try:
        with pytest.raises(ConnectionRefusedError):
            s.connect(("93.184.216.34", 443))
    finally:
        s.close()
