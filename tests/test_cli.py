"""The v1 command-line surface."""
from __future__ import annotations

import sys

import pytest

from voiceio import config
from voiceio.cli import main


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["voiceio", *argv])
    main()
    return capsys.readouterr().out


def test_bare_voiceio_prints_help(monkeypatch, capsys):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    out = run(monkeypatch, capsys)
    for cmd in ("run", "toggle", "setup", "doctor", "service", "models", "vocab",
                "corrections", "learn", "history", "config", "uninstall"):
        assert cmd in out


def test_bare_voiceio_under_systemd_still_runs_daemon(monkeypatch, capsys):
    """Pre-1.0 units run bare `voiceio`; they must keep working."""
    called = []
    monkeypatch.setenv("INVOCATION_ID", "abc")
    monkeypatch.setattr("voiceio.cli._cmd_run", lambda args: called.append(args))
    run(monkeypatch, capsys)
    assert called and called[0].model is None


@pytest.mark.parametrize("removed", ["correct", "test", "update", "eval", "logs"])
def test_removed_commands_are_gone(monkeypatch, capsys, removed):
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, removed)


def test_vocab_add_show_remove(monkeypatch, capsys):
    run(monkeypatch, capsys, "vocab", "add", "Kalshi", "HyperNEAT")
    assert "Kalshi" in run(monkeypatch, capsys, "vocab", "show")
    run(monkeypatch, capsys, "vocab", "remove", "kalshi")
    out = run(monkeypatch, capsys, "vocab", "show")
    assert "Kalshi" not in out and "HyperNEAT" in out


def test_corrections_add_list_remove(monkeypatch, capsys):
    run(monkeypatch, capsys, "corrections", "add", "kalchi", "Kalshi")
    assert "kalchi → Kalshi" in run(monkeypatch, capsys, "corrections", "list")
    run(monkeypatch, capsys, "corrections", "remove", "kalchi")
    assert "No corrections" in run(monkeypatch, capsys, "corrections")


def test_config_show_redacts_secrets_and_set_writes(monkeypatch, capsys):
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text('[autocorrect]\napi_key = "sk-secret"\n')
    out = run(monkeypatch, capsys, "config", "show")
    assert "sk-secret" not in out and "•••" in out
    run(monkeypatch, capsys, "config", "set", "output.streaming_freeze_secs", "8")
    assert config.load().output.streaming_freeze_secs == 8
    assert run(monkeypatch, capsys, "config", "get", "output.streaming_freeze_secs").strip() == "8"
    assert run(monkeypatch, capsys, "config", "get", "output.streaming").strip() == "true"
    with pytest.raises(SystemExit, match="secret"):
        run(monkeypatch, capsys, "config", "get", "autocorrect.api_key")


def test_service_unit_runs_the_run_subcommand():
    from voiceio.service import _service_unit
    assert "ExecStart=/x/voiceio run\n" in _service_unit("/x/voiceio")
