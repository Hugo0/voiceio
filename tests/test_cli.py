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
    config.CONFIG_PATH.write_text('[cloud]\napi_key = "sk-secret"\n')
    out = run(monkeypatch, capsys, "config", "show")
    assert "sk-secret" not in out and "•••" in out
    run(monkeypatch, capsys, "config", "set", "output.streaming_freeze_secs", "8")
    assert config.load().output.streaming_freeze_secs == 8
    assert run(monkeypatch, capsys, "config", "get", "output.streaming_freeze_secs").strip() == "8"
    assert run(monkeypatch, capsys, "config", "get", "output.streaming").strip() == "true"
    with pytest.raises(SystemExit, match="secret"):
        run(monkeypatch, capsys, "config", "get", "cloud.api_key")


def test_service_unit_runs_the_run_subcommand():
    from voiceio.service import _service_unit
    assert "ExecStart=/x/voiceio run\n" in _service_unit("/x/voiceio")


@pytest.mark.parametrize("argv", [("doctor", "--fix"), ("doctor",), ("setup", "--defaults"),
                                  ("service", "install")])
@pytest.mark.parametrize("var", ["SUDO_USER", "DOAS_USER"])
def test_user_commands_refuse_sudo(monkeypatch, capsys, argv, var):
    """Under sudo, voiceio would set up root's session: refuse, naming the fix."""
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setenv(var, "alice")
    monkeypatch.setattr("voiceio.cli.doctor._cmd_doctor", lambda a: pytest.fail("ran"))
    monkeypatch.setattr("voiceio.cli._cmd_setup", lambda a: pytest.fail("ran"))
    monkeypatch.setattr("voiceio.cli.servicecmd._cmd_service", lambda a: pytest.fail("ran"))
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys, *argv)
    msg = str(e.value.code)
    assert msg.startswith("Don't run voiceio with sudo")
    assert f"`voiceio {' '.join(argv)}` as alice" in msg


def test_sudo_guard_ignores_normal_users(monkeypatch, capsys):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    monkeypatch.setenv("SUDO_USER", "alice")  # leaked into a user shell: not root, not sudo
    called = []
    monkeypatch.setattr("voiceio.cli._cmd_setup", lambda a: called.append(a))
    run(monkeypatch, capsys, "setup", "--defaults")
    assert called and capsys.readouterr().err == ""


@pytest.mark.parametrize("sudo_user", [None, "root"])
def test_root_login_warns_and_runs(monkeypatch, capsys, sudo_user):
    """A container or root-only box: no one else to configure, so only warn."""
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.delenv("DOAS_USER", raising=False)
    if sudo_user:
        monkeypatch.setenv("SUDO_USER", sudo_user)
    else:
        monkeypatch.delenv("SUDO_USER", raising=False)
    called = []
    monkeypatch.setattr("voiceio.cli._cmd_setup", lambda a: called.append(a))
    monkeypatch.setattr(sys, "argv", ["voiceio", "setup", "--defaults"])
    main()
    assert called
    assert "running as root" in capsys.readouterr().err


def test_daemon_commands_skip_the_sudo_guard(monkeypatch, capsys):
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_USER", "alice")
    sent = []
    monkeypatch.setattr("voiceio.cli._send", lambda c: sent.append(c))
    run(monkeypatch, capsys, "toggle")
    assert sent == ["toggle"]
