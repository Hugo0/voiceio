import sys
from pathlib import Path

from voiceio import service


def test_install_symlinks_relinks_and_drops_legacy(monkeypatch, tmp_path):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    for name in service.SCRIPT_NAMES:
        (venv_bin / name).write_text("#!/bin/sh\n")
    old = tmp_path / "old"
    old.mkdir()
    local_bin = tmp_path / "local-bin"
    local_bin.mkdir()
    (local_bin / "voiceio").symlink_to(old / "voiceio")        # stale install
    (local_bin / "voiceio-doctor").symlink_to(old / "voiceio-doctor")
    monkeypatch.setattr(service, "LOCAL_BIN", local_bin)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(service, "_is_pipx_install", lambda: False)
    monkeypatch.setattr(service, "_local_bin_on_path", lambda: True)

    assert service.install_symlinks() == service.SCRIPT_NAMES
    assert Path(local_bin / "voiceio").resolve() == (venv_bin / "voiceio").resolve()
    assert not (local_bin / "voiceio-doctor").is_symlink()


def test_learn_timer_runs_maintain_at_idle_priority(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_systemctl", lambda *a: calls.append(a) or True)
    monkeypatch.setattr(service, "_find_voiceio_bin", lambda: "/opt/v/bin/voiceio")
    monkeypatch.setattr(service, "SERVICE_DIR", tmp_path)
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / service.LEARN_SERVICE_NAME)
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / service.LEARN_TIMER_NAME)

    assert service.install_learn_timer()
    unit = service.LEARN_SERVICE_PATH.read_text()
    assert "ExecStart=/opt/v/bin/voiceio learn maintain" in unit
    assert "CPUWeight=idle" in unit and "VOICEIO_IDLE_SCOPE=1" in unit
    assert "Persistent=true" in service.LEARN_TIMER_PATH.read_text()
    assert ("enable", "--now", service.LEARN_TIMER_NAME) in calls

    service.uninstall_learn_timer()
    assert not service.LEARN_TIMER_PATH.exists() and not service.LEARN_SERVICE_PATH.exists()


def test_daemon_publishes_state_and_restart_waits_for_idle(monkeypatch):
    import threading

    from voiceio import config
    from voiceio.app import VoiceIO, _State
    from voiceio.cli.servicecmd import wait_until_idle

    assert wait_until_idle(timeout=1)                 # not running: no wait
    vio = VoiceIO.__new__(VoiceIO)
    vio._state = _State.RECORDING
    assert config.STATE_PATH.read_text() == "recording"
    threading.Timer(0.3, lambda: setattr(vio, "_state", _State.IDLE)).start()
    assert wait_until_idle(timeout=5, poll=0.05)      # waited for the dictation
    vio._state = _State.FINALIZING
    assert not wait_until_idle(timeout=0.2, poll=0.05)   # gives up, says so


def test_learn_timer_auto_promote_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_systemctl", lambda *a: True)
    monkeypatch.setattr(service, "_find_voiceio_bin", lambda: "voiceio")
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / "s.service")
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / "s.timer")
    service.install_learn_timer()
    assert not service.learn_timer_auto_promotes()
    service.install_learn_timer(auto_promote=True)
    assert "learn maintain --auto-promote" in service.LEARN_SERVICE_PATH.read_text()
    assert service.learn_timer_auto_promotes()
