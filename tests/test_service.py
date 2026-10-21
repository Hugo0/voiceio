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
    assert "ExecStart=/opt/v/bin/voiceio learn maintain --when-idle" in unit
    assert "CPUWeight=idle" in unit and "VOICEIO_IDLE_SCOPE=1" in unit
    # Defence in depth behind the presence watch: idle CPU class inside the
    # cgroup too, idle disk, a memory ceiling, first in line for the OOM killer.
    for line in ("CPUSchedulingPolicy=idle", "Nice=19", "IOSchedulingClass=idle", "IOWeight=1",
                 "MemoryHigh=25%", "OOMScoreAdjust=500"):
        assert f"\n{line}\n" in unit, line
    timer = service.LEARN_TIMER_PATH.read_text()
    assert "Persistent=true" in timer and "OnCalendar=Sun *-*-* 03:30" in timer
    assert ("enable", "--now", service.LEARN_TIMER_NAME) in calls

    service.uninstall_learn_timer()
    assert not service.LEARN_TIMER_PATH.exists() and not service.LEARN_SERVICE_PATH.exists()


def test_daemon_publishes_state_and_restart_waits_for_idle(monkeypatch):
    import threading

    from voiceio import config, pidlock
    from voiceio.app import VoiceIO, _State
    from voiceio.cli.servicecmd import wait_until_idle

    vio = VoiceIO.__new__(VoiceIO)
    vio._state = _State.RECORDING                      # not the lock holder:
    assert not config.STATE_PATH.exists()              # says nothing
    vio._publishes_state = True
    vio._state = _State.RECORDING
    assert config.STATE_PATH.read_text() == "recording"

    monkeypatch.setattr(pidlock, "daemon_alive", lambda path: False)
    assert wait_until_idle(timeout=5)                  # dead daemon: stale file, no wait
    monkeypatch.setattr(pidlock, "daemon_alive", lambda path: True)
    threading.Timer(0.3, lambda: setattr(vio, "_state", _State.IDLE)).start()
    assert wait_until_idle(timeout=5, poll=0.05)       # waited for the dictation
    vio._state = _State.FINALIZING                     # a hung finalize: short budget
    assert not wait_until_idle(timeout=60, poll=0.05, finalizing_timeout=0.2)


def test_daemon_alive_probes_the_pid_lock(tmp_path):
    import fcntl

    from voiceio.pidlock import daemon_alive
    path = tmp_path / "voiceio.pid"
    assert not daemon_alive(path)
    with open(path, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert daemon_alive(path)
    assert not daemon_alive(path)


def test_learn_timer_auto_promote_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_systemctl", lambda *a: True)
    monkeypatch.setattr(service, "_find_voiceio_bin", lambda: "voiceio")
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / "s.service")
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / "s.timer")
    service.install_learn_timer()
    assert not service.learn_timer_auto_promotes()
    service.install_learn_timer(auto_promote=True)
    assert "learn maintain --when-idle --auto-promote" in service.LEARN_SERVICE_PATH.read_text()
    assert service.learn_timer_auto_promotes()


def test_learn_schedule_comes_from_config(monkeypatch, tmp_path):
    import pytest

    from voiceio import config
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_systemctl", lambda *a: True)
    monkeypatch.setattr(service, "_find_voiceio_bin", lambda: "voiceio")
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / "s.service")
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / "s.timer")
    service.install_learn_timer()
    assert "OnCalendar=Sun *-*-* 03:30" in service.LEARN_TIMER_PATH.read_text()   # weekly by default

    config.set_value("learn", "at", "2:05")
    config.set_value("learn", "every", "week")
    service.install_learn_timer()
    assert "OnCalendar=Sun *-*-* 02:05" in service.LEARN_TIMER_PATH.read_text()
    assert service.learn_schedule_summary() == "weekly on Sunday at 2:05"

    for at, every in (("25:00", "day"), ("3pm", "day"), ("03:30", "month")):
        with pytest.raises(ValueError):
            service.learn_calendar(at, every)


def test_reinstalling_the_service_refreshes_learning_units_that_are_on(monkeypatch, tmp_path):
    """An upgrade must not leave last version's learning unit running."""
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_systemctl", lambda *a: True)
    monkeypatch.setattr(service, "_find_voiceio_bin", lambda: "voiceio")
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / "s.service")
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / "s.timer")
    assert not service.refresh_learn_timer()               # off: stays off
    assert not service.LEARN_SERVICE_PATH.exists()

    service.LEARN_TIMER_PATH.write_text("old")
    service.LEARN_SERVICE_PATH.write_text("ExecStart=voiceio learn maintain --auto-promote\n")
    assert service.refresh_learn_timer()
    unit = service.LEARN_SERVICE_PATH.read_text()
    assert "MemoryHigh=" in unit and "--when-idle --auto-promote" in unit

    calls = []
    monkeypatch.setattr(service, "SERVICE_DIR", tmp_path)
    monkeypatch.setattr(service, "SERVICE_PATH", tmp_path / service.SERVICE_NAME)
    monkeypatch.setattr(service, "install_symlinks", lambda: [])
    monkeypatch.setattr(service, "uninstall_correct_timer", lambda: None)
    monkeypatch.setattr(service, "refresh_learn_timer", lambda: calls.append("refresh"))
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: None)
    assert service.install_service() and calls == ["refresh"]
