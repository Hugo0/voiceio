import argparse

import pytest

from voiceio import power
from voiceio.learn import quiet


def _supply(root, name, kind, online):
    d = root / name
    d.mkdir()
    (d / "type").write_text(kind + "\n")
    (d / "online").write_text(online + "\n")


def test_on_battery_needs_a_mains_supply_that_is_offline(tmp_path):
    assert not power.on_battery(tmp_path)                 # desktop: no supplies listed
    _supply(tmp_path, "BAT0", "Battery", "0")
    _supply(tmp_path, "AC", "Mains", "0")
    assert power.on_battery(tmp_path)
    (tmp_path / "AC" / "online").write_text("1\n")
    assert not power.on_battery(tmp_path)


def test_busy_reason_is_load_per_cpu_then_battery(monkeypatch):
    monkeypatch.setattr(quiet.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(quiet, "on_battery", lambda: False)
    monkeypatch.setattr(quiet.os, "getloadavg", lambda: (4.0, 0, 0))     # 0.5 per CPU
    assert "load 0.50 per CPU" in quiet.busy_reason(0.25)
    monkeypatch.setattr(quiet.os, "getloadavg", lambda: (1.0, 0, 0))
    assert quiet.busy_reason(0.25) is None
    monkeypatch.setattr(quiet, "on_battery", lambda: True)
    assert quiet.busy_reason(0.25) == "on battery"


def test_wait_until_quiet_waits_then_gives_up(monkeypatch):
    now = [0.0]
    sleeps = []
    reasons = iter(["busy", "busy", None])
    monkeypatch.setattr(quiet, "busy_reason", lambda *_: next(reasons))
    args = dict(poll=300, sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                clock=lambda: now[0], say=lambda _: None)
    assert quiet.wait_until_quiet(0.25, 3, **args)
    assert sleeps == [300, 300]

    monkeypatch.setattr(quiet, "busy_reason", lambda *_: "on battery")
    said = []
    now[0] = 0.0
    assert not quiet.wait_until_quiet(0.25, 1, **{**args, "say": said.append})
    assert now[0] <= 3600 and said[-1].startswith("still on battery after 1 h")


def test_busy_reason_waits_for_the_user_to_be_away(monkeypatch):
    from voiceio import presence
    monkeypatch.setattr(quiet, "on_battery", lambda: False)
    monkeypatch.setattr(quiet.os, "getloadavg", lambda: (0.0, 0, 0))
    seen = []
    monkeypatch.setattr(presence, "check",
                        lambda away: seen.append(away) or presence.Presence(True, "in use (x)"))
    assert quiet.busy_reason(0.25, away_after=600) == "in use (x)" and seen == [600]
    assert quiet.busy_reason(0.25) is None              # unattended `maintain`: no presence check
    monkeypatch.setattr(presence, "check", lambda away: presence.Presence(False, "away"))
    assert quiet.busy_reason(0.25, away_after=600) is None


def test_waiting_logs_each_reason_once_not_every_reading(monkeypatch):
    readings = iter(["in use (last input 3 s ago)", "in use (last input 9 s ago)",
                     "on battery", None])
    monkeypatch.setattr(quiet, "busy_reason", lambda *_: next(readings))
    said = []
    assert quiet.wait_until_quiet(0.25, 3, poll=1, sleep=lambda s: None, say=said.append)
    assert said == ["waiting: in use (last input 3 s ago)", "waiting: on battery"]


class _Proc:
    """A worker that runs for `polls` checks, then exits with `code`."""
    def __init__(self, polls=99, code=0):
        self.left, self.code, self.pid, self.stopped = polls, code, 4242, False

    def poll(self):
        if self.stopped:
            return -15
        if self.left <= 0:
            return self.code
        self.left -= 1
        return None

    @property
    def returncode(self):
        return self.poll()


def _watch(monkeypatch, tmp_path, procs, reasons, *, quiet_ok=True):
    spawned, said = [], []
    procs = iter(procs)
    reasons = iter(reasons)

    def spawn(argv, env):
        spawned.append(env)
        return next(procs)

    def stop(proc):
        proc.stopped = True
    monkeypatch.setattr(quiet, "wait_until_quiet", lambda *a, **k: quiet_ok)
    monkeypatch.setattr(quiet, "stop_reason", lambda away: next(reasons, None))
    code = quiet.run_while_away(["voiceio", "learn", "maintain"], max_load=0.25, away_minutes=10,
                                wait_hours=3, threads=3, hold_file=tmp_path / "hold",
                                spawn=spawn, stop=stop, sleep=lambda s: None, say=said.append)
    return code, spawned, said


def test_work_stops_when_the_user_returns_and_resumes_later(monkeypatch, tmp_path):
    first, second = _Proc(), _Proc(polls=3, code=0)
    code, spawned, said = _watch(monkeypatch, tmp_path, [first, second],
                                 [None, None, "in use (last input 0 s ago)"])
    assert code == 0 and first.stopped and not second.stopped
    assert said == ["paused: in use (last input 0 s ago); it resumes where it stopped "
                    "once you've been away 10 min"]
    assert spawned[0][quiet.WORKER_ENV] == "1" and spawned[0]["OMP_NUM_THREADS"] == "3"
    assert quiet.RESUMED_ENV not in spawned[0] and spawned[1][quiet.RESUMED_ENV] == "1"


def test_watcher_reports_the_work_exit_status_and_gives_up_quietly(monkeypatch, tmp_path):
    assert _watch(monkeypatch, tmp_path, [_Proc(polls=1, code=3)], [])[0] == 3
    code, spawned, _ = _watch(monkeypatch, tmp_path, [], [], quiet_ok=False)
    assert code == 0 and spawned == []          # never away: nothing started, unit doesn't fail


def test_a_model_switch_in_progress_is_never_cut_off(monkeypatch, tmp_path):
    proc = _Proc(polls=3, code=0)
    reasons = iter(["in use"] * 10)

    def spawn(argv, env):
        (tmp_path / "hold").touch()            # the worker holds it
        return proc
    monkeypatch.setattr(quiet, "wait_until_quiet", lambda *a, **k: True)
    monkeypatch.setattr(quiet, "stop_reason", lambda away: next(reasons))
    assert quiet.run_while_away(["x"], max_load=1, away_minutes=1, wait_hours=1, threads=1,
                                hold_file=tmp_path / "hold", spawn=spawn,
                                stop=lambda p: pytest.fail("cut off a model switch"),
                                sleep=lambda s: None, say=print) == 0
    assert not (tmp_path / "hold").exists()


def test_hold_marks_the_file_only_inside_the_block(monkeypatch, tmp_path):
    flag = tmp_path / "hold"
    monkeypatch.setenv(quiet.HOLD_ENV, str(flag))
    with quiet.hold():
        assert flag.exists()
    assert not flag.exists()
    monkeypatch.delenv(quiet.HOLD_ENV)
    with quiet.hold():                          # not under the watcher: a no-op
        pass


def test_stop_ends_the_whole_process_group_at_once():
    """The decoder runs in a subprocess of the worker: it must go too."""
    import os
    import time
    proc = quiet._spawn(["sh", "-c", "sleep 30 & sleep 30; wait"], dict(os.environ))
    time.sleep(0.2)
    t0 = time.monotonic()
    quiet._stop(proc)
    assert time.monotonic() - t0 < 3 and proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)                  # nothing left in the group


def test_threads_default_to_half_the_cpus(monkeypatch):
    monkeypatch.setattr(quiet.os, "cpu_count", lambda: 16)
    assert quiet.worker_threads(0) == 8 and quiet.worker_threads(5) == 5


def test_maintain_in_the_worker_does_the_work(monkeypatch):
    from voiceio.cli import learn as cli
    monkeypatch.setattr(cli, "_yield_cpu", lambda: None)
    monkeypatch.setenv(quiet.WORKER_ENV, "1")
    monkeypatch.setattr(quiet, "run_while_away", lambda *a, **k: pytest.fail("worker must not watch"))
    ran = []
    monkeypatch.setattr(cli, "_maintain", lambda args, cfg: ran.append(cfg.away_minutes))
    cli.cmd_maintain(argparse.Namespace(min_new=5, base="small", epochs=1, when_idle=True))
    assert ran == [10.0]


def test_scheduled_maintain_fails_when_its_work_does(monkeypatch):
    from voiceio.cli import learn as cli
    monkeypatch.setattr(cli, "_yield_cpu", lambda: None)
    monkeypatch.delenv(quiet.WORKER_ENV, raising=False)
    monkeypatch.setattr(quiet, "run_while_away", lambda *a, **k: 2)
    with pytest.raises(SystemExit) as e:
        cli.cmd_maintain(argparse.Namespace(min_new=5, base="small", epochs=1, when_idle=True))
    assert e.value.code == 2


def test_scheduled_maintain_waits_for_a_quiet_machine(monkeypatch):
    from voiceio.cli import learn as cli
    monkeypatch.setattr(cli, "_yield_cpu", lambda: None)
    monkeypatch.setattr(quiet, "wait_until_quiet", lambda *a, **k: False)
    ran = []
    monkeypatch.setattr(cli, "cmd_label", lambda a: ran.append("label"))
    cli.cmd_maintain(argparse.Namespace(min_new=5, base="small", epochs=1, when_idle=True))
    assert ran == []                                      # busy all night: nothing ran


def test_learning_announcement_offers_stop_and_weekly(monkeypatch, tmp_path):
    from voiceio import feedback
    from voiceio.cli import learn as cli
    sent = []
    monkeypatch.setattr(feedback, "notify_actions", lambda *a: sent.append(a))
    monkeypatch.setattr(cli.store, "load_labels", lambda: {})
    monkeypatch.setattr(cli.store, "clips", lambda: [argparse.Namespace(audio=f"a{i}.wav") for i in range(3)])
    monkeypatch.setattr(cli.store, "latest_dataset", lambda: None)
    cli._announce_learning(min_new=50)
    (title, body, actions, _), = sent
    assert "learning" in title and body.startswith("Labeling 3 new recordings,")
    assert "while you're away" in body and set(actions) == {"stop", "weekly"}

    sent.clear()
    monkeypatch.setattr(cli.store, "load_labels", lambda: {f"a{i}.wav": 1 for i in range(3)})
    monkeypatch.setattr(cli.store, "latest_dataset", lambda: tmp_path)
    monkeypatch.setattr(cli.store, "dataset_clips", lambda _: {f"a{i}.wav" for i in range(3)})
    cli._announce_learning(min_new=50)
    assert sent == []                                     # nothing to do: no notification


def test_notify_actions_runs_the_picked_action(monkeypatch):
    import subprocess

    from voiceio import feedback
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(feedback, "_which", lambda _: "/usr/bin/notify-send")
    monkeypatch.setattr(feedback, "_notify_send_has_actions", lambda: True)
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="weekly\n")
    monkeypatch.setattr(feedback.subprocess, "run", fake_run)
    picked = []

    class Now:   # run the waiting thread inline
        def __init__(self, target, daemon): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(feedback.threading, "Thread", Now)
    feedback.notify_actions("t", "b", {"stop": "Stop", "weekly": "Weekly"}, picked.append)
    assert picked == ["weekly"]
    assert "--wait" in calls[0] and "--action=stop=Stop" in calls[0]
