import argparse

from voiceio.learn import quiet


def _supply(root, name, kind, online):
    d = root / name
    d.mkdir()
    (d / "type").write_text(kind + "\n")
    (d / "online").write_text(online + "\n")


def test_on_battery_needs_a_mains_supply_that_is_offline(tmp_path):
    assert not quiet.on_battery(tmp_path)                 # desktop: no supplies listed
    _supply(tmp_path, "BAT0", "Battery", "0")
    _supply(tmp_path, "AC", "Mains", "0")
    assert quiet.on_battery(tmp_path)
    (tmp_path / "AC" / "online").write_text("1\n")
    assert not quiet.on_battery(tmp_path)


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
    monkeypatch.setattr(quiet, "busy_reason", lambda _: next(reasons))
    args = dict(poll=300, sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                clock=lambda: now[0], say=lambda _: None)
    assert quiet.wait_until_quiet(0.25, 3, **args)
    assert sleeps == [300, 300]

    monkeypatch.setattr(quiet, "busy_reason", lambda _: "on battery")
    said = []
    now[0] = 0.0
    assert not quiet.wait_until_quiet(0.25, 1, **{**args, "say": said.append})
    assert now[0] <= 3600 and said[-1].startswith("still on battery after 1 h")


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
    assert "on this computer" in body and set(actions) == {"stop", "weekly"}

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
