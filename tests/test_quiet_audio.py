"""Quiet while recording: busctl/playerctl/wpctl/pactl driven through a fake desktop."""
from __future__ import annotations

import re
import subprocess
import threading
from unittest.mock import MagicMock

import pytest

from voiceio import quiet_audio
from voiceio.quiet_audio import QuietAudio

MPRIS = "org.mpris.MediaPlayer2."


def _ok(cmd, out=""):
    return subprocess.CompletedProcess(cmd, 0, out, "")


def _fail(cmd, err):
    return subprocess.CompletedProcess(cmd, 1, "", err)


class FakeDesktop:
    """Answers like the real tools for a set of players and output sinks."""

    def __init__(self, players=None, sinks=None, default="52"):
        self.players = dict(players or {})            # short name -> status
        self.sinks = {k: list(v) for k, v in (sinks or {"52": (0.40, False)}).items()}
        self.default = default
        self.log: list[str] = []                       # actions, in order
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        assert kwargs.get("timeout")
        self.calls.append(cmd)
        return getattr(self, cmd[0])(cmd)

    def _act(self, name, action):
        if name not in self.players:
            return False
        self.players[name] = {"pause": "Paused", "play": "Playing"}[action]
        self.log.append(f"{action} {name}")
        return True

    def busctl(self, cmd):
        assert cmd[1] == "--user"
        if cmd[-1] == "ListNames":
            names = ["org.freedesktop.DBus", ":1.42"] + [MPRIS + n for n in self.players]
            return _ok(cmd, f"as {len(names)} " + " ".join(f'"{n}"' for n in names) + "\n")
        name = cmd[3].removeprefix(MPRIS)
        if name not in self.players:
            return _fail(cmd, "Failed to get property: The name is not activatable")
        if cmd[2] == "get-property":
            return _ok(cmd, f's "{self.players[name]}"\n')
        self._act(name, cmd[-1].lower())
        return _ok(cmd)

    def playerctl(self, cmd):
        if cmd[1:3] == ["-a", "status"]:
            if not self.players:
                return _fail(cmd, "No players found")
            return _ok(cmd, "".join(f"{n}\t{s}\n" for n, s in self.players.items()))
        _, name, action = cmd[1:]
        return _ok(cmd) if self._act(name, action) else _fail(cmd, "No player could handle this command")

    def wpctl(self, cmd):
        verb, sink = cmd[1], cmd[2]
        if sink == "@DEFAULT_AUDIO_SINK@":
            sink = self.default
        if sink not in self.sinks:
            return _fail(cmd, f"Object '{sink}' not found")
        if verb == "inspect":
            return _ok(cmd, f'id {sink}, type PipeWire:Interface:Node\n    media.class = "Audio/Sink"\n')
        vol, muted = self.sinks[sink]
        if verb == "get-volume":
            return _ok(cmd, f"Volume: {vol:.2f}{' [MUTED]' if muted else ''}\n")
        assert verb == "set-volume" and re.fullmatch(r"\d+\.\d\d", cmd[3])
        self.sinks[sink][0] = float(cmd[3])
        self.log.append(f"volume {sink} {cmd[3]}")
        return _ok(cmd)

    def pactl(self, cmd):
        verb = cmd[1]
        if verb == "get-default-sink":
            return _ok(cmd, f"{self.default}\n")
        sink = cmd[2]
        if sink not in self.sinks:
            return _fail(cmd, "No such entity")
        vol, muted = self.sinks[sink]
        if verb == "get-sink-volume":
            pct = round(vol * 100)
            ch = f"{int(pct * 655.36)} / {pct:3d}% / -10.00 dB"
            return _ok(cmd, f"Volume: front-left: {ch},   front-right: {ch}\n"
                            "        balance 0.00\n")
        if verb == "get-sink-mute":
            return _ok(cmd, f"Mute: {'yes' if muted else 'no'}\n")
        assert verb == "set-sink-volume" and cmd[3].endswith("%")
        self.sinks[sink][0] = int(cmd[3][:-1]) / 100
        self.log.append(f"volume {sink} {cmd[3]}")
        return _ok(cmd)


@pytest.fixture
def tools(monkeypatch):
    """Which tools are "installed"; all four unless a test removes some."""
    installed = {"busctl", "playerctl", "wpctl", "pactl"}
    monkeypatch.setattr(quiet_audio, "_which",
                        lambda name: f"/usr/bin/{name}" if name in installed else None)
    return installed


def _drain(quiet: QuietAudio) -> None:
    for lane in (quiet._players, quiet._volume):
        if lane._executor is not None:
            lane._executor.submit(lambda: None).result(timeout=5)


def _cycle(quiet: QuietAudio) -> None:
    quiet.start()
    _drain(quiet)
    quiet.stop()
    _drain(quiet)


# ── Pausing players ─────────────────────────────────────────────────────

def test_pauses_only_playing_players_and_resumes_exactly_those(tools):
    desk = FakeDesktop({"spotify": "Playing", "vlc": "Paused", "brave.instance87427": "Playing"})
    quiet = QuietAudio(pause_media=True, run=desk)
    quiet.start()
    _drain(quiet)
    assert desk.players == {"spotify": "Paused", "vlc": "Paused", "brave.instance87427": "Paused"}
    quiet.stop()
    _drain(quiet)
    assert desk.log == ["pause spotify", "pause brave.instance87427",
                        "play spotify", "play brave.instance87427"]
    assert desk.players["vlc"] == "Paused"   # the user had paused it
    assert all(c[0] == "busctl" for c in desk.calls)


def test_busctl_call_shapes(tools):
    desk = FakeDesktop({"spotify": "Playing"})
    _cycle(QuietAudio(pause_media=True, run=desk))
    player = [MPRIS + "spotify", "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player"]
    assert desk.calls[1] == ["busctl", "--user", "get-property", *player, "PlaybackStatus"]
    assert desk.calls[2] == ["busctl", "--user", "call", *player, "Pause"]
    assert desk.calls[3] == ["busctl", "--user", "call", *player, "Play"]


def test_falls_back_to_playerctl_without_busctl(tools):
    tools.discard("busctl")
    desk = FakeDesktop({"spotify": "Playing", "vlc": "Paused"})
    _cycle(QuietAudio(pause_media=True, run=desk))
    assert desk.log == ["pause spotify", "play spotify"]
    assert all(c[0] == "playerctl" for c in desk.calls)


def test_resume_twice_does_not_replay(tools):
    desk = FakeDesktop({"spotify": "Playing"})
    quiet = QuietAudio(pause_media=True, run=desk)
    quiet.start()
    quiet.stop()
    quiet.stop()
    _drain(quiet)
    assert desk.log == ["pause spotify", "play spotify"]


def test_no_players_is_quiet(tools):
    desk = FakeDesktop({})
    _cycle(QuietAudio(pause_media=True, run=desk))
    assert desk.log == []


def test_player_gone_by_resume_is_tolerated(tools):
    desk = FakeDesktop({"mpv": "Playing"})
    quiet = QuietAudio(pause_media=True, run=desk)
    quiet.start()
    _drain(quiet)
    del desk.players["mpv"]
    quiet.stop()
    _drain(quiet)
    assert desk.calls[-1][-1] == "Play"
    assert quiet._paused == []


def test_tool_hang_never_raises(tools):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    quiet = QuietAudio(pause_media=True, duck_volume=0.2, run=run)
    stopped = MagicMock()
    _cycle(quiet)
    quiet.stop(then=stopped)
    _drain(quiet)
    stopped.assert_called_once()


def test_calls_never_block_the_caller(tools):
    """The hotkey path only enqueues; the tools run on the workers, in order."""
    gate = threading.Event()
    desk = FakeDesktop({"spotify": "Playing"})

    def slow(cmd, **kwargs):
        gate.wait(timeout=5)
        return desk(cmd, **kwargs)

    quiet = QuietAudio(pause_media=True, duck_volume=0.5, run=slow)
    quiet.start()
    quiet.stop()          # returns although the start is still running
    assert desk.calls == []
    gate.set()
    _drain(quiet)
    # Stopped before the duck ran, so the volume was never touched.
    assert desk.log == ["pause spotify", "play spotify"]


def test_close_undoes_whatever_is_still_in_effect(tools):
    desk = FakeDesktop({"spotify": "Playing"})
    quiet = QuietAudio(pause_media=True, duck_volume=0.5, run=desk)
    quiet.start()
    _drain(quiet)
    assert desk.sinks["52"][0] == 0.20
    quiet.close()
    assert desk.players["spotify"] == "Playing"
    assert desk.sinks["52"][0] == 0.40


def test_disabled_never_runs_anything(tools):
    run = MagicMock()
    quiet = QuietAudio(pause_media=False, duck_volume=1.0, run=run)
    stopped = MagicMock()
    quiet.start()
    quiet.stop(then=stopped)
    quiet.close()
    run.assert_not_called()
    stopped.assert_called_once()   # the stop chime still plays


def test_without_tools_is_a_no_op():
    run = MagicMock()   # conftest: none of the tools is installed
    quiet = QuietAudio(pause_media=True, duck_volume=0.2, run=run)
    stopped = MagicMock()
    quiet.start()
    quiet.stop(then=stopped)
    run.assert_not_called()
    stopped.assert_called_once()
    assert quiet._players._executor is None and quiet._volume._executor is None


def test_conftest_blocks_the_real_tools():
    assert quiet_audio._which("wpctl") is None
    assert quiet_audio._Caller(None)("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@") is None


# ── Ducking the output ──────────────────────────────────────────────────

def test_ducks_and_restores_the_exact_volume(tools):
    desk = FakeDesktop(sinks={"52": (1.26, False)})
    quiet = QuietAudio(pause_media=False, duck_volume=0.2, run=desk)
    quiet.start()
    _drain(quiet)
    assert desk.sinks["52"][0] == 0.25
    quiet.stop()
    _drain(quiet)
    assert desk.sinks["52"][0] == 1.26
    assert desk.log == ["volume 52 0.25", "volume 52 1.26"]
    # Resolved once to a concrete id; never set through the alias.
    assert all(c[2] == "52" for c in desk.calls if c[1] in ("get-volume", "set-volume"))


def test_muted_output_is_left_alone(tools):
    desk = FakeDesktop(sinks={"52": (0.40, True)})
    _cycle(QuietAudio(pause_media=False, duck_volume=0.2, run=desk))
    assert desk.log == []
    assert desk.sinks["52"] == [0.40, True]


def test_volume_changed_during_recording_is_not_stomped(tools):
    desk = FakeDesktop(sinks={"52": (0.40, False)})
    quiet = QuietAudio(pause_media=False, duck_volume=0.2, run=desk)
    quiet.start()
    _drain(quiet)
    desk.sinks["52"][0] = 0.70   # the user turned it up
    quiet.stop()
    _drain(quiet)
    assert desk.sinks["52"][0] == 0.70


def test_sink_switch_restores_the_ducked_device(tools):
    desk = FakeDesktop(sinks={"52": (0.40, False), "77": (0.90, False)})
    quiet = QuietAudio(pause_media=False, duck_volume=0.5, run=desk)
    quiet.start()
    _drain(quiet)
    desk.default = "77"          # headphones plugged in mid-recording
    quiet.stop()
    _drain(quiet)
    assert desk.sinks == {"52": [0.40, False], "77": [0.90, False]}


def test_sink_gone_by_restore_is_tolerated(tools):
    desk = FakeDesktop(sinks={"52": (0.40, False)})
    quiet = QuietAudio(pause_media=False, duck_volume=0.5, run=desk)
    quiet.start()
    _drain(quiet)
    del desk.sinks["52"]
    quiet.stop()
    _drain(quiet)
    assert quiet._ducked is None


def test_falls_back_to_pactl_without_wpctl(tools):
    tools.discard("wpctl")
    desk = FakeDesktop(sinks={"alsa_output.analog-stereo": (0.40, False)},
                       default="alsa_output.analog-stereo")
    quiet = QuietAudio(pause_media=False, duck_volume=0.25, run=desk)
    quiet.start()
    _drain(quiet)
    assert desk.sinks["alsa_output.analog-stereo"][0] == 0.10
    quiet.stop()
    _drain(quiet)
    assert desk.log == ["volume alsa_output.analog-stereo 10%",
                        "volume alsa_output.analog-stereo 40%"]


def test_pactl_muted_output_is_left_alone(tools):
    tools.discard("wpctl")
    desk = FakeDesktop(sinks={"s": (0.40, True)}, default="s")
    _cycle(QuietAudio(pause_media=False, duck_volume=0.2, run=desk))
    assert desk.log == []


def test_duck_waits_for_the_start_chime_and_is_skipped_if_stopped_first(tools):
    desk = FakeDesktop(sinks={"52": (0.40, False)})
    quiet = QuietAudio(pause_media=False, duck_volume=0.5, run=desk)
    quiet.start(duck_after=0.2)
    quiet.stop()
    _drain(quiet)
    assert desk.log == []


def test_each_part_is_independent(tools):
    tools.discard("wpctl")
    tools.discard("pactl")
    desk = FakeDesktop({"spotify": "Playing"})
    quiet = QuietAudio(pause_media=True, duck_volume=0.2, run=desk)
    _cycle(quiet)
    assert desk.log == ["pause spotify", "play spotify"]


# ── Wiring ──────────────────────────────────────────────────────────────

def test_record_start_and_stop_drive_it():
    from test_app_wiring import _allow_stop, _feed_audio, _make_vio
    vio, _, _ = _make_vio()
    vio._quiet = MagicMock()
    vio.on_hotkey()
    vio._quiet.start.assert_called_once()
    vio._quiet.stop.assert_not_called()
    _feed_audio(vio)
    _allow_stop(vio)
    vio.on_hotkey()
    vio._quiet.stop.assert_called_once()


def test_chimes_play_at_full_volume(tools, monkeypatch):
    """The start chime plays before the duck; the stop chime after the restore."""
    from test_app_wiring import _allow_stop, _feed_audio, _make_vio
    from voiceio import feedback
    vio, _, _ = _make_vio()
    desk = FakeDesktop(sinks={"52": (0.40, False)})
    vio._quiet = QuietAudio(pause_media=False, duck_volume=0.2, run=desk)
    vio.cfg.feedback.sound_enabled = True
    monkeypatch.setattr(feedback, "cue_secs", lambda name: 0.05)
    monkeypatch.setattr(feedback, "play_record_start", lambda: desk.log.append("start chime"))
    monkeypatch.setattr(feedback, "play_record_stop", lambda: desk.log.append("stop chime"))
    vio.on_hotkey()
    _drain(vio._quiet)
    _feed_audio(vio)
    _allow_stop(vio)
    vio.on_hotkey()
    _drain(vio._quiet)
    assert desk.log == ["start chime", "volume 52 0.08", "volume 52 0.40", "stop chime"]


def test_doctor_line(tools):
    from voiceio import config, health
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text(
        "[audio]\npause_media = true\nduck_volume = 0.2\n[tts]\nenabled = false\n")
    assert health._check_features().processing["Quiet while recording"] == (
        "pause players (busctl), duck output to 20% (wpctl)")

    tools.clear()
    tools.add("playerctl")
    assert health._check_features().processing["Quiet while recording"] == (
        "pause players (playerctl), duck output to 20%: wpctl/pactl not found")

    config.CONFIG_PATH.write_text("[tts]\nenabled = false\n")
    assert "Quiet while recording" not in health._check_features().processing
