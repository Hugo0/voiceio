"""Pause media while recording: playerctl driven through a fake runner."""
from __future__ import annotations

import subprocess
import threading
from unittest.mock import MagicMock

import pytest

from voiceio import media
from voiceio.media import MediaPauser


class FakePlayerctl:
    """Records argv and answers like playerctl for a set of players."""

    def __init__(self, players: dict[str, str]):
        self.players = players
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == "playerctl" and kwargs.get("timeout")
        self.calls.append(cmd[1:])
        if cmd[1:3] == ["-a", "status"]:
            if not self.players:
                return subprocess.CompletedProcess(cmd, 1, "", "No players found")
            out = "".join(f"{n}\t{s}\n" for n, s in self.players.items())
            return subprocess.CompletedProcess(cmd, 0, out, "")
        _, name, action = cmd[1:]
        if name not in self.players:
            return subprocess.CompletedProcess(cmd, 1, "", "No player could handle this command")
        self.players[name] = {"pause": "Paused", "play": "Playing"}[action]
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def actions(self):
        return [c for c in self.calls if c[0] == "-p"]


@pytest.fixture
def have_playerctl(monkeypatch):
    monkeypatch.setattr(media, "_playerctl", lambda: "/usr/bin/playerctl")


def _drain(pauser: MediaPauser) -> None:
    pauser._executor.submit(lambda: None).result(timeout=5)


def test_pauses_only_playing_players_and_resumes_exactly_those(have_playerctl):
    fake = FakePlayerctl({"spotify": "Playing", "vlc": "Paused",
                          "chromium.instance42": "Playing"})
    pauser = MediaPauser(True, run=fake)
    pauser.pause()
    _drain(pauser)
    assert fake.players == {"spotify": "Paused", "vlc": "Paused",
                            "chromium.instance42": "Paused"}
    pauser.resume()
    _drain(pauser)
    assert fake.actions() == [
        ["-p", "spotify", "pause"], ["-p", "chromium.instance42", "pause"],
        ["-p", "spotify", "play"], ["-p", "chromium.instance42", "play"],
    ]
    assert fake.players["vlc"] == "Paused"   # the user had paused it


def test_resume_twice_does_not_replay(have_playerctl):
    fake = FakePlayerctl({"spotify": "Playing"})
    pauser = MediaPauser(True, run=fake)
    pauser.pause()
    pauser.resume()
    pauser.resume()
    _drain(pauser)
    assert fake.actions().count(["-p", "spotify", "play"]) == 1


def test_no_players_is_quiet(have_playerctl):
    fake = FakePlayerctl({})
    pauser = MediaPauser(True, run=fake)
    pauser.pause()
    pauser.resume()
    _drain(pauser)
    assert fake.actions() == []


def test_player_gone_by_resume_is_tolerated(have_playerctl):
    fake = FakePlayerctl({"mpv": "Playing"})
    pauser = MediaPauser(True, run=fake)
    pauser.pause()
    _drain(pauser)
    del fake.players["mpv"]
    pauser.resume()
    _drain(pauser)
    assert fake.actions()[-1] == ["-p", "mpv", "play"]


def test_playerctl_hang_never_raises(have_playerctl):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    pauser = MediaPauser(True, run=run)
    pauser.pause()
    pauser.resume()
    _drain(pauser)


def test_calls_never_block_the_caller(have_playerctl):
    """The hotkey path only enqueues; playerctl runs on the worker, in order."""
    gate = threading.Event()
    fake = FakePlayerctl({"spotify": "Playing"})

    def slow(cmd, **kwargs):
        gate.wait(timeout=5)
        return fake(cmd, **kwargs)

    pauser = MediaPauser(True, run=slow)
    pauser.pause()
    pauser.resume()          # returns although the pause is still running
    assert fake.calls == []
    gate.set()
    _drain(pauser)
    assert fake.actions() == [["-p", "spotify", "pause"], ["-p", "spotify", "play"]]


def test_close_resumes_whatever_is_still_paused(have_playerctl):
    fake = FakePlayerctl({"spotify": "Playing"})
    pauser = MediaPauser(True, run=fake)
    pauser.pause()
    pauser.close()
    assert fake.players["spotify"] == "Playing"


def test_disabled_never_runs_anything(have_playerctl):
    run = MagicMock()
    pauser = MediaPauser(False, run=run)
    pauser.pause()
    pauser.resume()
    pauser.close()
    run.assert_not_called()


def test_without_playerctl_is_a_no_op():
    run = MagicMock()   # conftest: playerctl is "not installed"
    pauser = MediaPauser(True, run=run)
    pauser.pause()
    pauser.resume()
    run.assert_not_called()
    assert pauser._executor is None


def test_record_start_pauses_and_stop_resumes():
    from test_app_wiring import _allow_stop, _feed_audio, _make_vio
    vio, _, _ = _make_vio()
    vio._media = MagicMock()
    vio.on_hotkey()
    vio._media.pause.assert_called_once()
    vio._media.resume.assert_not_called()
    _feed_audio(vio)
    _allow_stop(vio)
    vio.on_hotkey()
    vio._media.resume.assert_called_once()


def test_doctor_flags_missing_playerctl(monkeypatch):
    from voiceio import config, health
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text("[audio]\npause_media = true\n[tts]\nenabled = false\n")
    monkeypatch.setattr(health.shutil, "which", lambda name: None)
    assert health._check_features().processing["Pause media"].startswith(
        "enabled but playerctl not installed")

    config.CONFIG_PATH.write_text("[tts]\nenabled = false\n")
    assert "Pause media" not in health._check_features().processing
