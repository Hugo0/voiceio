"""Idle audio policy: when the mic and cue streams are open between dictations."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sounddevice")

from voiceio import feedback, idle_audio  # noqa: E402
from voiceio.config import AudioConfig, Config  # noqa: E402
from voiceio.idle_audio import IdleAudio  # noqa: E402
from voiceio.recorder import AudioRecorder  # noqa: E402


class FakeInput:
    """A capture stream that never touches a device."""
    opened = 0

    def __init__(self, **kwargs):
        type(self).opened += 1
        self.active, self.closed, self.stopped = False, False, True

    def start(self):
        self.active, self.stopped = True, False

    def stop(self):
        self.active, self.stopped = False, True

    def close(self):
        self.closed = True


class FakeOutput(FakeInput):
    opened = 0
    latency = 0.08
    samplerate = 44100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.written = []

    def write(self, data):
        self.written.append(len(data))


@pytest.fixture
def streams(monkeypatch):
    FakeInput.opened = FakeOutput.opened = 0
    monkeypatch.setattr("voiceio.recorder.sd.InputStream", FakeInput)
    monkeypatch.setattr(feedback.sd, "OutputStream", FakeOutput)
    monkeypatch.setattr(feedback, "_stream", None)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _policy(mode, *, battery=False, warm_minutes=5.0, can_close=lambda: True,
            lock=None, cues=True):
    rec = AudioRecorder(AudioConfig())
    clock = Clock()
    power = {"battery": battery}
    pol = IdleAudio(mode, warm_minutes, recorder=rec, cues=lambda: cues,
                    lock=lock or threading.Lock(), can_close=can_close,
                    on_battery=lambda: power["battery"], clock=clock)
    return pol, rec, clock, power


def _open(rec):
    return rec.is_open and feedback.cue_stream_open()


def _closed(rec):
    return rec.is_parked and not rec.is_open and not feedback.cue_stream_open()


# ── Policies ────────────────────────────────────────────────────────────

def test_always_stays_open_on_battery(streams):
    pol, rec, clock, _ = _policy("always", battery=True)
    pol.start()
    assert _open(rec)
    clock.now += 24 * 3600
    pol.tick()
    assert _open(rec)


def test_auto_on_ac_stays_open(streams):
    pol, rec, clock, _ = _policy("auto")
    pol.start()
    clock.now += 3600
    pol.tick()
    assert _open(rec)


def test_auto_on_battery_closes_after_the_warm_window(streams):
    pol, rec, clock, _ = _policy("auto", battery=True, warm_minutes=5)
    pol.start()
    assert _closed(rec)                      # startup on battery: no dictation yet
    assert pol.ensure_open() is True         # hotkey: cold open
    assert _open(rec)
    pol.touch()                              # dictation ended
    clock.now += 299
    pol.tick()
    assert _open(rec)
    clock.now += 2
    pol.tick()
    assert _closed(rec)


def test_auto_battery_and_ac_transitions(streams):
    pol, rec, clock, power = _policy("auto")
    pol.start()
    assert _open(rec)
    power["battery"] = True                  # unplugged, last dictation long ago
    clock.now += 600
    pol.tick()
    assert _closed(rec)
    power["battery"] = False                 # plugged back in: warm again
    pol.tick()
    assert _open(rec)
    assert FakeInput.opened == FakeOutput.opened == 2


def test_never_closes_shortly_after_each_dictation(streams):
    pol, rec, clock, _ = _policy("never")
    pol.start()
    assert _closed(rec)
    assert pol.ensure_open() is True
    pol.touch()
    pol.shutdown()                           # drive the timer by hand
    pol.tick()
    assert _open(rec)                        # commit chime may still be playing
    clock.now += idle_audio.NEVER_GRACE_SECS + 0.1
    pol.tick()
    assert _closed(rec)


def test_never_timer_closes_without_the_watchdog(streams, monkeypatch):
    monkeypatch.setattr(idle_audio, "NEVER_GRACE_SECS", 0.05)
    rec = AudioRecorder(AudioConfig())
    pol = IdleAudio("never", 5, recorder=rec, cues=lambda: True,
                    lock=threading.Lock(), can_close=lambda: True)
    pol.ensure_open()
    pol.touch()
    deadline = time.monotonic() + 3
    while not rec.is_parked and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _closed(rec)


def test_unknown_mode_is_auto():
    assert idle_audio.normalize_mode("sometimes") == "auto"


def test_warm_ensure_open_is_not_cold(streams):
    pol, rec, _, _ = _policy("always")
    pol.start()
    assert pol.ensure_open() is False


def test_cues_off_leaves_the_output_alone(streams):
    pol, rec, _, _ = _policy("always", cues=False)
    pol.start()
    assert rec.is_open and not feedback.cue_stream_open()


# ── Never close while busy ──────────────────────────────────────────────

def test_tick_never_closes_while_busy(streams):
    busy = {"v": True}
    pol, rec, clock, _ = _policy("auto", battery=True, can_close=lambda: not busy["v"])
    pol.ensure_open()
    clock.now += 3600                        # a one-hour dictation
    pol.tick()
    assert _open(rec)
    busy["v"] = False                        # it ended: the window restarts now
    clock.now += 10
    pol.tick()
    assert _open(rec)
    clock.now += 300
    pol.tick()
    assert _closed(rec)


def test_tick_skips_while_the_state_lock_is_held(streams):
    lock = threading.Lock()
    pol, rec, clock, _ = _policy("auto", battery=True, lock=lock)
    pol.ensure_open()
    clock.now += 3600
    with lock:                               # a hotkey is starting a recording
        pol.tick()
    assert _open(rec)


def test_recorder_refuses_to_park_while_recording(streams):
    rec = AudioRecorder(AudioConfig())
    rec.open_stream()
    rec.start()
    with pytest.raises(RuntimeError):
        rec.park()
    assert rec.is_open


def test_close_waits_for_a_cue_being_written(streams):
    feedback.open_cue_stream()
    stream = feedback._stream
    with feedback._stream_lock:
        closer = threading.Thread(target=feedback.close_cue_stream)
        closer.start()
        closer.join(0.1)
        assert closer.is_alive() and not stream.closed
    closer.join(1)
    assert stream.closed and not feedback.cue_stream_open()
    feedback._play_sync("start")             # a late cue after close: no crash


# ── Recorder and watchdog with a parked stream ──────────────────────────

def test_parked_stream_is_healthy_and_resume_leaves_it_closed(streams):
    from voiceio.watchdog import Watchdog
    rec = AudioRecorder(AudioConfig())
    rec.open_stream()
    rec.park()
    assert rec.stream_health() == (True, "")
    wd = Watchdog(Config(), decoder=MagicMock(), recorder=rec, typers=MagicMock(),
                  ibus=MagicMock(), session=lambda: None, is_idle=lambda: True,
                  is_recording=lambda: False, on_decoder_failed=MagicMock(),
                  on_hotkey=MagicMock(), shutdown=threading.Event())
    wd.on_resume()
    wd.check()
    assert not rec.is_open and FakeInput.opened == 1


def test_reopen_resets_a_stale_heartbeat(streams):
    rec = AudioRecorder(AudioConfig())
    rec.open_stream()
    rec._last_callback_time = time.monotonic() - 600   # last callback before the park
    rec.park()
    rec.open_stream()
    assert rec.stream_health() == (True, "")


def test_watchdog_tick_applies_the_policy(streams):
    from voiceio.watchdog import Watchdog
    pol = MagicMock()
    rec = MagicMock()
    rec.stream_health.return_value = (True, "")
    wd = Watchdog(Config(), decoder=MagicMock(), recorder=rec, typers=MagicMock(),
                  ibus=MagicMock(), session=lambda: None, is_idle=lambda: True,
                  is_recording=lambda: False, on_decoder_failed=MagicMock(),
                  on_hotkey=MagicMock(), shutdown=threading.Event(), idle_audio=pol)
    wd.check()
    pol.tick.assert_called_once()


# ── The daemon: cold start, TTS, recording ──────────────────────────────

def _make_app(keep_warm="never"):
    from voiceio.app import VoiceIO
    hotkey = MagicMock()
    hotkey.name = "socket"
    typer = MagicMock()
    typer.name = "clipboard"
    cfg = Config()
    cfg.audio.keep_warm = keep_warm
    cfg.output.streaming = False
    with patch("voiceio.app.hotkey_chain.select", return_value=hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        vio = VoiceIO(cfg)
    vio._idle_audio.shutdown()
    return vio


def test_cold_start_records_without_the_zombie_check(streams):
    from voiceio.app import _State
    vio = _make_app("never")
    vio._idle_audio.start()
    assert vio.recorder.is_parked
    vio.recorder.is_zombie = MagicMock(return_value=True)
    vio.recorder.has_signal = MagicMock(return_value=False)
    with patch("voiceio.app.time.sleep") as sleep:
        vio.on_hotkey()
    assert vio._state == _State.RECORDING
    assert vio.recorder.is_recording and vio.recorder.is_open
    assert feedback.cue_stream_open()                 # the chime could play
    vio.recorder.is_zombie.assert_not_called()       # empty ring: can't judge
    sleep.assert_not_called()                        # no 0.4s wait to refill it
    vio._idle_audio.tick()                           # never closes mid-recording
    assert vio.recorder.is_open
    vio._idle_audio.shutdown()


def test_cold_start_audio_starts_at_the_open(streams):
    """No pre-buffer after a cold open, but nothing captured since is lost."""
    vio = _make_app("never")
    vio._idle_audio.start()
    vio.on_hotkey()
    chunk = np.full((1600, 1), 0.2, dtype=np.float32)
    vio.recorder._callback(chunk, 1600, None, None)
    assert len(vio.recorder.get_audio_so_far()) == 1600
    vio._idle_audio.shutdown()


def test_no_close_while_finalizing_or_tts(streams):
    from voiceio.app import _State
    vio = _make_app("never")
    vio._idle_audio.ensure_open()
    vio._idle_audio._last_activity = -1e9            # window long over
    vio._state = _State.FINALIZING
    vio._idle_audio.tick()
    assert vio.recorder.is_open
    vio._state = _State.IDLE
    vio._outputs = 1                                  # batch output in flight
    vio._idle_audio.tick()
    assert vio.recorder.is_open
    vio._outputs = 0
    vio._tts_player = MagicMock()
    vio._tts_player.is_playing.return_value = True
    vio._idle_audio._last_activity = -1e9
    vio._idle_audio.tick()
    assert vio.recorder.is_open and feedback.cue_stream_open()
    vio._tts_player.is_playing.return_value = False
    vio._idle_audio._last_activity = -1e9
    vio._idle_audio.tick()
    assert vio.recorder.is_parked and not feedback.cue_stream_open()


# ── Doctor ──────────────────────────────────────────────────────────────

def test_doctor_line():
    assert idle_audio.describe("auto", 5, on_battery=lambda: False) == (
        "kept warm (on AC) / closes after 5 min on battery")
    assert idle_audio.describe("auto", 5, on_battery=lambda: True) == (
        "closes after 5 min on battery (on battery now)")
    assert idle_audio.describe("always", 5).startswith("kept warm")
    assert idle_audio.describe("never", 5).startswith("closed")


def test_doctor_reports_it():
    from voiceio import health
    assert health._check_features().processing["Audio when idle"] == (
        "kept warm (on AC) / closes after 5 min on battery")
