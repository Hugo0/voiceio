"""A reopened mic re-resolves its device against a fresh PortAudio device list.

The real failure: the user switched to a USB mic (new default source) while
the daemon had been up for days; every dead-stream reopen landed on the old
device PortAudio had cached at import, and only a restart fixed it.
"""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from voiceio import audio_devices, feedback
from voiceio.config import AudioConfig
from voiceio.recorder import AudioRecorder

# Taken at import, before conftest's per-test stub replaces it.
_REAL_REINITIALIZE = audio_devices._reinitialize


class FakeStream:
    def __init__(self, pa: FakePortAudio, device: int, samplerate: int = 16000):
        self._pa = pa
        self.device = device
        self.samplerate = samplerate
        self.latency = 0.08
        self.closed = False
        self.active = False
        self.stopped = True
        self.written = 0

    def start(self):
        self.active, self.stopped = True, False

    def stop(self):
        self.active, self.stopped = False, True

    def close(self):
        self.closed = True

    def write(self, data):
        assert not self.closed and self._pa._initialized, "write on a freed stream"
        self.written += len(data)


class FakePortAudio:
    """sounddevice over a device list that only refreshes at Pa_Initialize,
    like the real one. Pa_Terminate with a stream still open is a failure."""

    PortAudioError = RuntimeError

    def __init__(self, devices: list[str], default: int):
        self.live_devices, self.live_default = list(devices), default
        self._initialized = 1
        self.inits = 0
        self.streams: list[FakeStream] = []
        self._snapshot()

    def _snapshot(self):
        self.devices, self.default = list(self.live_devices), self.live_default

    def plug(self, devices: list[str], default: int):
        self.live_devices, self.live_default = list(devices), default

    def _terminate(self):
        still_open = [s for s in self.streams if not s.closed]
        assert not still_open, f"Pa_Terminate with open streams: {still_open}"
        self._initialized -= 1

    def _initialize(self):
        self._initialized += 1
        self.inits += 1
        self._snapshot()

    def _resolve(self, device):
        if device is None:
            return self.default
        if isinstance(device, int):
            return device
        for i, name in enumerate(self.devices):
            if device.lower() in name.lower():
                return i
        raise ValueError(f"No input device matching {device!r}")

    def query_devices(self, device=None, kind=None):
        index = self.default if kind == "input" else device
        return {"name": self.devices[index], "index": index}

    def InputStream(self, device=None, samplerate=16000, **_):
        stream = FakeStream(self, self._resolve(device), samplerate)
        self.streams.append(stream)
        return stream

    def OutputStream(self, device=None, samplerate=44100, **_):
        stream = FakeStream(self, 0, samplerate)
        self.streams.append(stream)
        return stream


@pytest.fixture
def pa(monkeypatch):
    fake = FakePortAudio(["HDA Intel PCH: ALC257 Analog", "HD Pro Webcam C920"], default=0)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr("voiceio.recorder.sd", fake)
    monkeypatch.setattr(feedback, "sd", fake)
    monkeypatch.setattr(feedback, "_stream", None)
    # The real re-init, against the fake module (conftest stubs it out).
    monkeypatch.setattr(audio_devices, "_reinitialize", _REAL_REINITIALIZE)
    return fake


def _recorder(device: str = "default") -> AudioRecorder:
    return AudioRecorder(AudioConfig(sample_rate=16000, device=device, prebuffer_secs=0.5,
                                     auto_stop_silence_secs=0, auto_stop_no_speech_secs=0))


HYPERX = ["HDA Intel PCH: ALC257 Analog", "HD Pro Webcam C920", "HyperX SoloCast"]


def test_dead_stream_reopen_lands_on_the_new_default(pa):
    rec = _recorder()
    rec.open_stream()
    assert rec._stream.device == 0
    pa.plug(HYPERX, default=2)          # USB mic plugged in, made default

    rec.revive_stream()                 # the dead-mic (exact zeros) path

    assert pa.inits == 1
    assert rec._stream.device == 2
    assert not rec._stream.closed and rec._stream.active


def test_reopen_without_reinit_would_keep_the_stale_default(pa, monkeypatch):
    """The bug itself: PortAudio's cached list knows nothing of the new mic."""
    monkeypatch.setattr(audio_devices, "_reinitialize", lambda: None)
    rec = _recorder()
    rec.open_stream()
    pa.plug(HYPERX, default=2)
    rec.reopen_stream()
    assert rec._stream.device == 0


def test_named_device_is_re_resolved_by_name(pa):
    pa.plug(["HyperX SoloCast", "HDA Intel PCH: ALC257 Analog"], default=1)
    pa._snapshot()
    rec = _recorder("HyperX")
    rec.open_stream()
    assert rec._stream.device == 0
    pa.plug(HYPERX, default=0)          # replugged: its index moved

    rec.reopen_stream()

    assert rec._stream.device == 2
    assert pa.query_devices(rec._stream.device)["name"] == "HyperX SoloCast"


def test_cue_stream_closed_before_reinit_and_usable_after(pa):
    assert feedback.open_cue_stream()
    old_cue = feedback._stream
    rec = _recorder()
    rec.open_stream()
    old_mic = rec._stream

    rec.reopen_stream()                 # FakePortAudio asserts nothing is open

    assert pa.inits == 1
    assert old_cue.closed and old_mic.closed
    assert feedback._stream is not None and feedback._stream is not old_cue
    feedback._play_sync("start")
    assert feedback._stream.written > 0


def test_a_closed_cue_stream_stays_closed(pa):
    rec = _recorder()
    rec.open_stream()
    rec.reopen_stream()
    assert pa.inits == 1
    assert feedback._stream is None


def test_reinit_backoff(pa):
    rec = _recorder()
    rec.open_stream()
    pa.plug(HYPERX, default=2)
    rec.revive_stream()
    assert pa.inits == 1
    pa.plug(HYPERX, default=1)
    rec.revive_stream()                 # still dead right after: no second re-init
    assert pa.inits == 1
    assert rec._stream.device == 2      # reopened, on the list it already has
    audio_devices._last_refresh -= audio_devices.REFRESH_MIN_INTERVAL_SECS + 1
    rec.revive_stream()
    assert pa.inits == 2
    assert rec._stream.device == 1


def test_refresh_is_skipped_while_tts_plays(pa):
    playing, done = threading.Event(), threading.Event()

    def tts():
        with audio_devices.stream_in_use():
            stream = pa.OutputStream()
            playing.set()
            done.wait(2)
            stream.close()

    t = threading.Thread(target=tts)
    t.start()
    assert playing.wait(2)
    closed = []
    assert audio_devices.refresh(close=lambda: closed.append(1)) is False
    assert closed == [1]                # the caller's stream still closes
    assert pa.inits == 0
    done.set()
    t.join(2)
    assert audio_devices.refresh() is True
    assert pa.inits == 1


def test_an_open_waits_for_a_refresh_in_progress(pa, monkeypatch):
    monkeypatch.setattr(audio_devices, "_refreshing", True)
    entered = threading.Event()

    def opener():
        with audio_devices.stream_in_use():
            entered.set()

    t = threading.Thread(target=opener, daemon=True)
    t.start()
    assert not entered.wait(0.2)
    with audio_devices._cond:
        audio_devices._refreshing = False
        audio_devices._cond.notify_all()
    assert entered.wait(2)
    t.join(2)


def test_refresh_backoff_with_a_clock(pa):
    now = [1000.0]
    assert audio_devices.refresh(clock=lambda: now[0]) is True
    now[0] += audio_devices.REFRESH_MIN_INTERVAL_SECS - 1
    assert audio_devices.refresh(clock=lambda: now[0]) is False
    now[0] += 2
    assert audio_devices.refresh(clock=lambda: now[0]) is True
    assert pa.inits == 2


def test_a_failed_reinit_does_not_raise_and_next_one_recovers(pa, monkeypatch):
    working = pa._initialize
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Error initializing PortAudio")
        working()

    monkeypatch.setattr(pa, "_initialize", flaky)
    assert audio_devices.refresh() is False
    assert pa._initialized == 0         # terminated, the init failed
    audio_devices._last_refresh = float("-inf")
    assert audio_devices.refresh() is True
    assert pa._initialized == 1


def test_every_open_logs_the_device(pa, caplog):
    rec = _recorder()
    with caplog.at_level(logging.INFO, logger="voiceio.recorder"):
        rec.open_stream()
        pa.plug(HYPERX, default=2)
        rec.reopen_stream()
    opens = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Mic open:")]
    assert opens == [
        "Mic open: 'HDA Intel PCH: ALC257 Analog' (index 0, audio.device = default)",
        "Mic open: 'HyperX SoloCast' (index 2, audio.device = default)",
    ]


def test_a_close_that_raises_propagates_and_skips_the_reinit(pa):
    def broken():
        raise RuntimeError("stop failed")

    with pytest.raises(RuntimeError):
        audio_devices.refresh(close=broken)
    assert pa.inits == 0
    assert audio_devices.refresh() is not None   # not left "refreshing"
    with audio_devices.stream_in_use():
        pass
