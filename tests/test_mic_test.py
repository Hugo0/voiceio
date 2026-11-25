"""The setup/doctor mic test (`_streaming_test`): when it stops listening.

Runs the real loop on a fake clock and a fake sounddevice stream, so a
"delayed audio path" (IP-KVM, remote desktop) is seconds of silence first.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from voiceio.setup import speech

SR = 16000
CHUNK = 0.5


class _Clock:
    """time.time/sleep for the speech module; sleeping feeds the stream."""

    def __init__(self):
        self.now = 0.0
        self.stream = None

    def time(self):
        return self.now

    def sleep(self, secs):
        self.now += secs
        if self.stream is not None:
            self.stream.feed(self.now)


def _fake_sd(clock: _Clock, level, deliver: bool = True):
    """A sounddevice whose InputStream delivers 0.5 s chunks in clock time;
    `level(t)` is the speech amplitude of the audio recorded at t."""

    class InputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback):
            self.cb = callback
            self.sent = 0
            clock.stream = self

        def start(self):
            pass

        def feed(self, now):
            while deliver and (self.sent + 1) * CHUNK <= now:
                t0 = self.sent * CHUNK
                n = int(SR * CHUNK)
                t = t0 + np.arange(n) / SR
                amp = level(t0)
                data = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
                self.cb(data.reshape(-1, 1), n, None, None)
                self.sent += 1

        def stop(self):
            pass

        def close(self):
            pass

    return types.SimpleNamespace(InputStream=InputStream)


class _Model:
    """Recognizes "hello world" in any audio that holds a tone."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **_):
        self.calls += 1
        if np.max(np.abs(audio), initial=0.0) > 1e-3:
            return [types.SimpleNamespace(text="hello world")], None
        return [], None


@pytest.fixture
def run(monkeypatch):
    def _run(level, deliver=True):
        clock = _Clock()
        monkeypatch.setattr(speech, "time", clock)
        monkeypatch.setitem(__import__("sys").modules, "sounddevice",
                            _fake_sd(clock, level, deliver))
        text = speech._streaming_test(model=_Model(), language="en")
        return text, clock.now
    return _run


def _speech(start, length, amp=0.1):
    return lambda t: amp if start <= t < start + length else 0.0


def test_delayed_speech_is_waited_for(run, capsys):
    # The KVM case: 6 s of silence, then 3 s of speech.
    text, ended = run(_speech(6.0, 3.0))
    assert "hello world" in text.lower()
    assert ended >= 9.0 + speech._STOP_SILENCE_SECS - 1.0
    assert ended < 9.0 + speech._STOP_SILENCE_SECS + 2.0
    assert "No speech" not in capsys.readouterr().out


def test_prompt_speech_stops_soon_after_it_ends(run):
    # The normal case is no slower: speech at once, stop ~1.5 s after.
    text, ended = run(_speech(0.0, 2.0))
    assert text
    assert ended < 2.0 + speech._STOP_SILENCE_SECS + 1.5


def test_silence_waits_then_says_nothing_arrived(run, capsys):
    text, ended = run(lambda t: 0.0)
    assert text == ""
    assert speech._WAIT_FOR_SPEECH_SECS <= ended < speech._WAIT_FOR_SPEECH_SECS + 1.0
    out = capsys.readouterr().out
    assert f"No speech arrived within {speech._WAIT_FOR_SPEECH_SECS}s" in out
    assert "IP-KVM" in out and "keep talking" in out


def test_quiet_mic_recognized_words_count_as_speech(run):
    # Below the RMS threshold for its whole length, yet decodable: the words
    # alone keep the test going and end it once they stop changing.
    text, ended = run(_speech(5.0, 3.0, amp=0.005))
    assert text
    assert ended < speech._WAIT_FOR_SPEECH_SECS + speech._MAX_SPEECH_SECS


def test_long_speech_is_capped(run):
    text, ended = run(_speech(1.0, 60.0))
    assert text
    assert ended < 1.0 + speech._MAX_SPEECH_SECS + 1.5


def test_stream_with_no_audio_says_so(run, capsys):
    text, _ = run(lambda t: 0.0, deliver=False)
    assert text == ""
    assert "delivered no audio" in capsys.readouterr().out


@pytest.mark.parametrize("elapsed,heard_at,silent,stop", [
    (1.6, None, 1.6, False),     # initial silence alone never stops it
    (11.9, None, 11.9, False),
    (12.0, None, 12.0, True),
    (7.0, 6.0, 1.0, False),
    (8.0, 6.0, 1.5, True),
    (16.0, 6.0, 0.0, True),      # speech cap
])
def test_should_stop(elapsed, heard_at, silent, stop):
    assert speech._test_should_stop(elapsed, heard_at, silent) is stop


def test_late_voice_suggests_input_latency(run, capsys):
    run(_speech(2.5, 2.0))
    out = capsys.readouterr().out
    assert "voiceio config set audio.input_latency_ms 1900" in out


@pytest.mark.parametrize("start", [0.5, 1.5])   # normal reaction times
def test_prompt_voice_has_no_latency_hint(run, capsys, start):
    run(_speech(start, 2.0))
    assert "input_latency_ms" not in capsys.readouterr().out


def test_no_sound_has_no_latency_hint(run, capsys):
    run(lambda t: 0.0)
    assert "input_latency_ms" not in capsys.readouterr().out


@pytest.mark.parametrize("onset,ms", [
    (None, None), (0.4, None), (1.5, None), (1.8, None),
    (1.84, 1200), (2.06, 1500), (2.6, 2000), (9.0, 5000),
])
def test_latency_hint_value(onset, ms):
    hint = speech._latency_hint(onset)
    if ms is None:
        assert hint is None
    else:
        assert hint.endswith(f"audio.input_latency_ms {ms}")


def test_first_loud_sample_inside_a_block():
    block = np.zeros(SR // 2, dtype=np.float32)
    block[4000:] = 0.1
    assert speech._first_loud_sample(block.reshape(-1, 1), 0.01, SR) == 3840  # its 20 ms frame
