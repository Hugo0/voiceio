"""A mic stream that dies mid-recording is reopened, not recorded to the end.

The real failure: after a suspend/resume the stream delivered 2 s of audio,
then 102 s of exact zeros; the whole dictation was lost.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from voiceio.config import AudioConfig
from voiceio.recorder import AudioRecorder


def _recorder() -> AudioRecorder:
    rec = AudioRecorder(AudioConfig(sample_rate=16000, prebuffer_secs=0.5,
                                    auto_stop_silence_secs=0, auto_stop_no_speech_secs=0))
    rec._stream = MagicMock()
    return rec


def _feed(rec: AudioRecorder, value: float, secs: float) -> None:
    for _ in range(int(secs * 16000 / 1600)):
        rec._callback(np.full((1600, 1), value, dtype=np.float32), 1600, None, None)


def _armed(rec: AudioRecorder) -> threading.Event:
    fired = threading.Event()
    rec.set_on_dead_stream(fired.set)
    return fired


def test_zeros_after_real_audio_fire_once():
    rec = _recorder()
    rec.start()
    fired = _armed(rec)
    _feed(rec, 0.02, 1.0)
    _feed(rec, 0.0, 1.2)
    assert fired.wait(2)
    fired.clear()
    _feed(rec, 0.0, 1.2)            # still reviving: not fired again
    assert not fired.wait(0.2)


def test_a_mic_silent_from_the_start_is_left_to_the_muted_check():
    rec = _recorder()
    rec.start()
    fired = _armed(rec)
    _feed(rec, 0.0, 2.0)
    assert not fired.wait(0.2)


def test_quiet_room_noise_is_not_a_dead_stream():
    rec = _recorder()
    rec.start()
    fired = _armed(rec)
    _feed(rec, 0.02, 0.5)
    _feed(rec, 0.0005, 3.0)        # a working mic's silence is never exact zero
    assert not fired.wait(0.2)


def test_revive_reopens_and_allows_at_most_two_attempts():
    rec = _recorder()
    rec.start()
    with patch.object(rec, "open_stream") as opened, patch.object(rec, "close_stream"):
        for expected in (True, True, False):
            fired = _armed(rec)
            _feed(rec, 0.02, 0.2)
            _feed(rec, 0.0, 1.2)
            assert fired.wait(0.5) is expected
            if expected:
                rec.revive_stream()
        assert opened.call_count == 2
    assert rec.is_recording


def test_daemon_revives_and_warns_only_when_still_dead():
    from voiceio.app import VoiceIO, _State
    vio = VoiceIO.__new__(VoiceIO)
    vio._hotkey_lock = threading.Lock()
    vio._state = _State.RECORDING
    vio._recording = 7
    vio.recorder = MagicMock(silent_signal_secs=0.0)
    with patch("voiceio.app.time.sleep"), patch("voiceio.feedback.notify") as notify:
        vio._revive_mic(7)
        vio.recorder.revive_stream.assert_called_once()
        notify.assert_not_called()
        vio.recorder.silent_signal_secs = 1.5
        vio._revive_mic(7)
        notify.assert_called_once()
        vio._revive_mic(8)              # a later recording: not ours to touch
        assert vio.recorder.revive_stream.call_count == 2
