"""The persistent cue stream: buffer size and cue timing."""
from __future__ import annotations

import pytest

pytest.importorskip("sounddevice")

from voiceio import feedback  # noqa: E402


class FakeStream:
    opened: list[dict] = []

    def __init__(self, **kwargs):
        FakeStream.opened.append(kwargs)
        latency = kwargs["latency"]
        self.latency = latency if isinstance(latency, float) else 0.035
        self.samplerate = kwargs["samplerate"]

    def start(self):
        pass


@pytest.fixture
def fake_stream(monkeypatch):
    FakeStream.opened = []
    monkeypatch.setattr(feedback.sd, "OutputStream", FakeStream)
    monkeypatch.setattr(feedback, "_stream", None)
    monkeypatch.setattr(feedback, "_sounds", {})
    return FakeStream


def test_cue_stream_has_headroom_against_underruns(fake_stream):
    feedback.warm_up()
    assert fake_stream.opened[0]["latency"] == feedback.CUE_LATENCY_SECS >= 0.05


def test_cue_secs_includes_the_buffer(fake_stream):
    feedback.warm_up()
    played = len(feedback._sounds["start"]) / 44100
    assert feedback.cue_secs("start") == pytest.approx(played + feedback.CUE_LATENCY_SECS)


def test_tts_keeps_the_default_latency(fake_stream):
    """TTS shares open_output_stream; it streams long audio in chunks and is
    left on PortAudio's default."""
    feedback.open_output_stream(samplerate=22050)
    assert fake_stream.opened[0]["latency"] == "high"
