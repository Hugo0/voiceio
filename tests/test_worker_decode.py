"""Hotwords must never be combined with the sampling fallback."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from voiceio.core.decoders.faster_whisper import worker


def _seg(text="ok", ratio=1.5, logprob=-0.3, no_speech=0.01):
    return SimpleNamespace(text=text, compression_ratio=ratio, avg_logprob=logprob,
                           no_speech_prob=no_speech, tokens=[1, 2])


class FakeModel:
    def __init__(self, with_hotwords, without_hotwords):
        self.calls: list[dict] = []
        self._with, self._without = with_hotwords, without_hotwords
        self.hf_tokenizer = SimpleNamespace(
            encode=lambda s, add_special_tokens=False: SimpleNamespace(ids=s.split()))

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(self._with if kw.get("hotwords") else self._without), None


AUDIO = np.zeros(16000 * 5, dtype=np.float32)


def test_hotwords_decode_greedily_and_keep_a_clean_result():
    m = FakeModel([_seg("Kalshi")], [_seg("kalchi")])
    segs = worker.decode(m, AUDIO, {"hotwords": "Kalshi", "max_new_tokens": 80})
    assert [s.text for s in segs] == ["Kalshi"]
    assert len(m.calls) == 1 and m.calls[0]["temperature"] == 0.0


def test_a_looping_hotwords_decode_is_redone_without_them():
    m = FakeModel([_seg("go go go go go go", ratio=6.0)], [_seg("go")])
    segs = worker.decode(m, AUDIO, {"hotwords": "Kalshi", "max_new_tokens": 80})
    assert [s.text for s in segs] == ["go"]
    retry = m.calls[1]
    assert retry["hotwords"] is None and "temperature" not in retry  # library fallback


def test_low_confidence_silence_is_not_a_failure():
    m = FakeModel([_seg("", logprob=-1.5, no_speech=0.9)], [_seg("x")])
    worker.decode(m, AUDIO, {"hotwords": "Kalshi"})
    assert len(m.calls) == 1


def test_without_hotwords_the_library_fallback_is_untouched():
    m = FakeModel([], [_seg("hello")])
    worker.decode(m, AUDIO, {"hotwords": None})
    assert len(m.calls) == 1 and "temperature" not in m.calls[0]


def test_a_loop_spread_over_segments_is_redone_without_hotwords():
    loop = [_seg("because it should not work out in the complex world")] * 4
    m = FakeModel(loop, [_seg("because it should not work out")])
    segs = worker.decode(m, AUDIO, {"hotwords": "Kalshi"})
    assert [s.text for s in segs] == ["because it should not work out"]
