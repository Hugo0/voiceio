"""Tests for the push-fed audio source and the collecting typer.

These are the two shims that let StreamingSession run without a desktop, so
what matters is that they satisfy the protocols the session type-checks
against, and that the PCM conversion is exact.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from voiceio.audio_source import (
    AudioSource,
    PushAudioSource,
    pcm16_to_float32,
)
from voiceio.typers.base import StreamingTyper, TyperBackend
from voiceio.typers.collecting import CollectingTyper


# ── pcm16_to_float32 ─────────────────────────────────────────────────────

def test_pcm16_roundtrip_is_exact():
    original = np.array([0, 1000, -1000, 32767, -32768], dtype="<i2")
    out = pcm16_to_float32(original.tobytes())
    assert np.allclose(out, original.astype(np.float32) / 32768.0)
    # Nothing escapes [-1, 1) — the whole point of dividing by 32768.
    assert out.max() < 1.0
    assert out.min() >= -1.0


def test_pcm16_odd_length_drops_partial_frame():
    """A 16-bit frame split across two WebSocket messages must not blow up."""
    data = np.array([100, 200], dtype="<i2").tobytes() + b"\x01"
    out = pcm16_to_float32(data)
    assert len(out) == 2


def test_pcm16_empty():
    assert len(pcm16_to_float32(b"")) == 0


# ── PushAudioSource ──────────────────────────────────────────────────────

def test_satisfies_audio_source_protocol():
    assert isinstance(PushAudioSource(), AudioSource)


def test_empty_source_returns_none():
    src = PushAudioSource()
    assert src.get_audio_so_far() is None
    assert src.duration_secs == 0.0


def test_feed_bytes_accumulates_in_order():
    src = PushAudioSource(16000)
    a = np.array([1000, 2000], dtype="<i2")
    b = np.array([3000, 4000], dtype="<i2")
    src.feed(a.tobytes())
    src.feed(b.tobytes())
    out = src.get_audio_so_far()
    assert len(out) == 4
    expected = np.concatenate([a, b]).astype(np.float32) / 32768.0
    assert np.allclose(out, expected)


def test_feed_float_array():
    src = PushAudioSource()
    src.feed(np.array([0.1, 0.2], dtype=np.float32))
    assert np.allclose(src.get_audio_so_far(), [0.1, 0.2])


def test_feed_returns_running_total_for_duration_caps():
    src = PushAudioSource(16000)
    chunk = np.zeros(1600, dtype="<i2").tobytes()  # 0.1s
    assert src.feed(chunk) == 1600
    assert src.feed(chunk) == 3200
    assert src.duration_secs == pytest.approx(0.2)


def test_stereo_is_downmixed():
    src = PushAudioSource()
    src.feed(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.allclose(src.get_audio_so_far(), [0.5, 0.5])


def test_close_refuses_new_audio_but_keeps_the_old():
    src = PushAudioSource(16000)
    src.feed(np.zeros(800, dtype="<i2").tobytes())
    src.close()
    src.feed(np.zeros(800, dtype="<i2").tobytes())
    # The final decode must still see everything captured before the close.
    assert src.n_samples == 800
    assert len(src.get_audio_so_far()) == 800


def test_speech_pause_callback_is_optional():
    src = PushAudioSource()
    src.notify_speech_pause()  # nobody listening — must not raise

    fired = []
    src.set_on_speech_pause(lambda: fired.append(1))
    src.notify_speech_pause()
    assert fired == [1]

    src.set_on_speech_pause(None)
    src.notify_speech_pause()
    assert fired == [1]


def test_concurrent_feed_and_read_lose_no_samples():
    """The producer (socket) and consumer (decode thread) really do run at
    once; a lost or torn chunk would show up as dropped words."""
    src = PushAudioSource(16000)
    chunk = np.ones(160, dtype="<i2").tobytes()
    n_writers, per_writer = 4, 50
    errors = []

    def write():
        try:
            for _ in range(per_writer):
                src.feed(chunk)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def read():
        try:
            for _ in range(100):
                src.get_audio_so_far()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(n_writers)]
    threads.append(threading.Thread(target=read))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert src.n_samples == n_writers * per_writer * 160
    assert len(src.get_audio_so_far()) == src.n_samples


# ── CollectingTyper ──────────────────────────────────────────────────────

def test_collecting_typer_satisfies_protocols():
    typer = CollectingTyper()
    assert isinstance(typer, TyperBackend)
    assert isinstance(typer, StreamingTyper)
    assert typer.probe().ok


def test_preedit_is_visible_but_not_committed():
    typer = CollectingTyper()
    typer.update_preedit("hello wor")
    assert typer.text == "hello wor"
    assert typer.committed == ""
    typer.commit_text("hello world")
    assert typer.text == "hello world"
    assert typer.committed == "hello world"


def test_clear_preedit_discards_only_preedit():
    typer = CollectingTyper()
    typer.type_text("kept ")
    typer.update_preedit("dropped")
    typer.clear_preedit()
    assert typer.text == "kept "


def test_delete_chars_clamps_and_ignores_nonpositive():
    typer = CollectingTyper()
    typer.type_text("hello")
    typer.delete_chars(0)
    assert typer.text == "hello"
    typer.delete_chars(2)
    assert typer.text == "hel"
    typer.delete_chars(99)  # more than exists — must not raise
    assert typer.text == ""


def test_reset():
    typer = CollectingTyper()
    typer.type_text("a")
    typer.update_preedit("b")
    typer.reset()
    assert typer.text == ""
