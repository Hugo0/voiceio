"""Where streaming transcription gets its samples from.

`StreamingSession` never cared *how* audio was captured — it only ever touched
three members of the recorder it was handed: the sample rate, a speech-pause
hook, and "give me everything captured so far". `AudioSource` names that trio,
so the streaming machinery is no longer tied to a microphone.

Two implementations satisfy it:

- `voiceio.recorder.AudioRecorder` — pulls from a sound device (desktop).
  It already matched this shape; nothing about it changed.
- `PushAudioSource` — the inverse: samples are *pushed in* from somewhere
  else. A WebSocket feeding PCM from a phone, a WAV file in a test, an
  upstream service. Needs no audio hardware, so it imports on a headless
  server where `sounddevice` cannot even load.

Keeping this protocol in its own module (rather than importing `AudioRecorder`
for the type) is what lets `voiceio.streaming` be imported with no audio stack
present at all.
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol, runtime_checkable

import numpy as np

__all__ = ["AudioSource", "PushAudioSource", "pcm16_to_float32"]

# The only sample rate Whisper accepts. Everything upstream resamples to it.
SAMPLE_RATE = 16000


@runtime_checkable
class AudioSource(Protocol):
    """The audio surface `StreamingSession` actually uses."""

    sample_rate: int

    def get_audio_so_far(self) -> np.ndarray | None:
        """Everything captured this session as float32 mono, or None if
        nothing has arrived yet."""
        ...

    def set_on_speech_pause(self, callback: Callable[[], None] | None) -> None:
        """Install (or clear, with None) the hook fired when the speaker
        pauses. A source with no voice-activity detection may never call it —
        the streaming worker also wakes on a timer, so pauses only make interim
        results *sooner*, they are never required for correctness."""
        ...


class AudioBuffer:
    """Growable mono float32 buffer: O(1) amortized append, O(1) read.

    `view()` returns the samples so far without copying. Appends never move
    data a previous view points at (growth allocates a new array; the old
    view keeps the old one alive), so a reader can decode a view while the
    audio callback keeps writing. Not thread-safe by itself; callers lock.
    """

    def __init__(self, initial_samples: int = SAMPLE_RATE * 60):
        self._buf = np.empty(max(1, initial_samples), dtype=np.float32)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def append(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        end = self._n + len(samples)
        if end > len(self._buf):
            grown = np.empty(max(end, 2 * len(self._buf)), dtype=np.float32)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n:end] = samples
        self._n = end

    def view(self) -> np.ndarray:
        return self._buf[:self._n]

    def clear(self) -> None:
        self._n = 0
        # Fresh array: a view handed out earlier must not see overwrites.
        self._buf = np.empty(len(self._buf), dtype=np.float32)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM → float32 in [-1, 1).

    Divides by 32768 rather than 32767: it is the exact inverse of the scaling
    every encoder uses, so a round-trip is bit-exact and -32768 cannot overflow
    past -1.0.
    """
    if len(data) % 2:  # a frame split across two WebSocket messages
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class PushAudioSource:
    """An `AudioSource` fed from outside, instead of pulling from a device.

    Thread-safe by design: the producer (a WebSocket read loop) and the
    consumer (the streaming worker thread) run concurrently and neither may
    block the other for long. Chunks are appended to a list under a lock and
    only concatenated when actually read, so `feed()` stays O(1) and the
    per-second interim decode pays the copy.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._buffer = AudioBuffer(sample_rate * 60)
        self._lock = threading.Lock()
        self._on_speech_pause: Callable[[], None] | None = None
        self._closed = False
        # A 16-bit sample split across two network frames: its first byte
        # waits here. Dropping it would shift every later sample by one byte
        # and turn the rest of the stream into noise.
        self._carry = b""

    # ── producer side ────────────────────────────────────────────────────
    def feed(self, chunk: np.ndarray | bytes) -> int:
        """Append audio. Accepts raw PCM16-LE bytes or a float32 array.

        Returns the total sample count so far, so a caller can enforce a
        duration cap without reaching into the buffer.
        """
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            data = self._carry + bytes(chunk)
            cut = len(data) - len(data) % 2
            self._carry = data[cut:]
            samples = pcm16_to_float32(data[:cut])
        else:
            samples = np.asarray(chunk, dtype=np.float32)
        if samples.ndim > 1:  # interleaved stereo → mono
            samples = samples.mean(axis=1, dtype=np.float32)
        with self._lock:
            if len(samples) and not self._closed:
                self._buffer.append(samples)
            return len(self._buffer)

    def notify_speech_pause(self) -> None:
        """Tell the consumer a pause happened, so it can decode now rather
        than waiting out its timer. Safe to call when nobody is listening."""
        callback = self._on_speech_pause
        if callback is not None:
            callback()

    def close(self) -> None:
        """Refuse further audio. Already-buffered samples stay readable so the
        final decode still sees the complete utterance."""
        with self._lock:
            self._closed = True

    # ── consumer side (the AudioSource protocol) ─────────────────────────
    def get_audio_so_far(self) -> np.ndarray | None:
        with self._lock:
            return self._buffer.view() if len(self._buffer) else None

    def set_on_speech_pause(self, callback: Callable[[], None] | None) -> None:
        self._on_speech_pause = callback

    # ── introspection ────────────────────────────────────────────────────
    @property
    def duration_secs(self) -> float:
        with self._lock:
            return len(self._buffer) / self.sample_rate

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._buffer)
