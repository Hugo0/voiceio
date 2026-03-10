"""Audio capture with pre-buffer ring to prevent clipping."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from voiceio.config import AudioConfig

log = logging.getLogger(__name__)


class RingBuffer:
    """Fixed-size ring buffer for float32 audio samples."""

    def __init__(self, max_samples: int):
        self._buf = np.zeros(max_samples, dtype=np.float32)
        self._max = max_samples
        self._write_pos = 0
        self._filled = 0

    def append(self, data: np.ndarray) -> None:
        if self._max == 0:
            return
        flat = data.flatten()
        n = len(flat)
        if n >= self._max:
            # Data larger than buffer: just keep the tail
            self._buf[:] = flat[-self._max:]
            self._write_pos = 0
            self._filled = self._max
            return

        end = self._write_pos + n
        if end <= self._max:
            self._buf[self._write_pos:end] = flat
        else:
            first = self._max - self._write_pos
            self._buf[self._write_pos:] = flat[:first]
            self._buf[:n - first] = flat[first:]

        self._write_pos = end % self._max
        self._filled = min(self._filled + n, self._max)

    def get(self) -> np.ndarray:
        """Return buffered audio in chronological order."""
        if self._filled == 0:
            return np.zeros(0, dtype=np.float32)
        if self._filled < self._max:
            return self._buf[:self._filled].copy()
        # Full ring: read from write_pos (oldest) through the end
        return np.concatenate([
            self._buf[self._write_pos:],
            self._buf[:self._write_pos],
        ])

    def clear(self) -> None:
        self._write_pos = 0
        self._filled = 0


class AudioRecorder:
    """Audio recorder with always-on pre-buffer ring.

    The audio stream runs continuously. A ring buffer captures the last
    `prebuffer_secs` of audio. When recording starts, the ring buffer
    contents become the start of the recording, so no first syllable is lost.
    """

    def __init__(self, cfg: AudioConfig, on_speech_pause: Callable[[], None] | None = None):
        self.sample_rate = cfg.sample_rate
        self.device = None if cfg.device == "default" else cfg.device
        self.prebuffer_secs = cfg.prebuffer_secs

        self._ring = RingBuffer(int(self.prebuffer_secs * self.sample_rate))
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

        # Streaming VAD
        self._on_speech_pause = on_speech_pause
        self._silence_threshold = cfg.silence_threshold
        self._silence_duration = cfg.silence_duration
        self._silent_chunks = 0.0
        self._last_transcribed_len = 0
        self._total_samples = 0

        # Auto-stop on sustained silence
        self._auto_stop_secs = cfg.auto_stop_silence_secs
        self._sustained_silence = 0.0
        self._heard_speech = False
        self._on_auto_stop: Callable[[], None] | None = None

    def open_stream(self) -> None:
        """Start the always-on audio stream (feeds ring buffer)."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        log.debug("Audio stream opened (prebuffer=%.1fs)", self.prebuffer_secs)

    def close_stream(self) -> None:
        """Stop the always-on audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._ring.clear()

    def start(self) -> None:
        """Start recording. Grabs ring buffer contents as the beginning."""
        with self._lock:
            if self._recording:
                return
            # Ensure stream is running
            if self._stream is None:
                self.open_stream()
            # Grab pre-buffer
            prebuf = self._ring.get()
            self._chunks = [prebuf.reshape(-1, 1)] if len(prebuf) > 0 else []
            self._total_samples = sum(len(c) for c in self._chunks)
            self._silent_chunks = 0.0
            self._sustained_silence = 0.0
            self._heard_speech = False
            self._last_transcribed_len = 0
            self._recording = True
            prebuf_ms = len(prebuf) / self.sample_rate * 1000
            log.info("Recording started (%.0fms pre-buffer)", prebuf_ms)

    def stop(self) -> np.ndarray | None:
        """Stop recording, return captured audio."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False

            if not self._chunks:
                log.warning("No audio captured")
                return None

            audio = np.concatenate(self._chunks, axis=0).flatten()
            remaining = audio[self._last_transcribed_len:]
            duration = len(remaining) / self.sample_rate

            if duration < 0.3:
                if self._last_transcribed_len > 0:
                    return None
                log.warning("Audio too short (%.1fs), skipping", duration)
                return None

            log.info("Recording stopped, %.1fs audio", duration)
            self._chunks = []
            return remaining

    def get_audio_so_far(self) -> np.ndarray | None:
        """Get all audio captured so far (for streaming)."""
        with self._lock:
            if not self._chunks:
                return None
            return np.concatenate(self._chunks, axis=0).flatten()

    def set_on_speech_pause(self, callback: Callable[[], None] | None) -> None:
        """Set/clear the speech pause callback (used by streaming session)."""
        self._on_speech_pause = callback

    def set_on_auto_stop(self, callback: Callable[[], None] | None) -> None:
        """Set/clear the auto-stop callback (fires after sustained silence)."""
        self._on_auto_stop = callback

    def mark_transcribed(self, num_samples: int) -> None:
        self._last_transcribed_len = num_samples

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        if status:
            log.warning("Audio stream status: %s", status)

        # Always feed ring buffer
        self._ring.append(indata)

        # Only collect chunks when recording
        if not self._recording:
            return

        chunk = indata.copy()
        self._chunks.append(chunk)
        self._total_samples += chunk.shape[0]

        # Silence detection
        flat = indata.ravel()
        rms = float(np.sqrt(np.dot(flat, flat) / len(flat)))
        chunk_secs = frames / self.sample_rate
        is_silent = rms < self._silence_threshold

        if is_silent:
            self._silent_chunks += chunk_secs
            self._sustained_silence += chunk_secs
        else:
            self._silent_chunks = 0.0
            self._sustained_silence = 0.0
            self._heard_speech = True

        # Streaming VAD: trigger transcription on speech pause
        if self._on_speech_pause is not None:
            has_new = self._total_samples > self._last_transcribed_len + self.sample_rate
            if self._silent_chunks >= self._silence_duration and has_new:
                self._silent_chunks = 0.0
                self._on_speech_pause()

        # Auto-stop after sustained silence (only after hearing speech)
        if (self._on_auto_stop is not None
                and self._auto_stop_secs > 0
                and self._heard_speech
                and self._sustained_silence >= self._auto_stop_secs):
            # Capture and clear callback before firing to prevent
            # the next audio chunk from re-triggering (single-fire)
            cb = self._on_auto_stop
            self._on_auto_stop = None
            self._sustained_silence = 0.0
            log.info("Auto-stopping after %.0fs of silence", self._auto_stop_secs)
            cb()
