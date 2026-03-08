from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from voiceio.config import AudioConfig

log = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, cfg: AudioConfig):
        self.sample_rate = cfg.sample_rate
        self.device = None if cfg.device == "default" else cfg.device
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
            log.info("Recording started")

    def stop(self) -> np.ndarray | None:
        with self._lock:
            if not self._recording:
                return None
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False

            if not self._chunks:
                log.warning("No audio captured")
                return None

            audio = np.concatenate(self._chunks, axis=0).flatten()
            duration = len(audio) / self.sample_rate
            log.info("Recording stopped — %.1fs captured", duration)

            if duration < 0.3:
                log.warning("Audio too short (%.1fs), skipping", duration)
                return None

            return audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        if status:
            log.warning("Audio stream status: %s", status)
        self._chunks.append(indata.copy())
