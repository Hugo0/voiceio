"""TTS audio playback with cancellation support."""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

_CHUNK_SIZE = 4096


class TTSPlayer:
    """Plays TTS audio with responsive cancellation."""

    def __init__(self):
        self._cancel = threading.Event()
        self._playing = threading.Event()

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio synchronously. Check cancel between chunks."""
        if audio.size == 0:
            return

        self._cancel.clear()
        self._playing.set()

        # Ensure mono int16
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.int16).reshape(-1, 1)

        try:
            from voiceio.feedback import open_output_stream

            stream = open_output_stream(samplerate=sample_rate)
            if stream is None:
                log.warning("TTS: no audio output device")
                return

            stream.start()
            try:
                offset = 0
                while offset < len(audio) and not self._cancel.is_set():
                    end = min(offset + _CHUNK_SIZE, len(audio))
                    stream.write(audio[offset:end])
                    offset = end
            finally:
                stream.stop()
                stream.close()
        except Exception:
            log.debug("TTS playback error", exc_info=True)
        finally:
            self._playing.clear()

    def cancel(self) -> None:
        """Cancel any ongoing playback."""
        self._cancel.set()

    def is_playing(self) -> bool:
        return self._playing.is_set()
