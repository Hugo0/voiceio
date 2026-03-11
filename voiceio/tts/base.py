"""TTS engine protocol and shared types."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from voiceio.backends import ProbeResult


class TTSEngine(Protocol):
    """Interface that all TTS engines implement."""

    name: str

    def probe(self) -> ProbeResult:
        """Check if this engine can work on the current system."""
        ...

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        """Convert text to audio.

        Returns (audio, sample_rate) where audio is mono int16 numpy array.
        """
        ...

    def shutdown(self) -> None:
        """Release resources."""
        ...
