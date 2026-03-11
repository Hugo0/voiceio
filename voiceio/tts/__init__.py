"""Text-to-speech engine package.

Public API:
    tts.select(cfg) -> TTSEngine   — pick best available engine
    tts.probe_all(cfg) -> list      — probe all engines for doctor
"""
from __future__ import annotations

from voiceio.tts.chain import probe_all, select

__all__ = ["select", "probe_all"]
