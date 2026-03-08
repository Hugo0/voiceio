from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from voiceio.config import ModelConfig

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: ModelConfig):
        log.info(
            "Loading model '%s' (device=%s, compute_type=%s)...",
            cfg.name,
            cfg.device,
            cfg.compute_type,
        )
        self.language = cfg.language if cfg.language != "auto" else None
        self.model = WhisperModel(
            cfg.name,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        log.info("Model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        text = text.strip()

        if text:
            log.info("Transcribed: %s", text)
        else:
            log.info("No speech detected")

        return text
