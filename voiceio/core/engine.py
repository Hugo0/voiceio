"""The dictation engine: one decoder plus the user's personalization.

Every front end composes this rather than wiring decoder, vocabulary,
corrections and pipeline itself: the desktop daemon (microphone → typer), the
server (socket → JSON), evaluation and the learning loop (files → metrics).
That is what keeps them from drifting apart — the phone path once ran with no
vocabulary, no corrections and no prompt while the desktop had all three.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from voiceio.core.decoders.base import Bias, Decoder

if TYPE_CHECKING:
    from voiceio.audio_source import AudioSource
    from voiceio.commands import CommandProcessor
    from voiceio.config import Config
    from voiceio.core.bias import BiasProvider
    from voiceio.core.pipeline import Pipeline
    from voiceio.corrections import CorrectionDict
    from voiceio.llm import LLMProcessor
    from voiceio.postcorrect import PostCorrector
    from voiceio.streaming import StreamingSession
    from voiceio.typers.base import TyperBackend

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str                       # after the pipeline
    raw: str                        # decoder output
    segments: tuple[dict, ...] = ()
    latency: dict = field(default_factory=dict)
    abort: bool = False             # a voice command discarded the utterance


class Engine:
    def __init__(
        self,
        cfg: Config,
        decoder: Decoder,
        *,
        pipeline: Pipeline,
        bias: BiasProvider,
        corrections: CorrectionDict,
    ):
        self.cfg = cfg
        self.decoder = decoder
        self.pipeline = pipeline
        self.corrections = corrections
        self._bias = bias

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        decoder: Decoder | None = None,
        lanes: int | None = None,
        commands: CommandProcessor | None = None,
        postcorrect: PostCorrector | None = None,
        llm: LLMProcessor | None = None,
        prefix: bool = True,
    ) -> Engine:
        """Load the configured model (unless `decoder` is given) and the
        user's vocabulary, corrections and pipeline. Front ends decide which
        cursor-bound (commands) or cloud (postcorrect, llm) stages they want."""
        from voiceio.core.bias import BiasProvider
        from voiceio.core.decoders import build_decoder
        from voiceio.core.pipeline import Pipeline
        from voiceio.corrections import CorrectionDict
        from voiceio.prompt import PromptBuilder
        from voiceio.vocabulary import VocabularyLoader

        corrections = CorrectionDict()
        bias = BiasProvider(
            VocabularyLoader(cfg.model), corrections, PromptBuilder(),
            language=cfg.model.language,
        )
        pipeline = Pipeline.from_config(
            cfg, corrections=corrections, commands=commands,
            postcorrect=postcorrect, llm=llm, prefix=prefix,
        )
        if decoder is None:
            decoder = build_decoder(cfg.model, lanes=lanes)
        return cls(cfg, decoder, pipeline=pipeline, bias=bias, corrections=corrections)

    # ── personalization ──────────────────────────────────────────────────
    @property
    def vocabulary_terms(self) -> list[str]:
        return self._bias.vocabulary.get_all()

    def recent(self, n: int = 3) -> list[str]:
        return self._bias.recent(n)

    def bias(self) -> Bias:
        """Biasing for the next recording/request. Corrections and vocabulary
        are mtime-cached, so edits apply without a restart."""
        self.corrections.load()
        return self._bias.bias_for(self.decoder.capabilities)

    def record_usage(self, text: str) -> None:
        """Feed a committed utterance back: prompt history + vocab usage."""
        self._bias.record_usage(text)

    # ── transcription ────────────────────────────────────────────────────
    def transcribe(self, audio: np.ndarray, *, bias: Bias | None = None) -> Transcript:
        """One-shot: decode (final quality) and run the pipeline. Raises
        DecodeError if the decode fails — never returns lost audio as ""."""
        t0 = time.monotonic()
        result = self.decoder.decode(audio, final=True, bias=bias or self.bias())
        latency = {
            "audio_secs": round(len(audio) / 16000, 2),
            "transcribe": round(time.monotonic() - t0, 3),
        }
        if not result.text:
            return Transcript("", "", result.segments, latency)
        t1 = time.monotonic()
        processed = self.pipeline(result.text, final=True)
        latency["pipeline"] = round(time.monotonic() - t1, 3)
        if "postcorrect" in processed.timings:
            latency["postcorrect"] = processed.timings["postcorrect"]
        return Transcript(processed.text, result.text, result.segments, latency,
                          abort=processed.abort)

    def session(
        self,
        source: AudioSource,
        typer: TyperBackend,
        *,
        bias: Bias | None = None,
        generation: int = 0,
        on_interim: Callable[[str], None] | None = None,
        on_typer_broken: Callable[[], None] | None = None,
    ) -> StreamingSession:
        """A streaming session reading `source` and writing to `typer`."""
        from voiceio.streaming import StreamingSession
        return StreamingSession(
            self.decoder, typer, source,
            generation=generation,
            pipeline=self.pipeline,
            on_interim=on_interim,
            on_typer_broken=on_typer_broken,
            freeze_secs=self.cfg.output.streaming_freeze_secs,
            bias=bias or self.bias(),
        )

    def close(self) -> None:
        self.decoder.close()
