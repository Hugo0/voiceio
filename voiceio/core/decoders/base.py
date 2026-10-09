"""The decoder contract.

A decoder turns 16 kHz mono float32 audio into text. What differs between
engines — and what the rest of voiceio adapts to — is declared in
`Capabilities`, above all how the engine can be *biased* toward the user's
vocabulary, which is the axis that matters for jargon-heavy dictation:

- Whisper: hotwords + prompt, sharing a hard ~223-token budget
- sherpa-onnx transducers (modified beam search): an unbounded hotword list
- LLM-decoder ASR (Qwen3-ASR, Granite): free-text context

Biasing is passed *per request* (`Bias`), never stored on the decoder, so two
concurrent sessions can't see each other's vocabulary or prompt.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16000


class BiasMode(enum.Flag):
    NONE = 0
    PROMPT = enum.auto()      # free-text conditioning (Whisper initial_prompt, LLM context)
    HOTWORDS = enum.auto()    # a term list inside a token budget
    UNBOUNDED = enum.auto()   # a term list with no practical size limit


@dataclass(frozen=True)
class Capabilities:
    bias: BiasMode = BiasMode.NONE
    # Token budget for hotwords, when bounded. The vocabulary ranker fills it.
    max_bias_tokens: int | None = None
    # Decodes audio incrementally itself (no re-decoding of a growing tail).
    native_streaming: bool = False


@dataclass(frozen=True)
class Bias:
    """How to steer one decode. All fields optional."""
    hotwords: str | None = None   # rendered term list, already fit to the budget
    prompt: str | None = None     # e.g. recent transcripts
    context: str | None = None    # e.g. already-frozen text preceding this audio

    def with_context(self, context: str | None) -> Bias:
        return Bias(self.hotwords, self.prompt, context)


@dataclass(frozen=True)
class DecodeResult:
    text: str
    segments: tuple[dict, ...] = ()
    secs: float = 0.0


class DecodeError(RuntimeError):
    """The decode failed (timeout, crash, bad output). Distinct from a
    legitimate empty result — "" means silence; lost audio must never be
    mistaken for it."""


@runtime_checkable
class Decoder(Protocol):
    name: str
    capabilities: Capabilities

    def decode(self, audio: np.ndarray, *, final: bool = False,
               bias: Bias = Bias()) -> DecodeResult:
        """`final=True` marks a pass whose text the user keeps (beam search
        where supported); interim passes favour latency. Raises DecodeError."""
        ...

    def healthy(self) -> bool: ...

    def recover(self) -> None:
        """Make a dead decoder usable again (e.g. restart its worker).
        Raises if it can't — the caller decides whether that is fatal."""
        ...

    def close(self) -> None: ...
