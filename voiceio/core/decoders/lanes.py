"""Route interim and final decodes to separate decoder instances.

A single decoder serializes everything: when the user presses stop, the final
pass queues behind whatever interim preview happens to be decoding (measured
p50 2.2s, p90 10.4s of pure waiting). With a dedicated final lane, stop never
waits on a preview — the in-flight interim is simply abandoned.
"""
from __future__ import annotations

import numpy as np

from voiceio.core.decoders.base import Bias, Capabilities, DecodeResult, Decoder


class DualLane:
    """A `Decoder` composed of a draft lane (interims) and a final lane
    (freeze + final passes). Both must load the same model, so biasing and
    output are interchangeable."""

    def __init__(self, draft: Decoder, final: Decoder):
        self.draft = draft
        self.final = final
        self.name = f"{final.name}+draft"

    @property
    def capabilities(self) -> Capabilities:
        return self.final.capabilities

    def decode(self, audio: np.ndarray, *, final: bool = False,
               bias: Bias = Bias()) -> DecodeResult:
        lane = self.final if final else self.draft
        return lane.decode(audio, final=final, bias=bias)

    def healthy(self) -> bool:
        return self.final.healthy() and self.draft.healthy()

    def recover(self) -> None:
        self.final.recover()
        self.draft.recover()

    def close(self) -> None:
        self.draft.close()
        self.final.close()
