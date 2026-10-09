"""Shared test doubles."""
from __future__ import annotations

from unittest.mock import MagicMock

from voiceio.core.decoders.base import Bias, BiasMode, Capabilities, DecodeResult


class FakeDecoder(MagicMock):
    """A `Decoder` scripted through a mocked ``transcribe``.

    Tests set ``fake.transcribe.return_value`` / ``side_effect`` exactly as
    they would on a MagicMock; ``decode()`` calls it with
    ``(audio, final=..., context=...)`` and wraps the text, attaching
    ``last_segments`` as the result's segments.
    """

    capabilities = Capabilities(bias=BiasMode.PROMPT | BiasMode.HOTWORDS, max_bias_tokens=120)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_segments = []
        self.biases: list[Bias] = []

    def decode(self, audio, *, final=False, bias=Bias()):
        self.biases.append(bias)
        text = self.transcribe(audio, final=final, context=bias.context)
        return DecodeResult(text, tuple(self.last_segments))
