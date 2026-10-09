"""Known speech models and what each can do.

The default is chosen on the user's own audio (`voiceio models bench`), not
from leaderboards: what matters for dictation full of names and jargon is
*biasing* — how the model can be steered toward the user's vocabulary — and
leaderboard WER says nothing about it. Every entry states its license; the
defaults are permissive. voiceio never ships weights: they are fetched from
the model's own source on first use, under that model's license.
"""
from __future__ import annotations

from dataclasses import dataclass

from voiceio.core.decoders.base import BiasMode


@dataclass(frozen=True)
class ModelInfo:
    name: str               # what goes in `[model] name`
    runtime: str            # "faster-whisper" | "sherpa-onnx" | "whisper.cpp"
    params: str
    license: str
    bias: BiasMode
    streaming: bool = False
    languages: str = "multilingual"
    notes: str = ""
    experimental: bool = False


# Only models an adapter in voiceio.core.decoders can actually run. Next up,
# once sherpa-onnx ships its streaming modified_beam_search fix (PR #3895,
# after v1.13.8): Nemotron streaming via OnlineRecognizer (true streaming +
# unlimited hotwords), and Qwen3-ASR (free-text context biasing).
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo("small", "faster-whisper", "244M", "MIT", BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="default: ~5x realtime on a laptop CPU; hotwords cap ~35 terms"),
    ModelInfo("medium", "faster-whisper", "769M", "MIT", BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="better on names, ~2.5x slower than small"),
    ModelInfo("large-v3-turbo", "faster-whisper", "809M", "MIT",
              BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="best Whisper for dictation; GPU or whisper.cpp+Vulkan recommended; "
                    "also the default learn teacher"),
    ModelInfo("distil-large-v3", "faster-whisper", "756M", "MIT",
              BiasMode.HOTWORDS | BiasMode.PROMPT, languages="en"),
    ModelInfo("sherpa:parakeet-tdt-0.6b-v3", "sherpa-onnx", "600M", "CC-BY-4.0",
              BiasMode.NONE, notes="fast (≈12x realtime CPU); hotwords need offline MBS, "
                                   "broken until sherpa-onnx PR #3657", experimental=True),
    ModelInfo("whispercpp:http://127.0.0.1:8080", "whisper.cpp", "any GGUF", "per model",
              BiasMode.PROMPT, notes="a whisper.cpp server, e.g. a Vulkan build on an AMD "
                                     "iGPU; no hotwords, prompt only", experimental=True),
)


def info(name: str) -> ModelInfo | None:
    """Catalog entry for a model name (custom paths and fine-tunes return None)."""
    for m in CATALOG:
        if m.name == name or (m.name.startswith("whispercpp:") and name.startswith("whispercpp:")):
            return m
    return None
