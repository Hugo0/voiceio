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

WHISPER_DEFAULT = "small"
PARAKEET = "parakeet-tdt-0.6b-v2"


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
    # Models voiceio downloads itself (sherpa-onnx): a release tarball, its
    # sha256, and its size. faster-whisper models come from the HF cache.
    url: str = ""
    sha256: str = ""
    size_mb: int = 0


_SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# Only models an adapter in voiceio.core.decoders can actually run. Next up,
# once sherpa-onnx ships its streaming modified_beam_search fix (PR #3895,
# after v1.13.8): Nemotron streaming via OnlineRecognizer (true streaming +
# unlimited hotwords), and Qwen3-ASR (free-text context biasing).
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(PARAKEET, "sherpa-onnx", "600M", "CC-BY-4.0", BiasMode.UNBOUNDED, languages="en",
              notes="default when sherpa-onnx has the NeMo TDT beam-search fix: ~2x faster "
                    "than whisper-small, every proper noun of your vocabulary as a hotword",
              url=f"{_SHERPA_RELEASES}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
              sha256="157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
              size_mb=482),
    ModelInfo("small", "faster-whisper", "244M", "MIT", BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="~5x realtime on a laptop CPU; hotwords cap ~35 terms; "
                    "the model `voiceio learn` fine-tunes"),
    ModelInfo("medium", "faster-whisper", "769M", "MIT", BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="better on names, ~2.5x slower than small"),
    ModelInfo("large-v3-turbo", "faster-whisper", "809M", "MIT",
              BiasMode.HOTWORDS | BiasMode.PROMPT,
              notes="best Whisper for dictation; GPU or whisper.cpp+Vulkan recommended; "
                    "also the default learn teacher"),
    ModelInfo("distil-large-v3", "faster-whisper", "756M", "MIT",
              BiasMode.HOTWORDS | BiasMode.PROMPT, languages="en"),
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


def runtime_of(name: str) -> str:
    """The runtime that decodes `[model] name`: anything that isn't a sherpa
    or whisper.cpp model is faster-whisper (a size, HF repo, or CTranslate2
    path such as a `voiceio learn` fine-tune)."""
    if name.startswith("sherpa:"):
        return "sherpa-onnx"
    if name.startswith("whispercpp:"):
        return "whisper.cpp"
    m = info(name)
    return m.runtime if m is not None else "faster-whisper"


def is_whisper(name: str) -> bool:
    """Whether `voiceio learn` fine-tunes can stand in for this model: they
    are Whisper, so they only matter while dictation decodes with Whisper."""
    return runtime_of(name) == "faster-whisper"


def default_model(language: str = "en") -> str:
    """`[model] name` when the config doesn't set one.

    Parakeet once the installed sherpa-onnx has the NeMo TDT
    modified_beam_search fix (without it, hotwords lose speech — see
    CLAUDE.md "Model choice"), so the default flips by itself when upstream
    ships the fix. English only: other languages keep Whisper.
    """
    if language == "en":
        from voiceio.core.decoders.sherpa import features
        if features().length_reward:
            return PARAKEET
    return WHISPER_DEFAULT
