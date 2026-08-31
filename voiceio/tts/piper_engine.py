"""Piper TTS engine — high-quality offline synthesis."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "en_US-lessac-medium"

# piper 1.3 replaced the API this engine was written against: `piper.download`
# (ensure_voice_exists/get_voices) became `piper.download_voices`, and
# `synthesize_stream_raw(length_scale=...)` became `synthesize()` taking a
# `SynthesisConfig`. Probing for the *removed* module reported piper
# unavailable on every up-to-date install, so auto-selection skipped it
# silently and a piper-only host answered "no TTS engine available".
_UPGRADE_HINT = "pip install -U 'piper-tts>=1.3'"


def _models_dir() -> Path:
    """Where downloaded voices live. One dir, flat: `<voice>.onnx[.json]`."""
    return Path.home() / ".local" / "share" / "voiceio" / "tts-models"


class PiperEngine:
    name = "piper"

    def __init__(self, model: str = ""):
        self._model_name = model or _DEFAULT_MODEL
        self._voice = None  # lazy-loaded
        self._sample_rate = 0

    def probe(self) -> ProbeResult:
        """Check for the API we actually call, so "available" means it.

        The voice model itself is fetched on first use (see `_ensure_voice`),
        which is the contract callers already rely on — probing must stay cheap
        enough to run during backend selection.
        """
        try:
            from piper import PiperVoice, SynthesisConfig  # noqa: F401
            from piper.download_voices import download_voice  # noqa: F401
        except ImportError as e:
            return ProbeResult(
                ok=False,
                reason=f"piper-tts missing, or too old for the 1.3+ API: {e}",
                fix_hint=_UPGRADE_HINT,
            )
        return ProbeResult(ok=True)

    def _ensure_voice(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice
        from piper.download_voices import download_voice

        models_dir = _models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{self._model_name}.onnx"

        if not model_path.exists():
            log.info("TTS: downloading piper voice '%s'...", self._model_name)
            download_voice(self._model_name, models_dir)

        log.info("TTS: loading piper model '%s'...", self._model_name)
        # `download_dir` defaults to the *current working directory*, where
        # piper drops any extra runtime data it needs. A daemon's cwd is not
        # ours to write into.
        self._voice = PiperVoice.load(
            model_path,
            config_path=f"{model_path}.json",
            download_dir=models_dir,
        )
        self._sample_rate = self._voice.config.sample_rate
        log.info("TTS: piper model ready (sr=%d)", self._sample_rate)

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig

        self._ensure_voice()
        # length_scale stretches each phoneme, so it is the inverse of speed:
        # 1.0 is the voice's natural rate, 0.75 ≈ 1.33x faster.
        syn_config = SynthesisConfig(length_scale=1.0 / speed if speed > 0 else 1.0)

        chunks = [
            chunk.audio_int16_array
            for chunk in self._voice.synthesize(text, syn_config=syn_config)
        ]
        if not chunks:
            return np.array([], dtype=np.int16), self._sample_rate

        return np.concatenate(chunks), self._sample_rate

    def shutdown(self) -> None:
        self._voice = None
