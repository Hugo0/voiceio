"""Piper TTS engine — high-quality offline synthesis."""
from __future__ import annotations

import logging

import numpy as np

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "en_US-lessac-medium"


class PiperEngine:
    name = "piper"

    def __init__(self, model: str = ""):
        self._model_name = model or _DEFAULT_MODEL
        self._voice = None  # lazy-loaded

    def probe(self) -> ProbeResult:
        try:
            import piper  # noqa: F401
            return ProbeResult(ok=True)
        except ImportError:
            return ProbeResult(
                ok=False, reason="piper-tts not installed",
                fix_hint="pip install piper-tts",
            )

    def _ensure_voice(self):
        if self._voice is not None:
            return
        from piper import PiperVoice
        from piper.download import ensure_voice_exists, get_voices

        from pathlib import Path

        data_dir = Path.home() / ".local" / "share" / "voiceio" / "tts-models"
        data_dir.mkdir(parents=True, exist_ok=True)

        model_name = self._model_name
        log.info("TTS: loading piper model '%s'...", model_name)

        voices_info = get_voices(data_dir, update_voices=False)
        ensure_voice_exists(model_name, [data_dir], data_dir, voices_info)

        # Find the .onnx file
        model_dir = data_dir / model_name
        if not model_dir.exists():
            # Some models use flat layout
            onnx_files = list(data_dir.glob(f"{model_name}*.onnx"))
            if onnx_files:
                model_path = onnx_files[0]
            else:
                raise FileNotFoundError(f"Model {model_name} not found after download")
        else:
            onnx_files = list(model_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file in {model_dir}")
            model_path = onnx_files[0]

        config_path = model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            # Try without double extension
            config_path = model_path.parent / (model_path.stem + ".json")

        self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self._sample_rate = self._voice.config.sample_rate
        log.info("TTS: piper model ready (sr=%d)", self._sample_rate)

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        self._ensure_voice()
        length_scale = 1.0 / speed if speed > 0 else 1.0

        audio_chunks = []
        for audio_bytes in self._voice.synthesize_stream_raw(
            text, length_scale=length_scale,
        ):
            chunk = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_chunks.append(chunk)

        if not audio_chunks:
            return np.array([], dtype=np.int16), self._sample_rate

        return np.concatenate(audio_chunks), self._sample_rate

    def shutdown(self) -> None:
        self._voice = None
