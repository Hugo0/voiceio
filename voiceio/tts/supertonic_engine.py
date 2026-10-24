"""Supertonic 3 TTS engine — neural, offline, 31 languages, via sherpa-onnx.

Faster than real time on a laptop CPU (RTF ~0.5 at 2 threads) in ~300 MB,
where Kokoro and Pocket TTS ran slower than real time on the same box. It
renders a whole sentence before playing, so the first audio comes after
~2 s: worth it for reading text aloud, not for snappy one-word replies, which
is why Piper stays first in the auto chain. Weights are OpenRAIL-M.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

MODEL = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{MODEL}.tar.bz2"
_SHA256 = "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427"
_SIZE_MB = 129
_STEMS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
_FILES = ("tts.json", "unicode_indexer.bin", "voice.bin")
_HINT = "pip install 'python-voiceio[sherpa]' (then: voiceio models sherpa)"


def _complete(path: Path) -> bool:
    return all((path / f).is_file() for f in _FILES) and all(
        any(path.glob(f"{stem}*.onnx")) for stem in _STEMS)


def _onnx(path: Path, stem: str) -> str:
    return str(sorted(path.glob(f"{stem}*.onnx"))[0])


class SupertonicEngine:
    name = "supertonic"

    def __init__(self, language: str = "", threads: int = 2):
        # "auto" can't name a language to read in; English is the safe reading.
        self._language = language if language not in ("", "auto") else "en"
        self._threads = threads
        self._tts = None  # lazy: the first synthesize() loads (and maybe fetches)

    def probe(self) -> ProbeResult:
        try:
            import sherpa_onnx
        except ImportError as e:
            return ProbeResult(ok=False, reason=f"sherpa-onnx missing: {e}", fix_hint=_HINT)
        if not hasattr(sherpa_onnx, "OfflineTtsSupertonicModelConfig"):
            return ProbeResult(ok=False, reason="this sherpa-onnx has no Supertonic support",
                               fix_hint=_HINT)
        return ProbeResult(ok=True)

    def _ensure(self) -> None:
        if self._tts is not None:
            return
        import sherpa_onnx as so

        from voiceio.core.decoders import fetch
        path = fetch.download_archive(
            MODEL, _URL, _SHA256, _SIZE_MB, _complete,
            what=", ".join((*_STEMS, *_FILES)))
        model = so.OfflineTtsModelConfig(num_threads=self._threads, provider="cpu")
        model.supertonic = so.OfflineTtsSupertonicModelConfig(
            duration_predictor=_onnx(path, "duration_predictor"),
            text_encoder=_onnx(path, "text_encoder"),
            vector_estimator=_onnx(path, "vector_estimator"),
            vocoder=_onnx(path, "vocoder"),
            tts_json=str(path / "tts.json"),
            unicode_indexer=str(path / "unicode_indexer.bin"),
            voice_style=str(path / "voice.bin"))
        log.info("TTS: loading Supertonic 3 (%s)…", self._language)
        self._tts = so.OfflineTts(so.OfflineTtsConfig(model=model, max_num_sentences=1))

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        import sherpa_onnx as so

        self._ensure()
        gen = so.GenerationConfig()
        gen.sid = int(voice) if voice.isdigit() else 0
        gen.speed = speed if speed > 0 else 1.0
        gen.extra = {"lang": self._language}
        out = self._tts.generate(text, gen)
        samples = np.asarray(out.samples, dtype=np.float32)
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16), out.sample_rate

    def shutdown(self) -> None:
        self._tts = None
