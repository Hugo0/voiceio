"""sherpa-onnx transducer decoder (Parakeet, Nemotron streaming, …). Experimental.

sherpa-onnx's hotwords are an Aho-Corasick graph with no token cap — the one
way past Whisper's ~35-term ceiling. They only work with
`modified_beam_search`, which for NeMo TDT models lost speech on ~40% of real
clips when measured here (sherpa-onnx 1.13.4, issue #3267). So biasing is
declared only when the decoding method supports it, and `voiceio models bench`
must confirm MBS keeps the words greedy finds before anyone switches.

Model directory layout (as published by sherpa-onnx):
    encoder*.onnx  decoder*.onnx  joiner*.onnx  tokens.txt  [bpe.vocab]
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from voiceio.core.decoders.base import (
    Bias,
    BiasMode,
    Capabilities,
    DecodeError,
    DecodeResult,
)


def _find(model_dir: Path, stem: str) -> str:
    hits = sorted(model_dir.glob(f"{stem}*.onnx"), key=lambda p: ("int8" not in p.name, p.name))
    if not hits:
        raise DecodeError(f"no {stem}*.onnx in {model_dir}")
    return str(hits[0])


class SherpaTransducer:
    """Offline transducer decode via sherpa-onnx (one call per pass)."""

    def __init__(self, model_dir: str | Path, *, threads: int = 4,
                 method: str = "greedy_search", hotwords_score: float = 1.5,
                 model_type: str = "nemo_transducer"):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise DecodeError("sherpa-onnx not installed: pip install sherpa-onnx") from e
        self._sherpa = sherpa_onnx
        self._dir = Path(model_dir).expanduser()
        self.name = f"sherpa:{self._dir.name}:{method}"
        self._method = method
        self._score = hotwords_score
        self._threads = threads
        self._model_type = model_type
        self._lock = threading.Lock()
        self._recognizers: dict[str | None, object] = {}
        mbs = method == "modified_beam_search"
        self.capabilities = Capabilities(bias=BiasMode.UNBOUNDED if mbs else BiasMode.NONE)
        self._recognizer(None)  # load now: fail at startup, not mid-dictation

    def _recognizer(self, hotwords: str | None):
        """Hotwords are bound at construction, so cache one recognizer per
        hotword set (the vocabulary changes rarely)."""
        rec = self._recognizers.get(hotwords)
        if rec is not None:
            return rec
        kwargs = dict(
            encoder=_find(self._dir, "encoder"), decoder=_find(self._dir, "decoder"),
            joiner=_find(self._dir, "joiner"), tokens=str(self._dir / "tokens.txt"),
            num_threads=self._threads, decoding_method=self._method,
            model_type=self._model_type,
        )
        if hotwords and self._method == "modified_beam_search":
            f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
            f.write("\n".join(t.strip() for t in hotwords.split(",") if t.strip()) + "\n")
            f.close()
            kwargs.update(hotwords_file=f.name, hotwords_score=self._score)
            bpe = self._dir / "bpe.vocab"
            if bpe.exists():
                kwargs.update(modeling_unit="bpe", bpe_vocab=str(bpe))
        rec = self._sherpa.OfflineRecognizer.from_transducer(**kwargs)
        self._recognizers = {hotwords: rec}  # keep only the current set
        return rec

    def decode(self, audio: np.ndarray, *, final: bool = False,
               bias: Bias = Bias()) -> DecodeResult:
        t0 = time.monotonic()
        with self._lock:
            try:
                rec = self._recognizer(bias.hotwords if self.capabilities.bias else None)
                stream = rec.create_stream()
                stream.accept_waveform(16000, audio.astype(np.float32))
                rec.decode_stream(stream)
                text = stream.result.text.strip()
            except DecodeError:
                raise
            except Exception as e:  # noqa: BLE001
                raise DecodeError(f"sherpa-onnx decode failed: {e}") from e
        return DecodeResult(text, (), time.monotonic() - t0)

    def healthy(self) -> bool:
        return True

    def recover(self) -> None:
        pass

    def close(self) -> None:
        self._recognizers.clear()
