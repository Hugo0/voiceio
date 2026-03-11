"""Voice Activity Detection backends: Silero VAD (ONNX) with RMS fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from voiceio.config import AudioConfig

log = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"
_WINDOW_SIZE = 512  # Silero expects 512 samples at 16kHz (~32ms)
_SAMPLE_RATE = 16000


@runtime_checkable
class VadBackend(Protocol):
    def reset(self) -> None: ...
    def is_speech(self, chunk: np.ndarray) -> bool: ...


class SileroVad:
    """Silero VAD using ONNX runtime. Stateful (recurrent hidden states)."""

    def __init__(self, threshold: float = 0.5, model_path: Path | None = None):
        import onnxruntime  # noqa: F811

        path = str(model_path or _MODEL_PATH)
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(path, sess_options=opts)
        self._threshold = threshold

        # Detect model version from input names
        input_names = {inp.name for inp in self._session.get_inputs()}
        self._use_state = "state" in input_names  # v5+ uses single state tensor

        # Hidden state for the recurrent model
        if self._use_state:
            state_meta = [inp for inp in self._session.get_inputs() if inp.name == "state"][0]
            state_dim = state_meta.shape[2]  # 128 for v5
            self._state = np.zeros((2, 1, state_dim), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        # Internal accumulator for sub-window chunks
        self._buf = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        if self._use_state:
            self._state = np.zeros_like(self._state)
        else:
            self._h = np.zeros_like(self._h)
            self._c = np.zeros_like(self._c)
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> bool:
        flat = chunk.ravel()
        self._buf = np.concatenate([self._buf, flat])

        speech = False
        while len(self._buf) >= _WINDOW_SIZE:
            window = self._buf[:_WINDOW_SIZE]
            self._buf = self._buf[_WINDOW_SIZE:]
            prob = self._infer(window)
            if prob >= self._threshold:
                speech = True

        return speech

    def _infer(self, window: np.ndarray) -> float:
        input_data = window.reshape(1, -1)
        sr = np.array(_SAMPLE_RATE, dtype=np.int64)
        if self._use_state:
            ort_inputs = {"input": input_data, "state": self._state, "sr": sr}
            out, self._state = self._session.run(None, ort_inputs)
        else:
            ort_inputs = {"input": input_data, "h": self._h, "c": self._c, "sr": sr}
            out, self._h, self._c = self._session.run(None, ort_inputs)
        return float(out.squeeze())

    def warmup(self) -> None:
        """Run a dummy inference to avoid cold-start latency in the audio callback."""
        self.is_speech(np.zeros(_WINDOW_SIZE, dtype=np.float32))
        self.reset()


class RmsVad:
    """Simple RMS-based silence detection (the original voiceio approach)."""

    def __init__(self, threshold: float = 0.01):
        self._threshold = threshold

    def reset(self) -> None:
        pass  # stateless

    def is_speech(self, chunk: np.ndarray) -> bool:
        flat = chunk.ravel()
        rms = float(np.sqrt(np.dot(flat, flat) / max(len(flat), 1)))
        return rms >= self._threshold


def load_vad(cfg: AudioConfig) -> VadBackend:
    """Create the best available VAD backend, falling back to RMS."""
    if cfg.vad_backend == "rms":
        log.info("Using RMS VAD (configured)")
        return RmsVad(threshold=cfg.silence_threshold)

    try:
        vad = SileroVad(threshold=cfg.vad_threshold)
        vad.warmup()
        log.info("Using Silero VAD (threshold=%.2f)", cfg.vad_threshold)
        return vad
    except Exception:
        log.warning("Silero VAD unavailable, falling back to RMS", exc_info=True)
        return RmsVad(threshold=cfg.silence_threshold)
