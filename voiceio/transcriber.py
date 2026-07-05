"""Subprocess-isolated faster-whisper transcriber with crash recovery."""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voiceio.config import ModelConfig

log = logging.getLogger(__name__)

TRANSCRIBE_TIMEOUT = 30  # seconds
MAX_RESTARTS = 3

# Loudness normalization: Whisper's log-mel frontend is not scale-invariant at
# extremes, so too-quiet audio transcribes poorly. Normalize RMS toward a
# speech-typical level, but never amplify noise unboundedly and never push the
# peak into clipping.
_TARGET_RMS = 10 ** (-20 / 20)  # -20 dBFS
_MAX_GAIN = 10 ** (30 / 20)     # +30 dB cap
_PEAK_LIMIT = 10 ** (-1 / 20)   # -1 dBFS


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """RMS-normalize float32 audio toward -20 dBFS, peak-limited to -1 dBFS."""
    if len(audio) == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-6:  # digital silence — nothing to normalize
        return audio
    gain = min(_TARGET_RMS / rms, _MAX_GAIN)
    peak = float(np.max(np.abs(audio)))
    if peak * gain > _PEAK_LIMIT:
        gain = _PEAK_LIMIT / peak
    if abs(gain - 1.0) < 0.05:
        return audio
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


class Transcriber:
    def __init__(self, cfg: ModelConfig):
        self._cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._restarts = 0
        self._initial_prompt: str | None = None
        self._start_worker()

    def set_initial_prompt(self, prompt: str | None) -> None:
        """Set a static initial_prompt for vocabulary conditioning."""
        self._initial_prompt = prompt or None

    def _start_worker(self) -> None:
        log.info(
            "Loading model '%s' (device=%s, compute_type=%s)...",
            self._cfg.name, self._cfg.device, self._cfg.compute_type,
        )
        language = self._cfg.language if self._cfg.language != "auto" else None

        args = json.dumps({
            "model": self._cfg.name,
            "device": self._cfg.device,
            "compute_type": self._cfg.compute_type,
            "language": language,
        })
        from voiceio.config import LOG_DIR
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._stderr_path = LOG_DIR / "worker.log"
        self._stderr_file = open(self._stderr_path, "w")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "voiceio.worker", args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
        )

        t0 = time.monotonic()
        ready = self._proc.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"Worker failed to start: {ready}")
        elapsed = time.monotonic() - t0
        log.info("Model ready (%.1fs)", elapsed)

    def is_worker_alive(self) -> bool:
        """Check if the worker subprocess is still running."""
        return self._proc is not None and self._proc.poll() is None

    def _ensure_worker(self) -> None:
        """Restart worker if it has died."""
        if self._proc is not None and self._proc.poll() is None:
            return
        if hasattr(self, "_stderr_path") and self._stderr_path.exists():
            try:
                stderr = self._stderr_path.read_text().strip()
                if stderr:
                    log.error("Worker stderr: %s", stderr[-500:])
            except OSError:
                pass
        if self._restarts >= MAX_RESTARTS:
            raise RuntimeError(f"Transcriber worker crashed {MAX_RESTARTS} times, giving up")
        self._restarts += 1
        log.warning("Worker died, restarting (attempt %d/%d)", self._restarts, MAX_RESTARTS)
        self._start_worker()

    def transcribe(self, audio: np.ndarray, final: bool = False) -> str:
        """Transcribe audio. `final=True` marks the pass whose text the user
        keeps (streaming final / batch): it gets beam search; interim streaming
        passes stay greedy for latency."""
        with self._lock:
            self._ensure_worker()

            duration = len(audio) / 16000
            t0 = time.monotonic()

            audio = normalize_audio(audio)
            audio_b64 = base64.b64encode(audio.tobytes()).decode("ascii")
            req = {"audio_b64": audio_b64, "options": {"beam_size": 5 if final else 1}}
            if self._initial_prompt:
                req["initial_prompt"] = self._initial_prompt
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                log.warning("Worker pipe broken, restarting")
                self._kill_worker()
                self._ensure_worker()
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()

            # Read with timeout
            result_line = self._read_with_timeout(TRANSCRIBE_TIMEOUT)
            if result_line is None:
                log.warning("Transcription timed out after %ds, restarting worker", TRANSCRIBE_TIMEOUT)
                self._kill_worker()
                self._ensure_worker()
                return ""

            try:
                result = json.loads(result_line)
            except (json.JSONDecodeError, TypeError):
                log.warning("Invalid response from worker: %s", repr(result_line)[:100])
                return ""
            text = result.get("text", "")

            elapsed = time.monotonic() - t0
            ratio = duration / elapsed if elapsed > 0 else 999
            log.info(
                "Transcribed %.1fs audio in %.1fs (%.1fx realtime): %s",
                duration, elapsed, ratio, text or "(silence)",
            )
            # Reset restart counter on success
            self._restarts = 0
            return text

    def _read_with_timeout(self, timeout: float) -> str | None:
        """Read a line from stdout with a timeout."""
        result = [None]

        def read():
            try:
                result[0] = self._proc.stdout.readline()
            except (OSError, ValueError):
                pass

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        return result[0]

    def _kill_worker(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                self._proc.kill()
            self._proc = None
        if hasattr(self, "_stderr_file") and self._stderr_file:
            self._stderr_file.close()
            self._stderr_file = None

    def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._proc.terminate()
            self._proc = None

    def __del__(self):
        self._kill_worker()
