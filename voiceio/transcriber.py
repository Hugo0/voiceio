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

TRANSCRIBE_TIMEOUT = 30  # seconds (floor; scaled up for long audio)


class TranscriptionError(RuntimeError):
    """Decode failed (timeout / worker crash). Distinct from a legitimate
    empty result ("" = silence) so callers never treat lost audio as decoded
    silence — the streaming freeze/final paths must not advance state on it."""
# Realtime-factor headroom for the read timeout. Whisper decodes well faster
# than realtime, so 1.5x the audio duration is a generous ceiling that still
# never kills a long dictation mid-decode.
TIMEOUT_PER_SECOND = 1.5
MAX_RESTARTS = 3
# Ceiling on the worker's model-load handshake. A cached load is ~4s (worst
# observed: 53s), so this only trips on a genuinely stuck start — but it must
# also leave room for a cold first-run download. Unbounded, a start that hangs
# (blackholed network, see worker._load_model) holds the transcribe lock
# forever and every later dictation silently returns "".
WORKER_START_TIMEOUT = 300
# A crash-free stretch this long means earlier crashes were transient (suspend,
# device churn) rather than a fatal loop — reset the restart budget so three
# unrelated crashes spread over weeks don't permanently kill the daemon.
RESTART_RESET_SECS = 600


def transcribe_timeout(audio_duration: float) -> float:
    """Read timeout for one transcription, scaled to audio length.

    A fixed 30s timeout kills any dictation whose decode exceeds it (a >2.5min
    utterance) with total text loss. Scale with the audio so the worker always
    gets enough time, keeping the 30s floor for short clips.
    """
    return max(TRANSCRIBE_TIMEOUT, audio_duration * TIMEOUT_PER_SECOND)

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
        self._last_crash_time = 0.0  # monotonic; for restart-budget reset
        self._initial_prompt: str | None = None
        self._hotwords: str | None = None
        # Segment metadata (confidence etc.) from the most recent transcribe()
        # call. Read it immediately after the call that produced it.
        self.last_segments: list[dict] = []
        self._start_worker()

    def set_initial_prompt(self, prompt: str | None) -> None:
        """Set the initial_prompt (recent-transcript context conditioning)."""
        self._initial_prompt = prompt or None

    def set_hotwords(self, hotwords: str | None) -> None:
        """Set hotwords (vocabulary bias). Composes with initial_prompt:
        faster-whisper prepends hotwords before the prompt tokens."""
        self._hotwords = hotwords or None

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
        ready = self._read_with_timeout(WORKER_START_TIMEOUT)
        if ready is None or ready.strip() != "READY":
            self._kill_worker()
            detail = "timed out" if ready is None else f"said {ready.strip()!r}"
            raise TranscriptionError(
                f"worker failed to start ({detail} after {time.monotonic() - t0:.0f}s)"
            )
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
        # Reset the restart budget after a sustained crash-free period: three
        # transient crashes spread over weeks (suspend/device churn) must not
        # permanently disable the daemon. Only counts as sustained health if
        # we had actually crashed before (_last_crash_time set).
        now = time.monotonic()
        if (self._restarts > 0 and self._last_crash_time > 0
                and now - self._last_crash_time > RESTART_RESET_SECS):
            log.info(
                "Transcriber healthy for %.0fs since last crash, resetting restart budget",
                now - self._last_crash_time,
            )
            self._restarts = 0
        if self._restarts >= MAX_RESTARTS:
            raise RuntimeError(f"Transcriber worker crashed {MAX_RESTARTS} times, giving up")
        self._restarts += 1
        self._last_crash_time = now
        log.warning("Worker died, restarting (attempt %d/%d)", self._restarts, MAX_RESTARTS)
        self._start_worker()

    def transcribe(
        self, audio: np.ndarray, final: bool = False, context: str | None = None,
    ) -> str:
        """Transcribe audio. `final=True` marks the pass whose text the user
        keeps (streaming final / batch): it gets beam search; interim streaming
        passes stay greedy for latency.

        `context` is per-call text appended to the initial_prompt — used by
        incremental finalization to condition a tail decode on the already-
        frozen transcript so sentences stay coherent across the cut.
        """
        with self._lock:
            self._ensure_worker()

            duration = len(audio) / 16000
            t0 = time.monotonic()

            audio = normalize_audio(audio)
            audio_b64 = base64.b64encode(audio.tobytes()).decode("ascii")
            req = {"audio_b64": audio_b64, "options": {"beam_size": 5 if final else 1}}
            prompt = "\n".join(p for p in (self._initial_prompt, context) if p)
            if prompt:
                req["initial_prompt"] = prompt
            if self._hotwords:
                req["hotwords"] = self._hotwords
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                log.warning("Worker pipe broken, restarting")
                self._kill_worker()
                self._ensure_worker()
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()

            # Read with a timeout scaled to the audio length (never kill a
            # long dictation mid-decode).
            timeout = transcribe_timeout(duration)
            result_line = self._read_with_timeout(timeout)
            if result_line is None:
                self.last_segments = []
                log.warning("Transcription timed out after %.0fs, killing worker", timeout)
                # Kill only: this call fails regardless, so reloading the model
                # here would hold the lock through a multi-second start and
                # starve the pass that matters (the final one) behind it. The
                # next transcribe — or the health watchdog — restarts lazily.
                self._kill_worker()
                raise TranscriptionError(f"decode timed out after {timeout:.0f}s")

            try:
                result = json.loads(result_line)
            except (json.JSONDecodeError, TypeError):
                self.last_segments = []
                log.warning("Invalid response from worker: %s", repr(result_line)[:100])
                raise TranscriptionError("invalid worker response")
            text = result.get("text", "")
            self.last_segments = result.get("segments", [])

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
