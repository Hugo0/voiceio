"""The headless voiceio embedding: one loaded model, plus the personal layers.

This is the whole of the server's actual work — decoding audio, transcribing,
post-processing, synthesizing — with no HTTP in it, so it is testable and
reusable without standing up a socket. `voiceio.server.app` is a thin routing
layer over this class.

Everything here is blocking and serialized: one whisper worker, one utterance
at a time. Callers run it on an executor so an event loop stays free.
"""
from __future__ import annotations

import io
import logging
import subprocess
import threading
import time
import wave

import numpy as np

from voiceio.server.config import SAMPLE_RATE, ServerConfig

log = logging.getLogger("voiceio.server")

# ffmpeg is decode-only work; it should never outlast the audio by much.
FFMPEG_TIMEOUT = 60


class DecodeError(ValueError):
    """The supplied bytes are not audio we can decode."""


class AudioTooLong(ValueError):
    """Decoded fine, but longer than we are willing to transcribe."""


class TTSUnavailable(RuntimeError):
    """No text-to-speech engine could be selected."""


def resolve_device(configured: str) -> str:
    """Turn voiceio's "auto" into the device we actually got, for /health."""
    if configured and configured != "auto":
        return configured
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # noqa: BLE001 — probing must never block startup
        log.debug("CUDA probe failed", exc_info=True)
    return "cpu"


def decode_audio(data: bytes) -> np.ndarray:
    """Any container/codec ffmpeg knows → float32 mono 16 kHz, via pipes only.

    stdin→stdout, no temp files: browser MediaRecorder blobs (webm/opus) are
    streamable, and ffmpeg demuxes them from a pipe fine.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-vn", "-map", "0:a:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FFMPEG_TIMEOUT,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise DecodeError(detail[-1] if detail else "ffmpeg produced no audio")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """int16 mono samples → a complete RIFF/WAVE file in memory."""
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


class VoiceEngine:
    """One loaded model plus the owner's vocabulary, corrections and pipeline."""

    def __init__(self, cfg: ServerConfig | None = None) -> None:
        from voiceio import config as vio_config

        self.cfg = cfg or ServerConfig()
        self.started_at = time.time()

        # The owner's real config when present (model choice, language, and the
        # post-processing switches); stock defaults on a fresh server. The
        # server config overrides only what it was explicitly given.
        self.vio_cfg = vio_config.load()
        model_cfg = self.vio_cfg.model
        model_cfg.name = self.cfg.model or model_cfg.name
        model_cfg.compute_type = self.cfg.compute_type or model_cfg.compute_type
        model_cfg.language = self.cfg.language or model_cfg.language
        self.device = resolve_device(self.cfg.device or model_cfg.device)
        model_cfg.device = self.device
        self.model_cfg = model_cfg

        self.threads = self.cfg.threads
        if self.threads and self.threads != "0":
            model_cfg.cpu_threads = int(self.threads)

        from voiceio.core.engine import Engine

        t0 = time.monotonic()
        # Loads the model and runs a warmup decode before returning, so the
        # server is warm the moment it starts listening. The desktop's
        # vocabulary, corrections and prompt history apply here too; no voice
        # commands (no cursor), no cloud stages, no voice prefix (clients mark
        # text themselves).
        self.engine = Engine.from_config(
            self.vio_cfg, lanes=self.cfg.lanes, prefix=False)
        self.transcriber = self.engine.decoder
        self.load_secs = time.monotonic() - t0

        self._tts = None
        self._tts_probed = False

        if self.vio_cfg.data.retain_audio:
            import threading

            from voiceio import retention
            log.info("retaining dictations for `voiceio learn` ([data] retain_audio)")
            threading.Thread(target=retention.prune, args=(self.vio_cfg.data,),
                             daemon=True).start()

        log.info(
            "model=%s device=%s compute=%s threads=%s loaded in %.1fs "
            "(%d vocabulary terms, %d corrections)",
            model_cfg.name, self.device, model_cfg.compute_type, self.threads,
            self.load_secs,
            len(self.engine.vocabulary_terms), len(self.engine.corrections.list_all()),
        )

    def retain(self, audio, text: str, raw: str) -> None:
        """Keep this dictation for `voiceio learn` — only when the server's
        owner opted in with `[data] retain_audio = true` in its config.toml.
        Same caps and 0600 files as the desktop; tagged source="phone"."""
        if not (self.vio_cfg.data.retain_audio and text):
            return
        from voiceio import history, retention
        try:
            name = retention.save_audio(audio, time.time(), self.vio_cfg.data)
            history.append(text, raw=raw, duration=round(len(audio) / SAMPLE_RATE, 2),
                           extra={"audio": name, "source": "phone",
                                  "model": self.model_cfg.name},
                           cfg=self.vio_cfg.history)
        except Exception:  # noqa: BLE001 — retention must never fail a request
            log.exception("could not retain dictation")

    def record_usage(self, text: str) -> None:
        """Feed usage stats + prompt history so biasing follows real use."""
        self.engine.record_usage(text)

    def postprocess(self, text: str, final: bool = True) -> str:
        return self.engine.pipeline(text, final=final).text if text else text

    # ── transcription ────────────────────────────────────────────────────
    def transcribe(self, data: bytes) -> dict:
        """Encoded audio bytes → transcript. Runs on the executor thread."""
        t0 = time.monotonic()
        audio = decode_audio(data)
        decode_ms = (time.monotonic() - t0) * 1000

        secs = len(audio) / SAMPLE_RATE
        if secs > self.cfg.max_audio_secs:
            raise AudioTooLong(
                f"audio is {secs:.0f}s, limit is {self.cfg.max_audio_secs:.0f}s"
            )
        if secs < 0.1:
            return {"text": "", "empty": True, "audio_secs": round(secs, 2),
                    "decode_ms": round(decode_ms)}

        t1 = time.monotonic()
        result = self.engine.transcribe(audio)
        whisper_ms = (time.monotonic() - t1) * 1000
        text, raw = result.text, result.raw
        if text:
            self.record_usage(text)
            # Off the request path: the reply shouldn't wait on a WAV write.
            threading.Thread(target=self.retain, args=(audio, text, raw), daemon=True).start()

        return {
            "text": text,
            "raw": raw,
            "empty": not text,
            "audio_secs": round(secs, 2),
            "decode_ms": round(decode_ms),
            "whisper_ms": round(whisper_ms),
        }

    # ── streaming ────────────────────────────────────────────────────────
    def new_streaming_session(self, on_interim=None):
        """A `StreamingSession` wired to a push-fed audio source.

        Returns `(session, source, typer)`. The caller feeds PCM into `source`
        and ends with `session.stop(source.get_audio_so_far())`.
        """
        from voiceio.audio_source import PushAudioSource
        from voiceio.typers.collecting import CollectingTyper

        source = PushAudioSource(SAMPLE_RATE)
        typer = CollectingTyper()
        session = self.engine.session(source, typer, on_interim=on_interim)
        return session, source, typer

    # ── text to speech ───────────────────────────────────────────────────
    @property
    def tts(self):
        """The selected TTS engine, or None. Selected lazily and once: probing
        piper can hit the disk (and, the first time, the network for its voice
        model), which has no business happening during startup of a service
        whose main job is transcription."""
        if not self._tts_probed:
            self._tts_probed = True
            from voiceio import tts as tts_pkg
            from voiceio.config import TTSConfig

            tts_cfg = TTSConfig(
                engine=self.cfg.tts_engine or "auto",
                voice=self.cfg.tts_voice,
                speed=self.cfg.tts_speed,
                model=self.cfg.tts_model,
            )
            try:
                self._tts = tts_pkg.select(
                    tts_cfg, allow_network=self.cfg.tts_allow_network)
            except Exception:  # noqa: BLE001 — TTS is optional, never fatal
                log.warning("TTS selection failed", exc_info=True)
                self._tts = None
            if self._tts is not None:
                log.info("TTS engine: %s", self._tts.name)
        return self._tts

    def synthesize(self, text: str, voice: str = "", speed: float = 0.0) -> tuple[bytes, int, str]:
        """text → (WAV bytes, sample_rate, engine name)."""
        engine = self.tts
        if engine is None:
            raise TTSUnavailable(
                "no TTS engine available (install piper-tts or espeak-ng)")
        audio, rate = engine.synthesize(
            text,
            voice or self.cfg.tts_voice,
            speed or self.cfg.tts_speed,
        )
        return wav_bytes(audio, rate), rate, engine.name

    # ── introspection ────────────────────────────────────────────────────
    def health(self) -> dict:
        tts_name = None
        if self._tts_probed and self._tts is not None:
            tts_name = self._tts.name
        alive = self.transcriber.healthy()
        return {
            "ok": alive,
            "model": self.model_cfg.name,
            "device": self.device,
            "compute_type": self.model_cfg.compute_type,
            "language": self.model_cfg.language,
            "threads": self.threads,
            "worker_alive": alive,
            "load_secs": round(self.load_secs, 2),
            "vocabulary_terms": len(self.engine.vocabulary_terms),
            "corrections": len(self.engine.corrections.list_all()),
            "max_bytes": self.cfg.max_bytes,
            "max_audio_secs": self.cfg.max_audio_secs,
            # Added by the voiceio server module (the fields above are the
            # sidecar's historical contract and must not change shape).
            "uptime_secs": round(time.time() - self.started_at, 1),
            "sample_rate": SAMPLE_RATE,
            "tts_engine": tts_name,
            "tts_configured": self.cfg.tts_engine,
            "tts_allow_network": self.cfg.tts_allow_network,
            "voiceio_version": _version(),
        }

    def shutdown(self) -> None:
        self.engine.close()
        if self._tts is not None:
            try:
                self._tts.shutdown()
            except Exception:  # noqa: BLE001
                log.debug("TTS shutdown failed", exc_info=True)


def _version() -> str:
    try:
        import voiceio

        return voiceio.__version__
    except Exception:  # noqa: BLE001
        return "unknown"
