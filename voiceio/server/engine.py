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
import os
import subprocess
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
        from voiceio.corrections import CorrectionDict
        from voiceio.transcriber import Transcriber
        from voiceio.vocabulary import VocabularyLoader

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

        self._vocab = VocabularyLoader(model_cfg)
        self._corrections = CorrectionDict()
        self._hotwords: str | None = None

        # Inherited by the whisper worker subprocess that Transcriber spawns.
        self.threads = self.cfg.threads
        if self.threads and self.threads != "0":
            os.environ["OMP_NUM_THREADS"] = self.threads

        t0 = time.monotonic()
        # Loads the model and runs a warmup decode before returning READY, so
        # the server is warm the moment it starts listening.
        self.transcriber = Transcriber(model_cfg)
        self.load_secs = time.monotonic() - t0
        self._refresh_hotwords()

        self._tts = None
        self._tts_probed = False

        log.info(
            "model=%s device=%s compute=%s threads=%s loaded in %.1fs "
            "(%d vocabulary terms, %d corrections)",
            model_cfg.name, self.device, model_cfg.compute_type, self.threads,
            self.load_secs,
            len(self._vocab.get_all()), len(self._corrections.list_all()),
        )

    # ── vocabulary biasing ───────────────────────────────────────────────
    def _refresh_hotwords(self) -> None:
        """Whisper's hotword channel is small; voiceio ranks the vocabulary by
        usage to fill it. Both reads are mtime-cached, so this is ~two stat
        calls once warm and picks up `voiceio correct` edits without a restart.
        """
        try:
            terms = self._vocab.get_selected(
                token_budget=self.cfg.hotwords_token_budget)
        except Exception:  # noqa: BLE001 — biasing is a bonus, never a blocker
            log.debug("hotword refresh failed", exc_info=True)
            return
        hotwords = ", ".join(terms)
        if hotwords != self._hotwords:
            self._hotwords = hotwords
            self.transcriber.set_hotwords(hotwords or None)

    def _record_vocab_usage(self, text: str) -> None:
        """Feed voiceio's usage stats so the hotword budget follows real use."""
        try:
            from voiceio import vocab_stats

            vocab_stats.update_from_text(text, self._vocab.get_all())
        except Exception:  # noqa: BLE001 — a counter must never break dictation
            log.debug("vocab usage update failed", exc_info=True)

    def postprocess(self, text: str, final: bool = True) -> str:
        """The shared pipeline the desktop app uses, minus the parts that only
        make sense at a cursor (voice editing commands) or that would send text
        to a cloud LLM (postcorrect/llm) — this server is offline by
        construction."""
        if not text:
            return text
        from voiceio.postprocess import apply_pipeline

        out, _abort = apply_pipeline(
            text,
            do_cleanup=self.vio_cfg.output.punctuation_cleanup,
            remove_disfluencies=self.vio_cfg.output.remove_disfluencies,
            number_conversion=self.vio_cfg.output.number_conversion,
            language=self.model_cfg.language,
            corrections=self._corrections,
            final=final,
        )
        return out

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

        self._refresh_hotwords()
        t1 = time.monotonic()
        raw = self.transcriber.transcribe(audio, final=True)
        whisper_ms = (time.monotonic() - t1) * 1000

        text = self.postprocess(raw, final=True)
        if text:
            self._record_vocab_usage(text)

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
        from voiceio.streaming import StreamingSession
        from voiceio.typers.collecting import CollectingTyper

        source = PushAudioSource(SAMPLE_RATE)
        typer = CollectingTyper()
        self._refresh_hotwords()
        session = StreamingSession(
            self.transcriber,
            typer,
            source,
            cleanup=self.vio_cfg.output.punctuation_cleanup,
            remove_disfluencies=self.vio_cfg.output.remove_disfluencies,
            number_conversion=self.vio_cfg.output.number_conversion,
            language=self.model_cfg.language,
            corrections=self._corrections,
            # No commands (no cursor), no postcorrect/llm (no cloud).
            commands=None,
            postcorrect=None,
            llm=None,
            on_interim=on_interim,
            freeze_secs=self.vio_cfg.output.streaming_freeze_secs,
        )
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
        return {
            "ok": True,
            "model": self.model_cfg.name,
            "device": self.device,
            "compute_type": self.model_cfg.compute_type,
            "language": self.model_cfg.language,
            "threads": self.threads,
            "worker_alive": self.transcriber.is_worker_alive(),
            "load_secs": round(self.load_secs, 2),
            "vocabulary_terms": len(self._vocab.get_all()),
            "corrections": len(self._corrections.list_all()),
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
        self.transcriber.shutdown()
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
