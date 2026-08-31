"""Server configuration: a small object, populated from the environment.

Every knob has a working default, so `ServerConfig()` alone runs. `from_env()`
layers the environment on top, and layers *voiceio's own config* underneath —
so a box where the owner has set a model or language in `config.toml` inherits
it, and the environment overrides it. That ordering is what lets one server
image serve both a personal machine and a bare VPS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["ServerConfig", "DEFAULT_PORT", "SAMPLE_RATE"]

SAMPLE_RATE = 16000
DEFAULT_PORT = 8788

# ctranslate2 reads OMP_NUM_THREADS. Left unset it takes every core, which on a
# box that also runs production lets one dictation stall real work for seconds.
# 4 is where the scaling flattens for the small model anyway. "0" = library
# default (all cores).
DEFAULT_THREADS = "4"


def _env(prefix: str, name: str, default: str) -> str:
    value = os.environ.get(f"{prefix}{name}", "").strip()
    return value or default


@dataclass
class ServerConfig:
    """Everything the server needs to know, in one place."""

    # ── network ──────────────────────────────────────────────────────────
    # Loopback by default and on purpose: this service has no authentication,
    # because whatever fronts it (a hub, a reverse proxy) owns that. Binding
    # 0.0.0.0 would put an unauthenticated transcriber on the network.
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    # ── model ────────────────────────────────────────────────────────────
    model: str = ""          # "" = inherit voiceio's configured model
    language: str = ""       # "" = inherit
    compute_type: str = ""   # "" = inherit
    device: str = "auto"
    threads: str = DEFAULT_THREADS

    # ── limits ───────────────────────────────────────────────────────────
    # Request body cap. 25MB is ~2h of 24kbps opus, so the real limiter is
    # max_audio_secs; this exists so a bad client cannot stream us to death.
    max_bytes: int = 25 * 1024 * 1024
    max_audio_secs: float = 120.0

    # ── tts ──────────────────────────────────────────────────────────────
    tts_engine: str = "auto"
    tts_voice: str = ""
    tts_speed: float = 1.0
    tts_model: str = ""
    # Cloud TTS is opt-in. Auto-selection prefers piper, but edge-tts probes OK
    # anywhere there is a network — so an unattended server could start posting
    # its user's text to Microsoft without anyone choosing that.
    tts_allow_network: bool = False

    # ── misc ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    # Post-processing the desktop applies at a cursor (voice editing commands)
    # or via a cloud LLM (postcorrect) is off here by construction.
    hotwords_token_budget: int = 120

    _source_env: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls, prefix: str = "VOICE_") -> ServerConfig:
        """Build from `$VOICE_*` (or another prefix)."""
        e = _env
        cfg = cls(
            host=e(prefix, "HOST", "127.0.0.1"),
            port=int(e(prefix, "PORT", str(DEFAULT_PORT))),
            model=e(prefix, "MODEL", ""),
            language=e(prefix, "LANGUAGE", ""),
            compute_type=e(prefix, "COMPUTE", ""),
            device=e(prefix, "DEVICE", "auto"),
            threads=e(prefix, "THREADS", DEFAULT_THREADS),
            max_bytes=int(e(prefix, "MAX_BYTES", str(25 * 1024 * 1024))),
            max_audio_secs=float(e(prefix, "MAX_SECS", "120")),
            tts_engine=e(prefix, "TTS_ENGINE", "auto"),
            tts_voice=e(prefix, "TTS_VOICE", ""),
            tts_speed=float(e(prefix, "TTS_SPEED", "1.0")),
            tts_model=e(prefix, "TTS_MODEL", ""),
            tts_allow_network=e(prefix, "TTS_ALLOW_NETWORK", "0")
            not in ("0", "false", "no", ""),
            log_level=e(prefix, "LOG_LEVEL", "INFO"),
        )
        return cfg
