"""Headless HTTP/WebSocket server for voiceio.

The desktop app is one consumer of voiceio's transcription and speech stack;
this is another. It exposes the same machinery — the same model, the same
personal vocabulary, the same corrections and post-processing pipeline — over
HTTP, so a phone, a hub, or another service can use the owner's voice setup
without a display, an input method, or an audio device.

    python -m voiceio.server            # 127.0.0.1:8788

Install with the extra that carries the HTTP dependency:

    pip install "python-voiceio[server]"

Public API:
    ServerConfig      — configuration, `ServerConfig.from_env()` for $VOICE_*
    VoiceEngine       — the model + pipeline, no HTTP; usable on its own
    build_app(cfg)    — an aiohttp Application
    run(cfg)          — build and serve, blocking

Nothing here is imported by the desktop app, and importing it does not pull in
any desktop dependency — no audio device, no X11, no input method.
"""
from __future__ import annotations

from voiceio.server.config import ServerConfig

__all__ = ["ServerConfig", "VoiceEngine", "build_app", "run"]


def __getattr__(name: str):
    """Defer the aiohttp-dependent imports so that `voiceio.server.ServerConfig`
    (and `--help`) work even where the `server` extra is not installed, and the
    missing dependency is reported as itself rather than as an import error
    from three modules deep."""
    if name in ("build_app", "run"):
        from voiceio.server import app as _app

        return getattr(_app, name)
    if name == "VoiceEngine":
        from voiceio.server.engine import VoiceEngine

        return VoiceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
