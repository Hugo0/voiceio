"""`python -m voiceio.server` — run the headless voice server."""
from __future__ import annotations

import shutil
import sys

from voiceio.server.config import ServerConfig


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = ServerConfig.from_env()

    if "-h" in argv or "--help" in argv:
        print(__doc__)
        print(
            "\nConfiguration is environment-only ($VOICE_*):\n"
            "  VOICE_HOST, VOICE_PORT           bind address "
            f"(default {cfg.host}:{cfg.port})\n"
            "  VOICE_MODEL, VOICE_LANGUAGE      whisper model / language\n"
            "  VOICE_COMPUTE, VOICE_DEVICE      compute type / device\n"
            "  VOICE_THREADS                    CPU threads (default 4, 0=all)\n"
            "  VOICE_MAX_BYTES, VOICE_MAX_SECS  request limits\n"
            "  VOICE_TTS_ENGINE, VOICE_TTS_VOICE, VOICE_TTS_SPEED\n"
            "  VOICE_TTS_ALLOW_NETWORK          1 to permit cloud TTS (off)\n"
            "  VOICE_LOG_LEVEL                  default INFO\n"
        )
        return 0

    # ffmpeg decodes every uploaded container; without it /transcribe can only
    # fail, and it should fail now with an actionable message rather than on
    # the first request.
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH — install it (apt install ffmpeg)",
              file=sys.stderr)
        return 1
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print('aiohttp not installed — pip install "python-voiceio[server]"',
              file=sys.stderr)
        return 1

    from voiceio.server.app import run

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
