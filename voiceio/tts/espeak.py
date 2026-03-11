"""espeak-ng TTS engine — lightweight system fallback."""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave

import numpy as np

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)


class EspeakEngine:
    name = "espeak"

    def probe(self) -> ProbeResult:
        if not shutil.which("espeak-ng"):
            return ProbeResult(ok=False, reason="espeak-ng not installed",
                               fix_hint="install espeak-ng")
        return ProbeResult(ok=True)

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        """Synthesize text using espeak-ng --stdout."""
        wpm = int(175 * speed)
        cmd = ["espeak-ng", "--stdout", "-s", str(wpm)]
        if voice:
            cmd.extend(["-v", voice])
        else:
            cmd.extend(["-v", "en"])
        cmd.append(text)

        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {result.stderr.decode()[:200]}")

        with wave.open(io.BytesIO(result.stdout), "rb") as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        return audio, sample_rate

    def shutdown(self) -> None:
        pass
