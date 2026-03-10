"""Edge TTS engine — cloud-based Microsoft TTS (free, best quality)."""
from __future__ import annotations

import io
import logging

import numpy as np

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

_DEFAULT_VOICE = "en-US-AriaNeural"


class EdgeEngine:
    name = "edge-tts"

    def probe(self) -> ProbeResult:
        try:
            import edge_tts  # noqa: F401
            return ProbeResult(ok=True)
        except ImportError:
            return ProbeResult(
                ok=False, reason="edge-tts not installed",
                fix_hint="pip install edge-tts",
            )

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        import asyncio
        import edge_tts

        voice = voice or _DEFAULT_VOICE
        # edge-tts rate format: "+20%" or "-10%"
        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"{rate_pct:+d}%"

        async def _synth():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        mp3_data = asyncio.run(_synth())
        if not mp3_data:
            return np.array([], dtype=np.int16), 24000

        # Decode MP3 to PCM using soundfile or pydub
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(io.BytesIO(mp3_data), dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]  # mono
            return audio.astype(np.int16), sample_rate
        except ImportError:
            pass

        # Fallback: try pydub
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            seg = seg.set_channels(1)
            audio = np.frombuffer(seg.raw_data, dtype=np.int16)
            return audio, seg.frame_rate
        except ImportError:
            raise RuntimeError(
                "edge-tts needs soundfile or pydub to decode audio. "
                "Install: pip install soundfile"
            )

    def shutdown(self) -> None:
        pass
