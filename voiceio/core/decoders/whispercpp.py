"""whisper.cpp server decoder. Experimental.

whisper.cpp runs on GPUs that CTranslate2 can't use — notably AMD iGPUs via
its Vulkan build — so a laptop's Radeon can serve large-v3-turbo. Rather than
binding a native library, voiceio talks to `whisper-server` over HTTP:

    whisper-server -m ggml-large-v3-turbo-q5_0.bin --host 127.0.0.1 --port 8080

and `[model] name = "whispercpp:http://127.0.0.1:8080"`. Biasing is prompt
only (whisper.cpp has no hotwords).
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import uuid
import wave

import numpy as np

from voiceio.core.decoders.base import (
    Bias,
    BiasMode,
    Capabilities,
    DecodeError,
    DecodeResult,
)


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _multipart(fields: dict[str, str], wav: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                     .encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WhisperCppServer:
    capabilities = Capabilities(bias=BiasMode.PROMPT)

    def __init__(self, url: str, *, language: str = "en", timeout: float = 120.0):
        self._url = url.rstrip("/")
        self.name = f"whispercpp:{self._url}"
        self._language = language
        self._timeout = timeout

    def decode(self, audio: np.ndarray, *, final: bool = False,
               bias: Bias = Bias()) -> DecodeResult:
        prompt = "\n".join(p for p in (bias.hotwords, bias.prompt, bias.context) if p)
        fields = {"response_format": "json", "language": self._language,
                  "temperature": "0.0"}
        if prompt:
            fields["prompt"] = prompt
        body, ctype = _multipart(fields, _wav_bytes(audio))
        req = urllib.request.Request(f"{self._url}/inference", data=body,
                                     headers={"Content-Type": ctype})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                text = json.loads(resp.read()).get("text", "").strip()
        except Exception as e:  # noqa: BLE001
            raise DecodeError(f"whisper.cpp server: {e}") from e
        return DecodeResult(text, (), time.monotonic() - t0)

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._url}/", timeout=2):
                return True
        except Exception:  # noqa: BLE001
            return False

    def recover(self) -> None:
        if not self.healthy():
            raise DecodeError(f"whisper.cpp server at {self._url} is not reachable")

    def close(self) -> None:
        pass
