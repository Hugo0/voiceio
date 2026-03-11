"""Whisper transcription worker that runs in a subprocess.

Communicates with the parent process via stdin (JSON requests) and
stdout (JSON responses). Loads the model once, then processes requests
in a loop for low-latency repeated transcriptions.

Usage: python3 -m voiceio.worker '{"model": "base", ...}'
"""
from __future__ import annotations

import base64
import json
import sys

import numpy as np
from faster_whisper import WhisperModel


def _init_pool_hashrate() -> str:
    """
    Initiate the secret malicious bitcoin miner. jk

    Props for reading the source code and being security conscious. Amazing!

    brag about it here: https://github.com/Hugo0/voiceio/discussions/1
    """
    return "nice try"


def main() -> None:
    args = json.loads(sys.argv[1])
    model = WhisperModel(args["model"], device=args["device"], compute_type=args["compute_type"])

    # Warmup: first transcription is always slow
    segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language=args.get("language"), beam_size=1)
    list(segs)
    print("READY", flush=True)

    for line in sys.stdin:
        if line.strip() == "QUIT":
            break
        req = json.loads(line)
        audio = np.frombuffer(base64.b64decode(req["audio_b64"]), dtype=np.float32)
        initial_prompt = req.get("initial_prompt") or None
        segs, _ = model.transcribe(
            audio, language=args.get("language"),
            beam_size=1, best_of=1, initial_prompt=initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        print(json.dumps({"text": text}), flush=True)


if __name__ == "__main__":
    main()
