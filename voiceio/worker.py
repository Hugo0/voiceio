"""Whisper transcription worker that runs in a subprocess.

Communicates with the parent process via stdin (JSON requests) and
stdout (JSON responses). Loads the model once, then processes requests
in a loop for low-latency repeated transcriptions.

Usage: python3 -m voiceio.worker '{"model": "base", ...}'
"""
from __future__ import annotations

import base64
import faulthandler
import json
import math
import os
import signal
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


def auto_cpu_threads() -> int:
    """Half the logical cores (≈ physical cores), clamped to 4..8."""
    return max(4, min(8, (os.cpu_count() or 8) // 2))


def load_model(
    name: str, *, device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 0,
) -> WhisperModel:
    """Load a whisper model, preferring the local cache.

    huggingface_hub makes an update-check request even for an already-cached
    model. That connect has no timeout, so on a blackholed route (e.g. a broken
    IPv6 path to the CDN) it hangs indefinitely — wedging a caller whose model
    is sitting on disk. Try offline first and only reach for the network when
    the model genuinely isn't cached yet (first run, or a newly chosen model).

    Shared by every whisper load in the codebase: the worker, the teacher audit
    and the wizard. A cached model must never depend on the network.
    """
    kwargs = dict(
        device=device, compute_type=compute_type,
        cpu_threads=cpu_threads or auto_cpu_threads(),
    )
    try:
        return WhisperModel(name, local_files_only=True, **kwargs)
    except Exception:
        return WhisperModel(name, **kwargs)


# Whisper shares one 448-token window between prompt and output.
_MAX_LENGTH = 448
# Generous ceiling on output rate: fast English speech is ~4 words/s ≈ 5.5
# tokens/s. A decode stuck in a repetition loop runs to the window limit.
_TOKENS_PER_SEC = 7
_TOKEN_SLACK = 20


def max_new_tokens(model: WhisperModel, audio_secs: float,
                   hotwords: str | None, prompt: str | None) -> int | None:
    """Per-window output cap scaled to the audio, or None if it can't fit.

    faster-whisper raises if prompt + max_new_tokens exceeds the window, and
    truncates hotwords and prompt to 223 tokens each, so mirror that here.
    """
    half = _MAX_LENGTH // 2 - 1
    enc = model.hf_tokenizer.encode
    used = 8  # sot/language/task/timestamp special tokens + margin
    for text in (hotwords, prompt):
        if text:
            used += min(half, len(enc(" " + text.strip(), add_special_tokens=False).ids))
    cap = math.ceil(min(audio_secs, 30.0) * _TOKENS_PER_SEC) + _TOKEN_SLACK
    cap = min(cap, _MAX_LENGTH - used - 1)
    return cap if cap >= 32 else None


def main() -> None:
    # A hung decode is killed by the parent after a timeout; SIGUSR1 first
    # dumps every thread's stack to stderr (worker.log) so hangs are diagnosable.
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    args = json.loads(sys.argv[1])
    model = load_model(
        args["model"], device=args["device"], compute_type=args["compute_type"],
        cpu_threads=int(args.get("cpu_threads") or 0),
    )

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
        options = req.get("options") or {}
        hotwords = req.get("hotwords") or None
        kwargs = dict(
            language=args.get("language"),
            beam_size=int(options.get("beam_size", 1)),
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            # Greedy-at-T=0 only. The default 6-step temperature fallback
            # re-decodes the window whenever compression ratio / logprob trips
            # (common on jargon), multiplying latency, and the hot samples are
            # where hallucinations come from.
            temperature=0.0,
            max_new_tokens=max_new_tokens(model, len(audio) / 16000, hotwords, initial_prompt),
            # Short dictation utterances: conditioning on previous text lets one
            # bad window poison the rest (repetition loops); vad_filter drops the
            # pre-buffer noise and trailing silence that cause "Thank you."-style
            # hallucinations.
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        try:
            segs, _ = model.transcribe(audio, **kwargs)
            seg_list = list(segs)
        except ValueError:
            # Token estimate overshot the window; never fail a decode over it.
            kwargs["max_new_tokens"] = None
            segs, _ = model.transcribe(audio, **kwargs)
            seg_list = list(segs)
        text = " ".join(s.text.strip() for s in seg_list).strip()
        segments = [
            {
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 4),
                "no_speech_prob": round(s.no_speech_prob, 4),
                "compression_ratio": round(s.compression_ratio, 3),
            }
            for s in seg_list
        ]
        print(json.dumps({"text": text, "segments": segments}), flush=True)


if __name__ == "__main__":
    main()
