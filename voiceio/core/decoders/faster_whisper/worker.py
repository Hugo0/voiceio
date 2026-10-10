"""Whisper transcription worker that runs in a subprocess.

Communicates with the parent process via stdin (JSON requests) and
stdout (JSON responses). Loads the model once, then processes requests
in a loop for low-latency repeated transcriptions.

Usage: python3 -m voiceio.core.decoders.faster_whisper.worker '{"model": "base", ...}'
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


def auto_cpu_threads(lanes: int = 1) -> int:
    """Physical cores (≈ half the logical ones) shared across decode lanes,
    2..4 each. More doesn't pay: on a 7940HS whisper-small decodes 7.9s of
    audio in 4.6s at 4 threads and 4.7s at 8, and two lanes at 8 threads each
    collapse to 13.9s under load (6.0s at 4) — OpenMP oversubscription."""
    return max(2, min(4, (os.cpu_count() or 8) // 2 // max(1, lanes)))


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
# tokens/s, and proper nouns cost 5-7 tokens each, so allow ~2x that. A
# decode stuck in a repetition loop otherwise runs to the window limit.
_TOKENS_PER_SEC = 10
_TOKEN_SLACK = 32


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


# faster-whisper's own fallback triggers; a segment past them is a failed decode.
_COMPRESSION_RATIO_MAX = 2.4
_LOGPROB_MIN = -1.0
_NO_SPEECH_MIN = 0.6


def _failed(seg) -> bool:
    return seg.compression_ratio > _COMPRESSION_RATIO_MAX or (
        seg.avg_logprob < _LOGPROB_MIN and seg.no_speech_prob <= _NO_SPEECH_MIN)


def _transcribe(model: WhisperModel, audio: np.ndarray, kwargs: dict) -> list:
    try:
        segs, _ = model.transcribe(audio, **kwargs)
        return list(segs)
    except ValueError:
        # Token estimate overshot the window; never fail a decode over it.
        segs, _ = model.transcribe(audio, **{**kwargs, "max_new_tokens": None})
        return list(segs)


def decode(model: WhisperModel, audio: np.ndarray, kwargs: dict) -> list:
    """Transcribe; with hotwords, never let the fallback *sample* with them.

    Hotwords push Whisper into repetition loops on some windows (a 165s clip:
    one 224-token segment). The library's fallback then re-decodes by sampling
    at rising temperatures, which with hotwords dropped up to 40% of the speech,
    differently on every run. So hotwords get one greedy pass; if any segment
    fails the fallback's own checks, the audio is decoded again without them,
    where the usual fallback schedule is safe.
    """
    if not kwargs.get("hotwords"):
        return _transcribe(model, audio, kwargs)
    from voiceio.learn.metrics import count_repetitions
    segs = _transcribe(model, audio, {**kwargs, "temperature": 0.0})
    # A loop can also span several short segments that each look fine.
    if not (any(_failed(s) for s in segs)
            or count_repetitions(" ".join(s.text for s in segs))):
        return segs
    return _transcribe(model, audio, {**kwargs, "hotwords": None,
                                      "max_new_tokens": max_new_tokens(
                                          model, len(audio) / 16000, None,
                                          kwargs.get("initial_prompt"))})


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
            # Temperature: library default fallback schedule for now. T=0 alone
            # was measured to fall into repetition loops on ~1 in 5 long clips
            # (a 4-gram repeated 11-35x) that the fallback re-decode rescues.
            max_new_tokens=max_new_tokens(model, len(audio) / 16000, hotwords, initial_prompt),
            # Short dictation utterances: conditioning on previous text lets one
            # bad window poison the rest (repetition loops); vad_filter drops the
            # pre-buffer noise and trailing silence that cause "Thank you."-style
            # hallucinations.
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        seg_list = decode(model, audio, kwargs)
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
