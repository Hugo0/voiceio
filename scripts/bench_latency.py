#!/usr/bin/env python3
"""Stop-to-text latency on your own recordings, with the real streaming path.

Replays retained clips through `StreamingSession` at real-time speed (as if
spoken), presses "stop" at the end, and measures how long the final text takes
— the number a user actually waits for. Compare configurations:

    uv run python scripts/bench_latency.py --lanes 1 --freeze 15
    uv run python scripts/bench_latency.py --lanes 2 --freeze 8

Clips come from voiceio's retained recordings (`[data] retain_audio`) or any
directory of 16 kHz mono WAVs. Prints timings only, never transcript text.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError(f"{path.name}: need 16 kHz mono")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main() -> None:
    from voiceio import config
    from voiceio.audio_source import PushAudioSource
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders import build_decoder
    from voiceio.core.pipeline import Pipeline
    from voiceio.streaming import StreamingSession
    from voiceio.typers.collecting import CollectingTyper
    from voiceio.vocabulary import VocabularyLoader

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=config.RECORDINGS_DIR)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--min-secs", type=float, default=10)
    ap.add_argument("--max-secs", type=float, default=120)
    ap.add_argument("--lanes", type=int, default=None, help="default: config")
    ap.add_argument("--freeze", type=float, default=None, help="default: config")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = config.load()
    freeze = cfg.output.streaming_freeze_secs if args.freeze is None else args.freeze
    clips = []
    for p in sorted(args.dir.glob("*.wav")):
        try:
            audio = load_wav(p)
        except (ValueError, wave.Error):
            continue
        if args.min_secs <= len(audio) / 16000 <= args.max_secs:
            clips.append(audio)
    random.Random(args.seed).shuffle(clips)
    clips = clips[: args.clips]
    if not clips:
        raise SystemExit(f"no usable clips in {args.dir}")

    decoder = build_decoder(cfg.model, lanes=args.lanes)
    bias = BiasProvider(VocabularyLoader(cfg.model)).bias_for(decoder.capabilities)
    print(f"decoder={decoder.name} freeze={freeze}s clips={len(clips)}")

    waits = []
    chunk = 1600  # 100 ms
    for audio in clips:
        source = PushAudioSource(16000)
        session = StreamingSession(decoder, CollectingTyper(), source,
                                   pipeline=Pipeline([]), freeze_secs=freeze, bias=bias)
        session.start()
        for i in range(0, len(audio), chunk):
            source.feed(audio[i:i + chunk])
            time.sleep(chunk / 16000 / args.speed)
        source.close()
        t0 = time.monotonic()
        session.stop(source.get_audio_so_far())
        wait = time.monotonic() - t0
        waits.append(wait)
        lat = session.final_latency
        print(f"  {len(audio) / 16000:6.1f}s audio → {wait:5.2f}s wait "
              f"(tail {lat.get('tail_secs', 0):.1f}s, frozen {lat.get('frozen_secs', 0):.1f}s)",
              flush=True)
    decoder.close()
    q = statistics.quantiles(waits, n=10) if len(waits) > 1 else waits
    print(f"stop→text: median {statistics.median(waits):.2f}s  p90 {q[-1]:.2f}s  "
          f"max {max(waits):.2f}s")


if __name__ == "__main__":
    main()
