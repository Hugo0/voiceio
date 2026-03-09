#!/usr/bin/env python3
"""Record a test clip and pre-compute Whisper outputs at growing windows.

Usage:
    python tests/record_fixture.py <name> [--duration 10]

Records audio, then runs Whisper on growing windows (1s, 2s, 3s, ..., full)
to capture the exact sequence of transcriptions streaming would produce.
Saves to tests/fixtures/<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def record(duration: float, sr: int = 16000) -> np.ndarray:
    print(f"Recording {duration}s... (speak now)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Done recording.")
    return audio.flatten()


def compute_windows(audio: np.ndarray, sr: int = 16000) -> list[dict]:
    """Run Whisper on growing windows, mimicking streaming behavior."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="auto", compute_type="int8")
    # Warmup
    segs, _ = model.transcribe(np.zeros(sr, dtype=np.float32), language="en", beam_size=1)
    list(segs)

    results = []
    step = sr  # 1-second steps
    for end in range(step, len(audio) + step, step):
        end = min(end, len(audio))
        chunk = audio[:end]
        t0 = time.monotonic()
        segs, _ = model.transcribe(chunk, language="en", beam_size=1, best_of=1)
        text = " ".join(s.text.strip() for s in segs).strip()
        elapsed = time.monotonic() - t0
        results.append({
            "audio_seconds": round(end / sr, 1),
            "text": text,
            "transcribe_seconds": round(elapsed, 2),
        })
        print(f"  {end/sr:.1f}s → {text!r} ({elapsed:.2f}s)")
        if end >= len(audio):
            break

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Fixture name (e.g. 'hello_world')")
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()

    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    audio = record(args.duration)
    print(f"\nComputing Whisper outputs at 1s intervals...")
    windows = compute_windows(audio)

    fixture = {
        "name": args.name,
        "duration": args.duration,
        "sample_rate": 16000,
        "windows": windows,
    }

    out_path = fixtures_dir / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
