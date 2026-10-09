"""Synthetic training examples for the user's hard words.

A personal recording set contains each rare term only a handful of times —
too few for a model to learn its spelling. Speaking the term in varied
sentences with TTS and adding those clips to the training split is an
established fix for rare-word recognition (e.g. arXiv 2508.17796,
2509.09196). Real audio must stay the majority, or the model drifts toward
synthetic prosody; `build_rows` caps the synthetic share.

Sentence frames are deliberately neutral — no domain or user assumptions.
Synthesis uses the local TTS chain and never a cloud engine by default.
"""
from __future__ import annotations

import json
import logging
import random
import wave
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from voiceio.learn.store import learn_dir

log = logging.getLogger(__name__)

_FRAMES = (
    "I was just looking at {t} again.",
    "Can you open {t} for me?",
    "Let's talk about {t} tomorrow.",
    "{t} is the one I meant.",
    "We should check what {t} says about that.",
    "I think {t} needs another look.",
    "Did you see the update to {t}?",
    "Put {t} at the top of the list.",
    "So the plan is to use {t} first.",
    "Send the notes on {t} to the team.",
    "Why did {t} stop working?",
    "Remind me to ask about {t}.",
)


def augment_dir() -> Path:
    return learn_dir() / "augment"


def _to_16k_mono(audio: np.ndarray, rate: int) -> np.ndarray:
    x = audio.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.5:  # int16 scale
        x /= 32768.0
    if rate != 16000:
        n = int(len(x) * 16000 / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def synthesize(terms: Iterable[str], tts, *, per_term: int = 8, voices: tuple[str, ...] = ("",),
               speeds: tuple[float, ...] = (0.9, 1.0, 1.15), seed: int = 0,
               progress: Callable[[str, int], None] | None = None) -> int:
    """Write synthetic clips + manifest rows for each term. Returns clip count."""
    rng = random.Random(seed)
    out = augment_dir()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    n = 0
    with open(manifest, "a", encoding="utf-8") as mf:
        for term in terms:
            frames = rng.sample(_FRAMES, k=min(per_term, len(_FRAMES)))
            for i, frame in enumerate(frames):
                text = frame.format(t=term)
                try:
                    audio, rate = tts.synthesize(text, rng.choice(voices), rng.choice(speeds))
                except Exception:  # noqa: BLE001
                    log.exception("TTS failed for %r", term)
                    continue
                pcm = (np.clip(_to_16k_mono(audio, rate), -1, 1) * 32767).astype(np.int16)
                safe = "".join(c if c.isalnum() else "_" for c in term)[:40]
                path = out / f"{safe}-{i:02d}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(pcm.tobytes())
                mf.write(json.dumps({"file_name": str(path), "start": 0.0,
                                     "end": round(len(pcm) / 16000, 2), "text": text,
                                     "term": term, "source": "synthetic"}) + "\n")
                n += 1
            if progress:
                progress(term, n)
    return n


def rows(max_share: float = 0.3, real_rows: int = 0, seed: int = 0) -> list[dict]:
    """Synthetic train rows, capped at `max_share` of the final mix."""
    manifest = augment_dir() / "manifest.jsonl"
    if not manifest.exists():
        return []
    items = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    items = [r for r in items if Path(r["file_name"]).exists()]
    cap = int(real_rows * max_share / (1 - max_share)) if real_rows else len(items)
    random.Random(seed).shuffle(items)
    return items[:cap]
