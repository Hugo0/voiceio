"""Teacher labeling: a stronger model transcribes the user's clips once.

There is no ground truth for someone's own dictation, and hand-labelling
hours of it won't happen. A bigger model, run offline with the *whole*
vocabulary as context and no latency budget, is far more accurate than the
live decoder — good enough to train on after the user reviews the spans that
matter (vocabulary terms, disagreements with the live output).

Differences from the live decode, all deliberate:
  * a larger model (default large-v3-turbo)
  * VAD off: the live VAD drops quiet word onsets, exactly where jargon dies
  * the full vocabulary rendered into the prompt, not just the ~35 hotwords
  * word timestamps, so labels can be cut into ≤30s training windows
"""
from __future__ import annotations

import logging
import os
import time
import wave
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from voiceio.learn.store import Clip, Segment, add_label, load_labels

log = logging.getLogger(__name__)

DEFAULT_TEACHER = "large-v3-turbo"


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class WhisperTeacher:
    def __init__(self, model: str = DEFAULT_TEACHER, *, device: str = "auto",
                 compute_type: str = "int8", vocabulary: Iterable[str] = (),
                 language: str = "en"):
        from voiceio.core.decoders.faster_whisper.worker import load_model
        self.name = f"teacher:{model}"
        # A background batch job (see cli.learn._yield_cpu): use every core idle.
        self._model = load_model(model, device=device, compute_type=compute_type,
                                 cpu_threads=max(4, (os.cpu_count() or 8) // 2))
        self._language = None if language in ("", "auto") else language   # None: detect
        terms = [t for t in vocabulary if t]
        # Prompt, not hotwords: hotwords are capped at 223 tokens; a prompt
        # of the most important terms still biases spelling. The tokenizer
        # truncates from the front, so put the bulk first, best terms last.
        self._prompt = ", ".join(reversed(terms)) or None

    def transcribe(self, audio: np.ndarray) -> tuple[Segment, ...]:
        segs, _ = self._model.transcribe(
            audio, language=self._language, beam_size=5, vad_filter=False,
            initial_prompt=self._prompt, condition_on_previous_text=False,
            # Library-default temperature fallback: slower on hard windows but
            # it rescues repetition loops, and a teacher has time to spare.
        )
        return tuple(Segment(round(s.start, 2), round(s.end, 2), s.text.strip())
                     for s in segs if s.text.strip())


def label_clips(clips: Iterable[Clip], teacher: WhisperTeacher, *,
                relabel: bool = False, limit: int | None = None,
                progress: Callable[[int, Clip, float], None] | None = None) -> int:
    """Teacher-label clips that have no label yet. Returns how many."""
    have = load_labels()
    done = 0
    for clip in clips:
        if limit is not None and done >= limit:
            break
        if not relabel and clip.audio in have:
            continue
        t0 = time.monotonic()
        try:
            segments = teacher.transcribe(load_wav(clip.path))
        except Exception:  # noqa: BLE001 — one bad file must not stop a batch
            log.exception("teacher failed on %s", clip.audio)
            continue
        add_label(clip.audio, segments, teacher.name)
        done += 1
        if progress:
            progress(done, clip, time.monotonic() - t0)
    return done
