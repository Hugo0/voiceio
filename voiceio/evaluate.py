"""Replay retained audio across decoder configs and score each one.

Every decoder knob in this project was set from a plausible story and one
example: `vad_filter=True` to stop "Thank you."-style hallucinations on silence,
`condition_on_previous_text=False` to stop repetition loops. Both stories are
reasonable. Neither was ever measured across the corpus, so nobody could say
what they cost — an ad-hoc probe on a single 422s clip suggested vad_filter was
costing real accuracy, but a single clip with hand-picked probe words is exactly
how you fool yourself.

There is no ground truth for a user's own dictation, and hand-labelling hours of
it is not going to happen. So this borrows the trick `audit.py` already relies
on: score against a stronger TEACHER model (distil-large-v3) instead of truth.
That is a proxy, not an oracle — the teacher is wrong sometimes too, and a
config could in principle beat the teacher and look worse for it. What it does
support is *ranking configs against each other* on the same audio, which is the
actual question.

Alongside teacher-WER it counts the two failure modes the knobs exist to
prevent, so a config that wins on WER while hallucinating is visibly not a win:

  * hallucinations — canonical Whisper filler emitted over silence/noise
  * repetitions    — the runaway n-gram loops that eat long dictations

Everything runs locally and offline; latency is irrelevant here.
"""
from __future__ import annotations

import logging
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voiceio import config
from voiceio.audit import _wav_duration, _wer

log = logging.getLogger(__name__)

# Whisper's greatest hits over silence — it emits these with high confidence
# when there is nothing to transcribe. Matched whole-string against a segment,
# not as substrings, so real speech containing "thank you" isn't flagged.
_HALLUCINATION_MARKERS = {
    "thank you.", "thanks for watching!", "thank you for watching!",
    "you", "bye.", "thanks for watching.", "please subscribe.",
    "thank you very much.", "subtitles by the amara.org community",
    "。", "♪", "[music]", "[silence]",
}

# A phrase repeated this many times back-to-back is a decode loop, not speech.
# Two is emphasis ("no, no"); three-plus is the decoder stuck in a cycle.
_REPEAT_THRESHOLD = 3
# Loops come at whatever length the stuck phrase happens to be — a real one was
# "because it should not work out in the complex world" (10 words) x4. Scanning
# a fixed n-gram width only catches loops whose period matches that width.
_MIN_PERIOD = 3
_MAX_PERIOD = 20


@dataclass
class DecodeConfig:
    """One point in the decoder config space."""

    name: str
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    hotwords: bool = True
    beam_size: int = 5

    def as_dict(self) -> dict:
        return {
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "hotwords": self.hotwords,
            "beam_size": self.beam_size,
        }


@dataclass
class ConfigScore:
    config: DecodeConfig
    clips: int = 0
    audio_secs: float = 0.0
    wer: float = 0.0
    hallucinations: int = 0
    repetitions: int = 0
    empties: int = 0
    crashes: int = 0
    decode_secs: float = 0.0
    per_clip: list[dict] = field(default_factory=list)


def count_hallucinations(segments: list[str]) -> int:
    """Segments that are pure Whisper filler."""
    n = 0
    for s in segments:
        t = s.strip().lower()
        if t and t in _HALLUCINATION_MARKERS:
            n += 1
    return n


def count_repetitions(text: str) -> int:
    """Count phrases repeated back-to-back at least _REPEAT_THRESHOLD times.

    Detects a loop of ANY period, not a fixed n-gram width: a real 422s
    dictation looped on "because it should not work out in the complex world"
    (10 words) four times, which a 4-gram scan slides straight past.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < _MIN_PERIOD * _REPEAT_THRESHOLD:
        return 0

    n = 0
    i = 0
    while i < len(words):
        matched = 0
        # Longest period first, so one 10-word loop is reported once rather
        # than as several shorter overlapping ones.
        for period in range(_MAX_PERIOD, _MIN_PERIOD - 1, -1):
            if i + period * _REPEAT_THRESHOLD > len(words):
                continue
            gram = words[i:i + period]
            reps = 1
            j = i + period
            while j + period <= len(words) and words[j:j + period] == gram:
                reps += 1
                j += period
            if reps >= _REPEAT_THRESHOLD:
                n += 1
                matched = j - i
                break
        i += matched if matched else 1
    return n


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# A clip at least this long can host a repetition loop and exercises the
# freeze/chunking path. Below it, `condition_on_previous_text`'s failure mode
# essentially cannot appear.
LONG_CLIP_SECS = 90.0

# Share of the audio budget reserved for long clips. Newest-first sampling is
# dominated by short notes (median utterance ~18s), which silently hides the
# risk that `cond=True` is only dangerous on long dictation — score it on short
# clips alone and it looks free.
_LONG_BUDGET_SHARE = 0.5


def sample_clips(max_audio_secs: float) -> list[Path]:
    """Retained WAVs up to an audio budget, stratified short vs long.

    Newest-first within each stratum, because the question is how the decoder
    behaves on how the user dictates *now*. But a pure newest-first sample is
    all short notes, and the two knobs under test fail in opposite regimes:
    `vad_filter` earns its keep on short/silent clips, while
    `condition_on_previous_text` only loops on long ones. A sample missing
    either regime will confidently recommend the wrong default.
    """
    rec = config.RECORDINGS_DIR
    if not rec.exists():
        return []

    wavs = sorted(rec.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    durations = {w: _wav_duration(w) for w in wavs}
    wavs = [w for w in wavs if durations[w] > 0]

    long_wavs = [w for w in wavs if durations[w] >= LONG_CLIP_SECS]
    short_wavs = [w for w in wavs if durations[w] < LONG_CLIP_SECS]

    def take(pool: list[Path], budget: float) -> tuple[list[Path], float]:
        out: list[Path] = []
        total = 0.0
        for w in pool:
            if total + durations[w] > budget and out:
                break
            out.append(w)
            total += durations[w]
        return out, total

    long_sel, long_used = take(long_wavs, max_audio_secs * _LONG_BUDGET_SHARE)
    # Anything the long stratum didn't spend goes to short clips rather than
    # being wasted — a user with no long recordings still gets a full sample.
    short_sel, _ = take(short_wavs, max_audio_secs - long_used)
    return short_sel + long_sel


def evaluate(
    cfg,
    configs: list[DecodeConfig],
    *,
    max_audio_secs: float = 900,
    teacher=None,
    on_progress=None,
) -> list[ConfigScore]:
    """Decode every clip under every config; score each against the teacher.

    `teacher(wav_path) -> str` is injectable so tests can skip the model.
    """
    from voiceio.audit import _default_teacher
    from voiceio.vocab_stats import VocabStats
    from voiceio.vocabulary import load_terms, select_terms
    from voiceio.worker import load_model

    clips = sample_clips(max_audio_secs)
    if not clips:
        log.warning("No retained audio to evaluate")
        return []

    if teacher is None:
        teacher = _default_teacher()

    # Teacher transcribes each clip once; every config is scored against it.
    refs: dict[Path, str] = {}
    for i, c in enumerate(clips, 1):
        if on_progress:
            on_progress("teacher", i, len(clips))
        try:
            refs[c] = teacher(c)
        except Exception as e:  # noqa: BLE001
            log.debug("teacher failed on %s: %s", c.name, e)

    clips = [c for c in clips if refs.get(c, "").strip()]
    if not clips:
        log.warning("Teacher produced no reference text")
        return []

    from voiceio.app import _HOTWORDS_TOKEN_BUDGET

    stats = VocabStats()
    stats.load()
    hot = ", ".join(select_terms(
        load_terms(cfg.model), token_budget=_HOTWORDS_TOKEN_BUDGET,
        model_name=cfg.model.name, stats=stats,
    ))

    model = load_model(cfg.model.name, compute_type=cfg.model.compute_type)
    scores: list[ConfigScore] = []

    for dc in configs:
        score = ConfigScore(config=dc)
        wers: list[float] = []
        for i, c in enumerate(clips, 1):
            if on_progress:
                on_progress(dc.name, i, len(clips))
            audio = _load_wav(c)
            t0 = time.monotonic()
            try:
                segs, _ = model.transcribe(
                    audio,
                    language=cfg.model.language,
                    beam_size=dc.beam_size,
                    condition_on_previous_text=dc.condition_on_previous_text,
                    vad_filter=dc.vad_filter,
                    vad_parameters=(
                        {"min_silence_duration_ms": 300} if dc.vad_filter else None
                    ),
                    hotwords=hot if (dc.hotwords and hot) else None,
                )
                seg_texts = [s.text for s in segs]
            except Exception as e:  # noqa: BLE001
                # cond_on_prev + hotwords can exhaust the 448-token budget and
                # raise "maximum decoding length must be > 0". A config that
                # cannot decode is a result, not an error.
                log.debug("%s crashed on %s: %s", dc.name, c.name, e)
                score.crashes += 1
                continue
            score.decode_secs += time.monotonic() - t0

            text = " ".join(seg_texts).strip()
            ref = refs[c]
            w = _wer(ref, text)
            wers.append(w)
            score.clips += 1
            score.audio_secs += _wav_duration(c)
            if not text:
                score.empties += 1
            halluc = count_hallucinations(seg_texts)
            reps = count_repetitions(text)
            score.hallucinations += halluc
            score.repetitions += reps
            score.per_clip.append({
                "clip": c.name, "wer": round(w, 4),
                "hallucinations": halluc, "repetitions": reps,
                "words": len(text.split()),
            })

        score.wer = round(sum(wers) / len(wers), 4) if wers else 1.0
        scores.append(score)

    return sorted(scores, key=lambda s: s.wer)


def default_matrix() -> list[DecodeConfig]:
    """The configs worth comparing against what ships today.

    `condition_on_previous_text=True` is only offered WITHOUT hotwords: the two
    share faster-whisper's `sot_prev` slot, each capped at 223 tokens with their
    sum unchecked against a 448 budget, so together they reliably raise
    "The maximum decoding length must be > 0". That combination is not a config,
    it's a crash.
    """
    return [
        DecodeConfig("shipped", vad_filter=True, condition_on_previous_text=False,
                     hotwords=True),
        DecodeConfig("no-vad", vad_filter=False, condition_on_previous_text=False,
                     hotwords=True),
        DecodeConfig("no-hotwords", vad_filter=True, condition_on_previous_text=False,
                     hotwords=False),
        DecodeConfig("cond-no-hotwords", vad_filter=True,
                     condition_on_previous_text=True, hotwords=False),
        DecodeConfig("no-vad-cond-no-hotwords", vad_filter=False,
                     condition_on_previous_text=True, hotwords=False),
    ]
