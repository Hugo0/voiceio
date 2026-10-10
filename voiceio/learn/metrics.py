"""Scoring transcripts: word error rate, vocabulary recall, failure modes.

WER is computed from a real minimum-edit alignment (substitutions, deletions,
insertions). The difflib ratio used before overstates matches when words are
reordered and can't tell the error types apart.

Vocabulary recall is the number that matters for personalization: of the
user's terms spoken in the reference, how many did the hypothesis get right?
General WER barely moves when a model learns "Kalshi"; recall does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Bump when scoring changes: saved evals are only comparable within a version.
METRICS_VERSION = 2

_WORD = re.compile(r"[\w']+|%")
# Spellings that differ by style, not by what was heard. Without these a model
# fine-tuned on teacher labels "improves" by copying the teacher's habits
# ("sub-agents" vs "sub agents", "okay" vs "ok"). Hyphens split words, as in
# Whisper's own English normalizer.
_SPELLING = {"okay": "ok", "alright": "all right", "%": "percent"}

# Whisper's greatest hits over silence — it emits these with high confidence
# when there is nothing to transcribe. Matched whole-segment, so real speech
# containing "thank you" isn't flagged.
HALLUCINATION_MARKERS = frozenset({
    "thank you.", "thanks for watching!", "thank you for watching!",
    "you", "bye.", "thanks for watching.", "please subscribe.",
    "thank you very much.", "subtitles by the amara.org community",
    "。", "♪", "[music]", "[silence]",
})

# A phrase repeated this many times back-to-back is a decode loop, not speech.
_REPEAT_THRESHOLD = 3
_MIN_PERIOD, _MAX_PERIOD = 3, 20


def words(text: str) -> list[str]:
    """Normalized word tokens: lowercase, punctuation and hyphens stripped,
    style-only spelling variants unified."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        out.extend(_SPELLING.get(w, w).split())
    return out


@dataclass(frozen=True)
class Alignment:
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else float(self.insertions > 0)


def align(ref: str, hyp: str) -> Alignment:
    """Minimum-edit word alignment counts (Levenshtein over words)."""
    r, h = words(ref), words(hyp)
    # Each cell: (cost, subs, dels, ins); ties prefer substitutions.
    prev = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i in range(1, len(r) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                key=lambda c: c[0],
            )
            cur.append(best)
        prev = cur
    _, s, d, n = prev[-1]
    return Alignment(len(r), s, d, n)


def wer(ref: str, hyp: str) -> float:
    return align(ref, hyp).wer


def _count_term(text_words: list[str], term_words: list[str]) -> int:
    n, k = 0, len(term_words)
    for i in range(len(text_words) - k + 1):
        if text_words[i:i + k] == term_words:
            n += 1
    return n


def term_hits(ref: str, hyp: str, terms: list[str]) -> tuple[int, int]:
    """(occurrences of `terms` in ref, how many of those the hyp also has).

    Counted per term with multiplicity, capped by the reference count, so a
    hypothesis can't score by repeating a term the speaker said once.
    """
    rw, hw = words(ref), words(hyp)
    spoken = recalled = 0
    for term in terms:
        tw = words(term)
        if not tw:
            continue
        in_ref = _count_term(rw, tw)
        if in_ref:
            spoken += in_ref
            recalled += min(in_ref, _count_term(hw, tw))
    return spoken, recalled


def count_hallucinations(segments: list[str]) -> int:
    """Segments that are pure Whisper filler."""
    return sum(1 for s in segments if s.strip().lower() in HALLUCINATION_MARKERS)


def count_repetitions(text: str) -> int:
    """Phrases repeated back-to-back >= 3 times, at any period (3-20 words) —
    a real dictation looped on a 10-word phrase four times."""
    ws = re.findall(r"\w+", text.lower())
    if len(ws) < _MIN_PERIOD * _REPEAT_THRESHOLD:
        return 0
    n = i = 0
    while i < len(ws):
        matched = 0
        for period in range(_MAX_PERIOD, _MIN_PERIOD - 1, -1):
            if i + period * _REPEAT_THRESHOLD > len(ws):
                continue
            gram, reps, j = ws[i:i + period], 1, i + period
            while j + period <= len(ws) and ws[j:j + period] == gram:
                reps += 1
                j += period
            if reps >= _REPEAT_THRESHOLD:
                n += 1
                matched = j - i
                break
        i += matched or 1
    return n
