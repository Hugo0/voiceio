"""Human review of teacher labels — only where it matters.

Reviewing hours of transcripts is not going to happen; reviewing a few
hundred short segments is. A segment is queued when it contains one of the
user's vocabulary terms (the words personalization exists for) or when the
teacher and the live decoder disagree about it (one of them is wrong).
Everything else keeps the teacher's text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voiceio.learn import metrics
from voiceio.learn.store import Clip, Label, Segment

# Below this the live decoder and teacher agree well enough to skip.
_DISAGREE_WER = 0.15


@dataclass(frozen=True)
class Item:
    clip: Clip
    label: Label
    index: int              # segment index within the label
    reason: str


def _contains(seg_words: list[str], live_words: list[str]) -> bool:
    """Whether the segment's words occur (nearly) contiguously in the live text."""
    n = len(seg_words)
    if not n:
        return True
    best = 0.0
    for i in range(0, max(1, len(live_words) - n + 1)):
        window = live_words[i:i + n]
        match = sum(a == b for a, b in zip(seg_words, window)) / n
        best = max(best, match)
        if best >= 1 - _DISAGREE_WER:
            return True
    return False


def queue(clips: list[Clip], labels: dict[str, Label], terms: list[str]) -> list[Item]:
    """Segments worth a human look, vocabulary ones first."""
    term_words = [metrics.words(t) for t in terms if t]
    vocab, disagree = [], []
    for clip in clips:
        lab = labels.get(clip.audio)
        if lab is None or lab.reviewed:
            continue
        live = metrics.words(clip.raw)
        for i, seg in enumerate(lab.segments):
            sw = metrics.words(seg.text)
            hit = next((" ".join(tw) for tw in term_words
                        if any(sw[j:j + len(tw)] == tw for j in range(len(sw)))), None)
            if hit:
                vocab.append(Item(clip, lab, i, f"vocabulary: {hit}"))
            elif live and not _contains(sw, live):  # imports have no live decode
                disagree.append(Item(clip, lab, i, "teacher and live decode disagree"))
    return vocab + disagree


def queue_all(clips: list[Clip], labels: dict[str, Label]) -> list[Item]:
    """Every segment of every unreviewed clip — for building a gold test set,
    where a reference is only trustworthy if all of it was checked."""
    return [Item(clip, labels[clip.audio], i, "test set")
            for clip in clips
            if clip.audio in labels and not labels[clip.audio].reviewed
            for i in range(len(labels[clip.audio].segments))]


def run(items: list[Item], *, ask: Callable[[Item], str | None],
        save: Callable[[Clip, list[Segment]], None]) -> int:
    """Drive a review session. `ask` returns the corrected text for a segment
    (unchanged text to accept, None to stop). Each clip is saved as a human
    label once all of its queued segments are answered. Returns clips saved."""
    by_clip: dict[str, list[Item]] = {}
    for it in items:
        by_clip.setdefault(it.clip.audio, []).append(it)
    saved = 0
    for its in by_clip.values():
        segs = list(its[0].label.segments)
        for it in its:
            answer = ask(it)
            if answer is None:
                return saved
            if answer.strip():
                segs[it.index] = Segment(segs[it.index].start, segs[it.index].end, answer.strip())
        save(its[0].clip, segs)
        saved += 1
    return saved
