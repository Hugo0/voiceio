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


def live_span(segment_text: str, live: str) -> str | None:
    """The stretch of the live decode that corresponds to a teacher segment,
    or None when it matches word for word (or can't be located). Seeing both
    versions side by side is the quickest way to judge which one is right."""
    import difflib
    live_raw = live.split()
    seg_n, live_n = metrics.words(segment_text), [metrics.words(w) for w in live_raw]
    flat = [w[0] if w else "" for w in live_n]
    if not seg_n or not flat:
        return None
    m = difflib.SequenceMatcher(a=flat, b=seg_n, autojunk=False)
    blocks = [b for b in m.get_matching_blocks() if b.size]
    if not blocks or sum(b.size for b in blocks) < len(seg_n) / 3:
        return None
    start = max(0, blocks[0].a - blocks[0].b)
    end = min(len(flat), blocks[-1].a + blocks[-1].size + (len(seg_n) - blocks[-1].b - blocks[-1].size))
    span = flat[start:end]
    if span == seg_n:
        return None
    return " ".join(live_raw[start:end])


def queue_all(clips: list[Clip], labels: dict[str, Label]) -> list[Item]:
    """Segments to check for a gold test set, cheapest clips first.

    A reference is only trustworthy if all of it was checked — but where the
    teacher and the live decode agree word for word, two independent models
    already did; those segments are kept as they are, and only the rest are
    asked. Clips needing the fewest answers come first, so each minute of
    review completes as many gold clips as possible. A clip with no live
    decode (an import) has every segment asked; one where everything agrees
    asks nothing and stays a teacher label.
    """
    per_clip: list[list[Item]] = []
    for clip in clips:
        lab = labels.get(clip.audio)
        if lab is None or lab.reviewed:
            continue
        live = metrics.words(clip.raw)
        items = [Item(clip, lab, i, "test set: teacher and live disagree" if live else "test set")
                 for i, seg in enumerate(lab.segments)
                 if not live or not _contains(metrics.words(seg.text), live)]
        if items:
            per_clip.append(items)
    per_clip.sort(key=len)
    return [it for items in per_clip for it in items]


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
