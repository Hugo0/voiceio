"""`voiceio vocab suggest`: words in your history that may be misrecognitions.

Offline frequency analysis against the language's word list: a word Whisper
keeps producing that no dictionary knows is usually a name it can't spell
(add it to your vocabulary) or a mishearing (add a correction).
"""
from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass, field

from wordfreq import top_n_list

from voiceio.wordfreq import extract_words, is_known, levenshtein


@dataclass
class SuspiciousWord:
    """A word that might be a Whisper transcription error."""
    word: str
    count: int                       # how often it appears in history
    contexts: list[str] = field(default_factory=list)  # example sentences
    similar_common: list[str] = field(default_factory=list)  # nearby common words
    reason: str = ""                 # why it's suspicious


def find_suspicious_words(
    entries: list[dict],
    language: str = "en",
    *,
    existing_corrections: set[str] | None = None,
    vocabulary: set[str] | None = None,
    dismissed: set[str] | None = None,
    min_word_length: int = 4,
) -> list[SuspiciousWord]:
    """Scan history entries for words that are likely Whisper mistakes.

    History v2 entries may carry a ``raw`` field (the pre-correction Whisper
    text). We prefer it when present, since post-corrected ``text`` hides the
    very misrecognitions we're hunting for. Entries with only ``text`` (v1)
    still work.
    """
    skip = existing_corrections or set()
    vocab = vocabulary or set()
    dismiss = dismissed or set()
    skip_lower = {w.lower() for w in skip}
    vocab_lower = {w.lower() for w in vocab}
    dismiss_lower = {w.lower() for w in dismiss}

    word_counts: Counter[str] = Counter()
    word_contexts: dict[str, list[str]] = {}

    for entry in entries:
        text = (entry.get("raw") or entry.get("text") or "").strip()
        if not text:
            continue
        words = extract_words(text)
        for w in words:
            wl = w.lower()
            word_counts[wl] += 1
            if wl not in word_contexts:
                word_contexts[wl] = []
            if len(word_contexts[wl]) < 3 and text not in word_contexts[wl]:
                word_contexts[wl].append(text)

    near = _common_words(language)

    candidates: list[SuspiciousWord] = []

    for word, count in word_counts.items():
        if len(word) < min_word_length:
            continue
        if word in skip_lower or word in vocab_lower or word in dismiss_lower:
            continue
        if is_known(word, language):
            continue
        similar = near.similar(word)
        reason = "not in common word list"
        if similar:
            reason += f", similar to: {', '.join(similar[:3])}"

        candidates.append(SuspiciousWord(
            word=word,
            count=count,
            contexts=word_contexts.get(word, []),
            similar_common=similar,
            reason=reason,
        ))

    candidates.sort(key=lambda s: (-len(s.similar_common), -s.count))
    return candidates


def _deletions(word: str, depth: int) -> set[str]:
    out, frontier = {word}, {word}
    for _ in range(depth):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= frontier
    return out


class NearWords:
    """Common words within a small edit distance of a given word.

    A deletion index (the SymSpell idea): two words within distance d share a
    string reachable by at most d deletions from each. So candidates come
    from dictionary lookups instead of comparing against every common word —
    `vocab suggest` went from ~80s to about a second on 6,000 dictations.
    """

    def __init__(self, ranked: dict[str, int], max_distance: int = 2):
        self._rank = ranked
        self._max = max_distance
        self._index: dict[str, set[str]] = {}
        for w in ranked:
            for d in _deletions(w, max_distance):
                self._index.setdefault(d, set()).add(w)

    def similar(self, word: str, n: int = 5) -> list[str]:
        found = set()
        for d in _deletions(word, self._max):
            found |= self._index.get(d, set())
        near = [w for w in found
                if levenshtein(word, w, self._max) <= self._max]
        return sorted(near, key=lambda w: self._rank.get(w, 99999))[:n]


@functools.lru_cache(maxsize=4)
def _common_words(language: str) -> NearWords:
    return NearWords({w: i for i, w in enumerate(top_n_list(language, 20000))})


def _find_similar_common(
    word: str, common_words: dict[str, int], max_distance: int = 2,
) -> list[str]:
    """Common words within `max_distance` edits of `word`, most common first."""
    return NearWords(common_words, max_distance).similar(word)
