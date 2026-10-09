"""Finding likely misrecognitions in history, and vetting correction rules.

Offline: frequency analysis against the language's word list, plus a safety
gate so a correction rule can't rewrite real words. Powers
`voiceio vocab suggest` and `voiceio corrections add`.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from wordfreq import top_n_list

from voiceio.wordfreq import extract_words, is_common, is_known

log = logging.getLogger(__name__)


# ── Safety gate for persisting mined correction rules ────────────────────

# A word is "real" in a protected language when it's reasonably common there.
# Higher than the English non-word bar (2.0) because sharing a rare cognate
# shouldn't block fixing an obvious English misrecognition.
_PROTECT_ZIPF = 3.0


def gate_correction(
    wrong: str, right: str,
    *,
    vocabulary: set[str] | None = None,
    language: str = "en",
    protect_languages: tuple[str, ...] | list[str] = (),
) -> str | None:
    """Guard against learning bad correction rules from mined pairs.

    A mined `wrong → right` pair is only safe to persist when:
      * `wrong` is a genuine non-word (zipf < 2.0, i.e. not is_known) — we
        never rewrite real words the user might legitimately dictate, and
      * `wrong` isn't a real word in any protected language (a bilingual
        user's Spanish "harina" must not be rewritten because it looks like
        a non-word in English), and
      * `right` is either a known-common word (is_common) or already a term
        in the user's vocabulary file — so we don't cement one misspelling
        into another (the historic "manteka"/"wordall" bug).

    Returns ``None`` when the pair passes, otherwise a short human-readable
    reason string explaining why it was rejected.
    """
    from wordfreq import zipf_frequency

    wrong = (wrong or "").strip()
    right = (right or "").strip()
    if not wrong or not right:
        return "empty term"
    vocab_lower = {v.lower() for v in (vocabulary or set())}
    if is_known(wrong, language):
        return f'"{wrong}" is a real word — refusing to auto-correct it'
    for pl in protect_languages:
        if pl != language and zipf_frequency(wrong.lower(), pl) >= _PROTECT_ZIPF:
            return f'"{wrong}" is a real word in protected language "{pl}"'
    if not (is_common(right, language) or right.lower() in vocab_lower):
        return f'"{right}" is neither a common word nor in your vocabulary'
    return None


# ── Suspicious word detection ────────────────────────────────────────────

@dataclass
class SuspiciousWord:
    """A word that might be a Whisper transcription error."""
    word: str
    count: int                       # how often it appears in history
    contexts: list[str] = field(default_factory=list)  # example sentences
    similar_common: list[str] = field(default_factory=list)  # nearby common words
    reason: str = ""                 # why it's suspicious


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


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

    common_list = top_n_list(language, 20000)
    common_set = {w: i for i, w in enumerate(common_list)}

    candidates: list[SuspiciousWord] = []

    for word, count in word_counts.items():
        if len(word) < min_word_length:
            continue
        if word in skip_lower or word in vocab_lower or word in dismiss_lower:
            continue
        if is_known(word, language):
            continue
        similar = _find_similar_common(word, common_set, max_distance=2)
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


def _find_similar_common(
    word: str, common_words: dict[str, int], max_distance: int = 2,
) -> list[str]:
    """Find common words within Levenshtein distance of the target word."""
    results = []
    word_len = len(word)
    for common, freq_rank in common_words.items():
        if abs(len(common) - word_len) > max_distance:
            continue
        if _levenshtein(word, common) <= max_distance:
            results.append(common)
    results.sort(key=lambda w: common_words.get(w, 99999))
    return results[:5]


# ── Levenshtein clustering ───────────────────────────────────────────────
