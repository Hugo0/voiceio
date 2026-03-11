"""Word frequency lookup for identifying uncommon words in dictation."""
from __future__ import annotations

import re

from wordfreq import zipf_frequency

# Zipf frequency threshold: words at or above this are considered "known".
# zipf 2.0 ≈ top ~30K words. Real Whisper errors score 0.0.
_ZIPF_KNOWN_THRESHOLD = 2.0

# Zipf frequency for "common" words (top ~5K). Used by autocorrect to find
# plausible corrections — only common words are suggested as replacements.
_ZIPF_COMMON_THRESHOLD = 3.5

# Common contractions — wordfreq doesn't always score these well
_CONTRACTIONS = frozenset({
    "i'm", "i've", "i'll", "i'd",
    "we're", "we've", "we'll", "we'd",
    "you're", "you've", "you'll", "you'd",
    "they're", "they've", "they'll", "they'd",
    "he's", "he'd", "he'll",
    "she's", "she'd", "she'll",
    "it's", "it'll",
    "that's", "that'll", "that'd",
    "there's", "there'll", "there'd",
    "here's",
    "what's", "what're", "what'll", "what'd",
    "who's", "who're", "who'll", "who'd",
    "where's", "where'd",
    "when's", "when'd",
    "why's", "why'd",
    "how's", "how'd",
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "won't", "wouldn't", "couldn't", "shouldn't",
    "can't", "cannot", "couldn't",
    "haven't", "hasn't", "hadn't",
    "let's",
    "ain't",
})


def is_common(word: str, language: str = "en", threshold: float = _ZIPF_COMMON_THRESHOLD) -> bool:
    """Check if a word is common (high frequency). Used for suggesting corrections."""
    return zipf_frequency(word.lower(), language) >= threshold


def rank(word: str, language: str = "en") -> int | None:
    """Get approximate frequency rank. None if not found."""
    z = zipf_frequency(word.lower(), language)
    if z == 0.0:
        return None
    # Convert zipf to approximate rank (higher zipf = lower rank number)
    # zipf 7 ≈ rank 1, zipf 3 ≈ rank 5000
    return max(0, int(10 ** (7.5 - z)))


def is_known(word: str, language: str = "en") -> bool:
    """Check if a word is recognized in the language."""
    wl = word.lower()
    if wl in _CONTRACTIONS:
        return True
    return zipf_frequency(wl, language) >= _ZIPF_KNOWN_THRESHOLD


def extract_words(text: str) -> list[str]:
    """Extract individual words from text, stripping punctuation."""
    return re.findall(r"[a-zA-Z\u00C0-\u024F']+", text.lower())
