"""Rule-based text cleanup for Whisper output. Near-zero latency."""
from __future__ import annotations

import re

# Languages that don't use letter casing
_NO_CASE_LANGUAGES = frozenset({"zh", "ja", "ko", "ar", "he", "th", "hi", "bn", "ka", "my"})

# Filler SOUNDS only — tokens with no lexical meaning, so deleting them can
# never change meaning. Deliberately conservative because this runs on every
# user's speech:
#   * CASE-SENSITIVE (no IGNORECASE): all-caps abbreviations that look like
#     fillers are real words and must survive — "ER" (emergency room), "UM"
#     (University of Michigan), "HM". We match lowercase and Title-case forms
#     ("um", "Um") — Whisper's filler spellings — but never all-caps.
#   * Excluded entirely: "er"/"erm" ("ER"/"Er"=erbium), "mm" (millimetres),
#     "ah" (interjection), bare "hm" (hectometre — require "hmm", 2+ m's).
#   * Surrounding whitespace is horizontal-only ([^\S\n]) so a filler on its
#     own line doesn't swallow the paragraph/list break around it.
# Word repetitions ("had had" is valid English) and filler "like"/"you know"
# need judgment and are left to the LLM layer. Order matters: multi-token
# "uh-huh" before "[Uu]h+" so it isn't clipped to a stray "-huh".
_FILLER_RE = re.compile(
    r"[^\S\n]*,?[^\S\n]*\b(?:[Uu]h[-\s]?huh|[Mm]hm|[Uu]h+m*|[Uu]m+|[Hh]m{2,})\b[^\S\n]*,?[^\S\n]*",
)
# A word (2+ letters, not part of a dotted name or an ellipsis) ending in "."
# and followed by a lowercase word: a segment boundary, not a sentence end.
_SEGMENT_STOP_RE = re.compile(r"(?<![.\w])([A-Za-zÀ-ɏ]{2,})\.(\s+)(?=[a-zà-ÿ])")
_ABBREVIATIONS = frozenset({
    "etc", "vs", "approx", "cf", "al", "mr", "mrs", "ms", "dr", "st", "jr", "sr",
    "inc", "ltd", "co", "fig", "eq", "vol", "no", "pp", "ca", "min", "max", "sec",
})


def _not_an_ending(word: str) -> bool:
    """The word before a "." is an abbreviation or a dotted name, not a
    sentence's last word."""
    return "." in word or word.lower() in _ABBREVIATIONS


# A re-decode artifact is a whole duplicated sentence; require this many words
# so emphatic short repeats ("No. No.", "Stop. Stop.") are preserved.
_MIN_DEDUP_WORDS = 4


def strip_disfluencies(text: str) -> str:
    """Delete-only, meaning-safe disfluency cleanup (regex layer).

    Removes filler sounds (um, uh, er, …) and exact duplicate adjacent
    sentences (a Whisper re-decode artifact). Only ever deletes — never
    rephrases, reorders, or touches lexical words — so meaning is preserved by
    construction. The judgment cases (false starts, filler "like", word
    repetitions) are handled by the guarded LLM layer in postcorrect.
    """
    if not text:
        return text
    text = _FILLER_RE.sub(" ", text)
    text = _dedup_adjacent_sentences(text)
    # Repair the debris the deletions leave, without crossing newlines (so
    # paragraph/list structure survives even when punctuation_cleanup is off).
    text = re.sub(r"[^\S\n]+([,.;:?!])", r"\1", text)  # space before punctuation
    text = re.sub(r"[^\S\n]{2,}", " ", text)           # collapse runs of spaces
    return text.strip()


def _dedup_adjacent_sentences(text: str) -> str:
    """Drop a sentence identical to the one immediately before it.

    Separators are captured and preserved on rejoin so paragraph/list breaks
    survive; only a full (>= _MIN_DEDUP_WORDS) verbatim repeat is removed.
    """
    tokens = re.split(r"((?<=[.?!])\s+)", text)  # [sent, sep, sent, sep, …]
    out: list[str] = []
    last_kept: str | None = None
    for i in range(0, len(tokens), 2):
        sentence = tokens[i]
        sep = tokens[i + 1] if i + 1 < len(tokens) else ""
        norm = sentence.strip().lower()
        if (last_kept is not None and norm == last_kept
                and len(sentence.split()) >= _MIN_DEDUP_WORDS):
            continue  # drop the duplicate sentence and its separator
        out.append(sentence)
        out.append(sep)
        last_kept = norm
    return "".join(out)


def cleanup(text: str, language: str = "en") -> str:
    """Clean up Whisper output. Idempotent — safe to run multiple times."""
    if not text:
        return text

    text = text.strip()
    if not text:
        return text

    # Normalize multiple spaces to single
    text = re.sub(r" {2,}", " ", text)

    # Whisper ends a segment with "..." and opens the next with "..." when a
    # thought runs across them: one pause, not "......".
    text = re.sub(r"\.\.\.\s*\.\.\.\s*", "... ", text).rstrip()

    # Ensure space after ?/! before a letter. Not after ".": on real
    # dictation every "x.Y" was a domain, file or identifier ("claude.ai",
    # "roadmap.md", "Print.Observer"), never a missing sentence space.
    text = re.sub(r"([?!])([A-Za-z\u00C0-\u024F])", r"\1 \2", text)

    # Remove space before punctuation marks
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)

    if language not in _NO_CASE_LANGUAGES:
        # Whisper closes every segment with "." even mid-sentence, and the next
        # segment then carries on in lowercase ("a bit bigger. and remove the
        # latch"): on real dictation that was 1 in 4 notes. A lowercase
        # continuation means the sentence did not end, so it gets a comma —
        # before the capitalization below would make the false stop permanent.
        text = _SEGMENT_STOP_RE.sub(
            lambda m: m.group(0) if m.group(1).lower() in _ABBREVIATIONS
            else m.group(1) + "," + m.group(2),
            text,
        )


        # Capitalize first character
        text = text[0].upper() + text[1:]

        # Capitalize after sentence-ending punctuation (an ellipsis is a
        # pause, not an ending, and so is the dot of "etc.", "e.g." or a name
        # like "hugo0.com")
        text = re.sub(
            r"(\S*)(?<!\.\.)([.?!]\s+)([a-z\u00E0-\u00FF])",
            lambda m: m.group(0) if m.group(2)[0] == "." and _not_an_ending(m.group(1))
            else m.group(1) + m.group(2) + m.group(3).upper(),
            text,
        )

    return text
