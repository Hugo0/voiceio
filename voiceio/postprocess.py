"""Rule-based text cleanup for Whisper output. Near-zero latency."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.commands import CommandProcessor
    from voiceio.corrections import CorrectionDict
    from voiceio.llm import LLMProcessor
    from voiceio.postcorrect import PostCorrector

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

    # Ensure space after sentence-ending punctuation before a letter
    text = re.sub(r"([.?!])([A-Za-z\u00C0-\u024F])", r"\1 \2", text)

    # Remove space before punctuation marks
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)

    if language not in _NO_CASE_LANGUAGES:
        # Capitalize first character
        text = text[0].upper() + text[1:]

        # Capitalize after sentence-ending punctuation
        text = re.sub(
            r"([.?!]\s+)([a-z\u00E0-\u00FF])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )

    return text


def apply_pipeline(
    text: str,
    *,
    do_cleanup: bool = False,
    remove_disfluencies: bool = False,
    number_conversion: bool = False,
    language: str = "en",
    commands: CommandProcessor | None = None,
    corrections: CorrectionDict | None = None,
    postcorrect: PostCorrector | None = None,
    llm: LLMProcessor | None = None,
    voice_input_prefix: str = "",
    final: bool = False,
) -> tuple[str, bool]:
    """Shared post-processing pipeline used by both streaming and batch modes.

    Returns (processed_text, abort). If abort is True, the caller should
    discard the result (e.g. undo/flag command was triggered).
    """
    # Disfluency removal runs BEFORE cleanup so cleanup re-fixes the spacing,
    # commas, and capitalization the deletions leave behind. Delete-only.
    if remove_disfluencies:
        text = strip_disfluencies(text)

    if do_cleanup:
        text = cleanup(text, language)

    if number_conversion:
        from voiceio.numbers import convert_numbers
        text = convert_numbers(text, language)

    if commands:
        text = commands.process(text, final=final)
        if commands.undo_requested or commands.flag_requested:
            if commands.flag_requested and corrections and commands.flagged_word:
                corrections.flag_word(commands.flagged_word)
            return "", True

    if corrections and text:
        text = corrections.apply(text)

    # Constrained LLM post-correction — final pass only, before generic LLM.
    if final and postcorrect and text:
        text = postcorrect.correct(text)

    if final and llm and text:
        text = llm.process(text)

    # Applied on every pass so the marker appears from the first streaming chunk.
    if voice_input_prefix and text:
        text = f"{voice_input_prefix} {text}"

    return text, False
