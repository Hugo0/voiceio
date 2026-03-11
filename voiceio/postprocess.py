"""Rule-based text cleanup for Whisper output. Near-zero latency."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.commands import CommandProcessor
    from voiceio.corrections import CorrectionDict
    from voiceio.llm import LLMProcessor

# Languages that don't use letter casing
_NO_CASE_LANGUAGES = frozenset({"zh", "ja", "ko", "ar", "he", "th", "hi", "bn", "ka", "my"})


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
    number_conversion: bool = False,
    language: str = "en",
    commands: CommandProcessor | None = None,
    corrections: CorrectionDict | None = None,
    llm: LLMProcessor | None = None,
    final: bool = False,
) -> tuple[str, bool]:
    """Shared post-processing pipeline used by both streaming and batch modes.

    Returns (processed_text, abort). If abort is True, the caller should
    discard the result (e.g. undo/flag command was triggered).
    """
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

    if final and llm and text:
        text = llm.process(text)

    return text, False
