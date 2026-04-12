"""Voice command detection and replacement in transcribed text."""
from __future__ import annotations

import re

_UNDO_SENTINEL = "__UNDO__"
_FLAG_SENTINEL = "__FLAG__"

DEFAULT_COMMANDS: dict[tuple[str, ...], str] = {
    # Punctuation
    ("period",): ".",
    ("full", "stop"): ".",
    ("comma",): ",",
    ("question", "mark"): "?",
    ("exclamation", "point"): "!",
    ("exclamation", "mark"): "!",
    ("colon",): ":",
    ("semicolon",): ";",
    ("hyphen",): "-",
    ("dash",): " -- ",
    ("open", "quote"): '"',
    ("close", "quote"): '"',
    ("open", "paren"): "(",
    ("close", "paren"): ")",
    # Formatting
    ("new", "line"): "\n",
    ("newline",): "\n",
    ("new", "paragraph"): "\n\n",
    # Editing
    ("scratch", "that"): _UNDO_SENTINEL,
    ("undo", "that"): _UNDO_SENTINEL,
    # Corrections
    ("correct", "that"): _FLAG_SENTINEL,
}


def _strip_punct(word: str) -> str:
    """Strip trailing/leading punctuation for command matching."""
    return re.sub(r"[^\w]", "", word).lower()


def _normalize_spacing(text: str) -> str:
    """Fix spacing around punctuation after command replacement."""
    # Remove space before closing/sentence punctuation (but not newlines)
    text = re.sub(r" +([.,;:?!)])", r"\1", text)
    # Ensure space after punctuation before a word char (not at end, not newlines)
    text = re.sub(r"([.,;:?!])([A-Za-z\u00C0-\u024F])", r"\1 \2", text)
    # Remove space after opening ( but not quotes (hard to distinguish open/close)
    text = re.sub(r"([(])\s+", r"\1", text)
    # Quotes: remove internal padding (space after open-quote, before close-quote)
    # Track quote parity to distinguish open from close
    result = []
    in_quote = False
    i = 0
    while i < len(text):
        if text[i] == '"':
            if not in_quote:
                # Opening quote: remove trailing space
                result.append('"')
                in_quote = True
                # Skip space after opening quote
                if i + 1 < len(text) and text[i + 1] == " ":
                    i += 2
                    continue
            else:
                # Closing quote: remove preceding space
                if result and result[-1] == " ":
                    result.pop()
                result.append('"')
                in_quote = False
                i += 1
                continue
        result.append(text[i])
        i += 1
    text = "".join(result)
    # Clean up spaces around newlines
    text = re.sub(r" *\n *", "\n", text)
    # Clean up multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


class CommandProcessor:
    """Detects and replaces voice commands in transcribed text."""

    def __init__(self, enabled: bool = True, editing: bool = False):
        self._enabled = enabled
        self._commands = {
            k: v for k, v in DEFAULT_COMMANDS.items()
            if editing or v not in (_UNDO_SENTINEL, _FLAG_SENTINEL)
        }
        self._max_words = max(len(k) for k in self._commands) if self._commands else 1
        self.undo_requested = False
        self.flag_requested = False
        self.flagged_word = ""

    def process(self, text: str, final: bool = False) -> str:
        """Replace voice commands in text. Sets undo_requested if scratch/undo detected."""
        if not self._enabled or not text:
            return text

        self.undo_requested = False
        self.flag_requested = False
        self.flagged_word = ""
        words = text.split()
        result: list[str] = []
        i = 0

        while i < len(words):
            matched = False
            # Try longest match first
            for length in range(min(self._max_words, len(words) - i), 0, -1):
                key = tuple(_strip_punct(w) for w in words[i:i + length])
                if key in self._commands:
                    replacement = self._commands[key]
                    if replacement == _UNDO_SENTINEL:
                        self.undo_requested = True
                        return _normalize_spacing(" ".join(result))
                    if replacement == _FLAG_SENTINEL:
                        self.flag_requested = True
                        # Capture the last word before "correct that"
                        if result:
                            self.flagged_word = result.pop()
                        return _normalize_spacing(" ".join(result))
                    result.append(replacement)
                    i += length
                    matched = True
                    break
            if not matched:
                result.append(words[i])
                i += 1

        return _normalize_spacing(" ".join(result))
