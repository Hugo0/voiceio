"""Voice command detection and replacement in transcribed text."""
from __future__ import annotations

import re
from dataclasses import dataclass

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
    # Ensure space after punctuation before a word char — but never after "."
    # or ":" inside a token: "Call.com", "Print.Observer", "host:port".
    text = re.sub(r"([,;?!])([A-Za-z\u00C0-\u024F])", r"\1 \2", text)
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


@dataclass(frozen=True)
class ParsedCommands:
    text: str
    undo: bool = False              # "scratch that" / "undo that"
    flagged_word: str | None = None  # "correct that": the word before it ("" if none)


class CommandProcessor:
    """Detects and replaces voice commands in transcribed text. Stateless:
    the result carries what was requested, so one instance can serve
    overlapping sessions."""

    def __init__(self, enabled: bool = True, editing: bool = False):
        self._enabled = enabled
        self._commands = {
            k: v for k, v in DEFAULT_COMMANDS.items()
            if editing or v not in (_UNDO_SENTINEL, _FLAG_SENTINEL)
        }
        self._max_words = max(len(k) for k in self._commands) if self._commands else 1

    def process(self, text: str) -> str:
        """Replace voice commands; the text only."""
        return self.parse(text).text

    def parse(self, text: str) -> ParsedCommands:
        if not self._enabled or not text:
            return ParsedCommands(text)

        words = text.split()
        result: list[str] = []
        replaced = False
        i = 0

        while i < len(words):
            matched = False
            # Try longest match first
            for length in range(min(self._max_words, len(words) - i), 0, -1):
                key = tuple(_strip_punct(w) for w in words[i:i + length])
                if key in self._commands:
                    replacement = self._commands[key]
                    if replacement == _UNDO_SENTINEL:
                        return ParsedCommands(_normalize_spacing(" ".join(result)), undo=True)
                    if replacement == _FLAG_SENTINEL:
                        # Capture the last word before "correct that"
                        flagged = result.pop() if result else ""
                        return ParsedCommands(
                            _normalize_spacing(" ".join(result)), flagged_word=flagged)
                    result.append(replacement)
                    i += length
                    matched = replaced = True
                    break
            if not matched:
                result.append(words[i])
                i += 1

        if not replaced:
            return ParsedCommands(text)  # no command spoken: leave the text alone
        return ParsedCommands(_normalize_spacing(" ".join(result)))
