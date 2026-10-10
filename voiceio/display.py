"""What a streaming session has put on screen, and how to change it.

Text reaches the focused app one of two ways. An IBus typer shows a preedit
(an underlined preview the app can't see yet) and commits once; the session
just replaces the preview. A keystroke typer has already typed every word it
showed, so a revision must be applied as an edit: append when the new text
extends what's there, and on the final pass a character-level diff that
deletes back to the first difference and retypes the rest.
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from voiceio.privacy import describe_transcript
from voiceio.typers.base import StreamingTyper

if TYPE_CHECKING:
    from voiceio.typers.base import TyperBackend

log = logging.getLogger(__name__)
DELETE_SETTLE_SECS = 0.05  # delay between delete and type for ydotool reliability


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _clean_word(w: str) -> str:
    """Strip punctuation for fuzzy word matching."""
    return re.sub(r'[^\w]', '', w).lower()


def _word_match_len(old_words: list[str], new_words: list[str]) -> int:
    """Count matching leading words, ignoring punctuation/case."""
    count = 0
    for o, n in zip(old_words, new_words):
        if _clean_word(o) == _clean_word(n):
            count += 1
        else:
            break
    return count


class TypedText:
    """The text a session has shown through `typer`."""

    def __init__(self, typer: TyperBackend):
        self.typer = typer
        self.text = ""

    def commit_preview(self) -> None:
        """Commit a preview that only exists as a preedit — when the final
        pass yields nothing, so a decode timeout doesn't drop it. Keystroke
        typers already typed it."""
        if self.text and isinstance(self.typer, StreamingTyper):
            log.warning("Final transcription empty; committing interim preedit text (%d chars)",
                        len(self.text))
            self.typer.commit_text(self.text)

    def clear(self) -> None:
        """Take back everything shown (a voice command discarded it)."""
        if isinstance(self.typer, StreamingTyper):
            self.typer.clear_preedit()
        elif self.text:
            self.typer.delete_chars(len(self.text))
        self.text = ""

    def apply(self, new_text: str, final: bool = False) -> None:
        """Show `new_text` in place of what is on screen.

        With StreamingTyper (IBus): use preedit during streaming, commit on final.
        Without: append-only via word-level matching, char-level diff on final.
        """
        old = self.text

        # Preedit path: trivial, just replace the preview text
        if isinstance(self.typer, StreamingTyper):
            if final:
                self.typer.commit_text(new_text)
                self.text = new_text
                log.debug("Preedit commit: %s", describe_transcript(new_text))
            elif new_text != old:
                self.typer.update_preedit(new_text)
                self.text = new_text
                log.debug("Preedit update: %s", describe_transcript(new_text))
            return

        if new_text == old:
            return

        if final:
            self._final_correction(new_text)
            return

        if not old:
            # First transcription, just type it
            self.typer.type_text(new_text)
            self.text = new_text
            log.debug("Initial: %s", describe_transcript(new_text))
            return

        # Short text: allow full replacement (Whisper still warming up)
        if len(old.split()) <= 2:
            self.typer.delete_chars(len(old))
            time.sleep(DELETE_SETTLE_SECS)
            self.typer.type_text(new_text)
            self.text = new_text
            log.debug("Replaced short text: %s", describe_transcript(new_text))
            return

        # Word-level append-only
        old_words = old.split()
        new_words = new_text.split()
        matched = _word_match_len(old_words, new_words)

        if matched >= len(old_words) and len(new_words) > matched:
            new_tail = " ".join(new_words[matched:])
            to_type = " " + new_tail
            self.typer.type_text(to_type)
            self.text = old + to_type
            log.debug("Appended: %s", describe_transcript(to_type.strip()))
        else:
            log.debug(
                "Skipping (matched %d/%d words)",
                matched, len(old_words),
            )

    def _final_correction(self, new_text: str) -> None:
        """Final pass: char-level diff to fix accumulated drift."""
        old = self.text
        prefix_len = _common_prefix_len(old, new_text)
        to_delete = len(old) - prefix_len
        to_type = new_text[prefix_len:]

        if to_delete > 0:
            log.debug("Final correction: delete %d, type %s", to_delete, describe_transcript(to_type))
            self.typer.delete_chars(to_delete)
            time.sleep(DELETE_SETTLE_SECS)
        if to_type:
            self.typer.type_text(to_type)

        self.text = new_text
