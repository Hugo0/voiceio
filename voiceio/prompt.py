"""Compose Whisper initial_prompt from vocabulary + recent transcript history."""
from __future__ import annotations

import collections
import threading


class PromptBuilder:
    """Builds initial_prompt combining static vocabulary with recent history.

    Thread-safe: add_transcript() and build() can be called from different threads.
    """

    def __init__(
        self,
        vocabulary: str = "",
        max_chars: int = 800,
        max_segments: int = 5,
    ):
        self._vocabulary = vocabulary
        self._max_chars = max_chars
        self._history: collections.deque[str] = collections.deque(maxlen=max_segments)
        self._lock = threading.Lock()

    def add_transcript(self, text: str) -> None:
        """Append a transcript segment to history."""
        if not text:
            return
        with self._lock:
            self._history.append(text)

    def build(self) -> str | None:
        """Compose the full prompt. Returns None if empty."""
        with self._lock:
            parts: list[str] = []

            if self._vocabulary:
                parts.append(self._vocabulary)

            if self._history:
                history_text = " ".join(self._history)
                parts.append(history_text)

            if not parts:
                return None

            result = " | ".join(parts) if len(parts) > 1 else parts[0]

            # Truncate history from the front if over budget
            if len(result) > self._max_chars and self._history:
                vocab_budget = len(self._vocabulary) + 3 if self._vocabulary else 0  # " | "
                remaining = self._max_chars - vocab_budget
                history_text = " ".join(self._history)
                if remaining > 0 and len(history_text) > remaining:
                    history_text = history_text[-remaining:]
                    # Snap to word boundary
                    space = history_text.find(" ")
                    if space > 0:
                        history_text = history_text[space + 1:]

                if self._vocabulary and history_text:
                    result = self._vocabulary + " | " + history_text
                elif self._vocabulary:
                    result = self._vocabulary
                else:
                    result = history_text

            return result if result else None

    def reset(self) -> None:
        """Clear history (vocabulary is preserved)."""
        with self._lock:
            self._history.clear()
