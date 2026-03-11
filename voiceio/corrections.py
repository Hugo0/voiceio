"""Corrections dictionary: auto-replace misheard words in transcription output."""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

from voiceio.config import CORRECTIONS_PATH, FLAGGED_PATH

log = logging.getLogger(__name__)


class CorrectionDict:
    """Thread-safe corrections dictionary with file persistence."""

    def __init__(self, path: Path | None = None):
        self._path = path or CORRECTIONS_PATH
        self._flagged_path = (
            path.parent / (path.stem + ".flagged.txt") if path else FLAGGED_PATH
        )
        self._lock = threading.Lock()
        # {lowered_wrong: (original_wrong, right)}
        self._corrections: dict[str, tuple[str, str]] = {}
        self._regex: re.Pattern | None = None
        self._mtime: float = 0.0  # tracks file mtime to skip redundant reloads
        self.load()

    def load(self) -> None:
        """Read corrections.json from disk. Skips reload if file hasn't changed."""
        with self._lock:
            try:
                st = self._path.stat()
                if st.st_mtime == self._mtime and self._corrections:
                    return  # file unchanged since last load
                self._mtime = st.st_mtime
            except OSError:
                self._corrections.clear()
                self._regex = None
                self._mtime = 0.0
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    log.warning("corrections.json is not a JSON object, ignoring")
                    return
                self._corrections.clear()
                for wrong, right in raw.items():
                    if isinstance(wrong, str) and isinstance(right, str):
                        self._corrections[wrong.lower()] = (wrong, right)
                self._rebuild_regex()
                if self._corrections:
                    log.info("Loaded %d correction(s)", len(self._corrections))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load corrections: %s", e)

    def save(self) -> None:
        """Write corrections.json atomically."""
        with self._lock:
            data = {orig: right for orig, right in self._corrections.values()}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            tmp.replace(self._path)

    def add(self, wrong: str, right: str) -> None:
        """Add a correction and save immediately."""
        with self._lock:
            self._corrections[wrong.lower()] = (wrong, right)
            self._rebuild_regex()
        self.save()
        log.info("Added correction: '%s' → '%s'", wrong, right)

    def remove(self, wrong: str) -> bool:
        """Remove a correction. Returns True if found."""
        with self._lock:
            key = wrong.lower()
            if key not in self._corrections:
                return False
            del self._corrections[key]
            self._rebuild_regex()
        self.save()
        log.info("Removed correction: '%s'", wrong)
        return True

    def list_all(self) -> dict[str, str]:
        """Return all corrections as {original_wrong: right}."""
        with self._lock:
            return {orig: right for orig, right in self._corrections.values()}

    def apply(self, text: str) -> str:
        """Apply all corrections to text. Whole-word, case-insensitive."""
        with self._lock:
            if not self._regex:
                return text
            regex = self._regex
            corrections = dict(self._corrections)

        def _replace(m: re.Match) -> str:
            key = re.sub(r"\s+", " ", m.group(0)).lower()
            entry = corrections.get(key)
            return entry[1] if entry else m.group(0)

        return regex.sub(_replace, text)

    def flag_word(self, word: str) -> None:
        """Append a word to the flagged list."""
        if not word or not word.strip():
            return
        word = word.strip()
        try:
            self._flagged_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._flagged_path, "a", encoding="utf-8") as f:
                f.write(word + "\n")
            log.info("Flagged word: '%s'", word)
        except OSError as e:
            log.warning("Failed to flag word: %s", e)

    def list_flagged(self) -> list[str]:
        """Read flagged words from disk."""
        if not self._flagged_path.exists():
            return []
        try:
            lines = self._flagged_path.read_text(encoding="utf-8").splitlines()
            return [w.strip() for w in lines if w.strip()]
        except OSError:
            return []

    def clear_flagged(self) -> None:
        """Clear the flagged words list."""
        try:
            self._flagged_path.unlink(missing_ok=True)
        except OSError:
            pass

    def vocabulary_terms(self) -> list[str]:
        """Return correction target values for Whisper initial_prompt conditioning."""
        with self._lock:
            return [right for _, right in self._corrections.values()]

    def _rebuild_regex(self) -> None:
        """Rebuild the compiled regex from current corrections. Must hold lock."""
        if not self._corrections:
            self._regex = None
            return
        # Sort by key length descending so longer matches win
        keys = sorted(self._corrections.keys(), key=len, reverse=True)
        # Escape each key and replace spaces with \s+ for multi-word matching
        patterns = []
        for key in keys:
            words = key.split()
            if len(words) > 1:
                pattern = r"\s+".join(re.escape(w) for w in words)
            else:
                pattern = re.escape(key)
            patterns.append(pattern)
        combined = "|".join(patterns)
        self._regex = re.compile(rf"\b(?:{combined})\b", re.IGNORECASE)
