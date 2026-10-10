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
            entry = corrections.get(re.sub(r"\s+", " ", m.group(0)).lower())
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


# What makes a correction rule worth a second look. Rules mined by the old
# cloud loop often learned a context instead of a mishearing ("virtualize" →
# "wake you"), or rewrote a real word the user may actually say.
_UNRELATED_BELOW = 0.4
_REAL_WORD_ZIPF = 1.5
# A real word in a bilingual user's other language ("harina") is protected
# only when reasonably common there, so a rare cognate can't block a fix.
_PROTECT_ZIPF = 3.0


def rule_warnings(
    wrong: str, right: str, *,
    language: str = "en",
    protect_languages: tuple[str, ...] | list[str] = (),
    vocabulary: set[str] | None = None,
) -> list[str]:
    """Why `wrong → right` might rewrite the wrong words; empty when it looks safe.

    One edit away is a typo fix ("andd" → "and"); a real word rewritten
    further is a different word ("debugged" → "debuted"). With `vocabulary`,
    a single-word target must also be a common word or one of your terms, so
    one misspelling can't be cemented into another.
    """
    from difflib import SequenceMatcher

    from wordfreq import zipf_frequency

    from voiceio.wordfreq import is_common, levenshtein
    language = "en" if language in ("", "auto") else language
    w, r = wrong.strip().lower(), right.strip().lower()
    if not w or not r:
        return ["empty term"]

    def zipf(lang: str) -> float:
        try:
            return zipf_frequency(w, lang) if " " not in w else 0.0
        except LookupError:   # a language without a word list
            return 0.0

    why = []
    if SequenceMatcher(None, w, r).ratio() < _UNRELATED_BELOW:
        why.append("replacement looks unrelated")
    if zipf(language) >= _REAL_WORD_ZIPF and levenshtein(w, r) > 1:
        why.append(f'"{wrong}" is a real word')
    why += [f'"{wrong}" is a real word in "{pl}"' for pl in protect_languages
            if pl != language and zipf(pl) >= _PROTECT_ZIPF]
    if (vocabulary is not None and " " not in r and not is_common(right, language)
            and r not in {v.lower() for v in vocabulary}):
        why.append(f'"{right}" is neither a common word nor in your vocabulary')
    return why


def risky_rules(rules: dict[str, str], language: str = "en") -> list[tuple[str, str, str]]:
    """(wrong, right, why) for each rule worth reviewing, most suspicious first."""
    from difflib import SequenceMatcher

    from wordfreq import zipf_frequency
    language = "en" if language in ("", "auto") else language
    out = []
    for wrong, right in rules.items():
        why = rule_warnings(wrong, right, language=language)
        if not why:
            continue
        sim = SequenceMatcher(None, wrong.lower(), right.lower()).ratio()
        try:
            zipf = zipf_frequency(wrong.lower(), language) if " " not in wrong else 0.0
        except LookupError:
            zipf = 0.0
        out.append((sim - zipf / 10, wrong, right, ", ".join(why)))
    return [(w, r, why) for _, w, r, why in sorted(out)]


def rule_evidence(rules: dict[str, str], clips, labels) -> dict[str, tuple[int, int]]:
    """For each rule that fired on a labeled dictation: (supported,
    contradicted) — whether the reference transcript has the replacement or
    kept the original word. Your own audio is the judge here, not word lists."""
    import re
    out: dict[str, list[int]] = {}
    for clip in clips:
        lab = labels.get(clip.audio)
        if lab is None or not clip.raw:
            continue
        raw, ref = clip.raw.lower(), lab.text.lower()
        for wrong, right in rules.items():
            pat = re.compile(rf"\b{re.escape(wrong.lower())}\b")
            if not pat.search(raw):
                continue
            counts = out.setdefault(wrong, [0, 0])
            if pat.search(ref):
                counts[1] += 1
            elif right.lower() in ref:
                counts[0] += 1
    return {w: (s, c) for w, (s, c) in out.items()}
