"""Load custom vocabulary for Whisper initial_prompt conditioning."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.config import ModelConfig

log = logging.getLogger(__name__)

# Vocabulary feeds the hotwords channel, which has its own ~224-token budget
# (separate from initial_prompt). ~800 chars ≈ 200 tokens of typical terms.
MAX_VOCAB_CHARS = 800


def resolve_vocab_path(model_cfg: ModelConfig) -> Path:
    """Return the vocabulary file path (explicit config or default location)."""
    from voiceio.config import CONFIG_DIR

    if model_cfg.vocabulary_file:
        return Path(model_cfg.vocabulary_file).expanduser()
    return CONFIG_DIR / "vocabulary.txt"


def _read_terms(path: Path) -> list[str]:
    """Read non-comment, non-blank vocabulary terms from a file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        log.warning("Failed to read vocabulary file: %s", e)
        return []
    terms = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


def load_vocabulary(model_cfg: ModelConfig) -> str:
    """Load vocabulary terms and return a comma-separated string.

    Checks model_cfg.vocabulary_file first, then the default location.
    Returns empty string if no vocabulary file found.
    """
    path = resolve_vocab_path(model_cfg)

    if not path.exists():
        if model_cfg.vocabulary_file:
            log.warning("Vocabulary file not found: %s", path)
        return ""

    terms = _read_terms(path)
    if not terms:
        return ""

    result = ", ".join(terms)
    if len(result) > MAX_VOCAB_CHARS:
        # Truncate by dropping terms from the end
        while len(result) > MAX_VOCAB_CHARS and terms:
            terms.pop()
            result = ", ".join(terms)
        log.warning("Vocabulary truncated to %d terms (%d chars)", len(terms), len(result))

    log.info("Loaded %d vocabulary terms from %s", len(terms), path)
    return result


# A vocabulary term is a word or short phrase of letters (allowing internal
# apostrophes/hyphens and CamelCase). Rejects punctuation-only junk, numbers,
# and stray symbols that would just waste the hotword budget.
_VALID_TERM = re.compile(r"^[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*"
                         r"(?: [A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*)*$")


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _is_sane_term(term: str, existing_lower: set[str]) -> str | None:
    """Return a rejection reason for a candidate term, or None if it's sane.

    Rejects non-alpha junk, single characters, and terms that look like a
    misspelling of an existing entry (Levenshtein 1-2 to something we already
    have) — those belong in corrections, not vocabulary.
    """
    t = term.strip()
    if len(t) < 2:
        return "too short"
    if not _VALID_TERM.match(t):
        return "not a valid word/phrase"
    tl = t.lower()
    if tl in existing_lower:
        return "already present"
    for other in existing_lower:
        # Only compare single tokens of similar length — phrases legitimately
        # share many characters, and short words trivially fall within 2 edits.
        if " " in other or " " in tl:
            continue
        if abs(len(other) - len(tl)) > 2 or len(tl) < 4:
            continue
        d = _levenshtein(tl, other)
        if 1 <= d <= 2:
            return f'looks like a misspelling of existing "{other}"'
    return None


def add_terms(terms: list[str], model_cfg: ModelConfig) -> int:
    """Append vocabulary `terms` to the user's file, deduped and sanity-checked.

    Case-insensitive dedupe against existing file content (and within the
    batch). Skips terms failing the sanity check (junk, single chars, or a
    Levenshtein 1-2 near-match to an existing entry — logged as a note).
    Creates the file if missing. Returns the number of terms actually added.
    """
    path = resolve_vocab_path(model_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_terms(path) if path.exists() else []
    existing_lower = {t.lower() for t in existing}

    to_add: list[str] = []
    for term in terms:
        t = (term or "").strip()
        reason = _is_sane_term(t, existing_lower)
        if reason:
            if reason != "already present":
                log.info("Skipping vocabulary term %r: %s", t, reason)
            continue
        to_add.append(t)
        existing_lower.add(t.lower())  # dedupe within the batch too

    if not to_add:
        return 0

    needs_nl = path.exists() and path.stat().st_size > 0 and \
        not path.read_text(encoding="utf-8").endswith("\n")
    with open(path, "a", encoding="utf-8") as f:
        if needs_nl:
            f.write("\n")
        for t in to_add:
            f.write(t + "\n")
    return len(to_add)


class VocabularyLoader:
    """mtime-cached vocabulary loader for the per-recording hot path.

    `get()` re-reads the file only when its mtime changed; otherwise it just
    pays a single `stat` and returns the cached comma-separated string. This
    lets the daemon pick up `voiceio correct` vocabulary edits without a
    restart while keeping `_do_start` cheap.
    """

    def __init__(self, model_cfg: ModelConfig):
        self._model_cfg = model_cfg
        self._mtime: float | None = None
        self._cached: str = ""
        self._loaded = False

    def get(self) -> str:
        path = resolve_vocab_path(self._model_cfg)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if not self._loaded or mtime != self._mtime:
            self._cached = load_vocabulary(self._model_cfg)
            self._mtime = mtime
            self._loaded = True
        return self._cached
