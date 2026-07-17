"""Load and rank custom vocabulary for Whisper hotword conditioning.

The hotwords channel is small and fixed: faster-whisper caps it at 223 tokens
and proper nouns — the entire reason a custom vocabulary exists — cost 5-7
tokens each. So only ~40-60 terms can ever reach the decoder, while
vocabulary.txt grows without bound and is never pruned.

That makes *selection* permanent, not incidental. It used to be decided by line
position: `add_terms` appends and the loader dropped from the tail, so the loop
deleted exactly what it learned and the surviving terms were simply the oldest.
Selection is now by usage (see vocab_stats), measured against a real token
budget (see tokens) rather than a character-count proxy.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.config import ModelConfig
    from voiceio.vocab_stats import VocabStats

log = logging.getLogger(__name__)

# Recency half-life for the usage score. A term dictated today should outrank a
# term used heavily months ago: vocabulary tracks what you're working on now.
_RECENCY_HALFLIFE_SECS = 30 * 86400.0


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


def load_terms(model_cfg: ModelConfig) -> list[str]:
    """Load every vocabulary term, unranked and untruncated.

    The full list — callers that must not see a truncated view (postcorrect's
    LLM, which has no token budget; the mining gate, which asks "is this term
    already in your vocabulary?") use this.
    """
    path = resolve_vocab_path(model_cfg)

    if not path.exists():
        if model_cfg.vocabulary_file:
            log.warning("Vocabulary file not found: %s", path)
        return []

    return _read_terms(path)


def _score(term: str, stats: VocabStats | None, now: float) -> float:
    """Usage score: hits, decayed by how long since the term was last dictated.

    Unscored terms score 0 and keep their file order, so a cold start (no stats
    yet) degrades exactly to the previous behaviour rather than to something
    arbitrary.
    """
    if stats is None:
        return 0.0
    rec = stats.get(term)
    hits = float(rec.get("hits", 0) or 0)
    if hits <= 0:
        return 0.0
    last = float(rec.get("last_seen_ts", 0) or 0)
    if last <= 0:
        return hits
    age = max(0.0, now - last)
    return hits * (0.5 ** (age / _RECENCY_HALFLIFE_SECS))


def select_terms(
    terms: list[str],
    *,
    token_budget: int,
    model_name: str,
    stats: VocabStats | None = None,
    now: float | None = None,
) -> list[str]:
    """Pick the highest-scoring terms that fit `token_budget`, whole terms only.

    Ranked by usage so the budget goes to what you actually dictate, then filled
    term-by-term. Never slices mid-term: the old code did `vocab[:600]`, which
    could emit a half-word and feed the decoder a partial token.

    Ties (including the all-zero cold start) fall back to file order, so this is
    a strict improvement on the previous behaviour rather than a reshuffle.
    """
    import time

    from voiceio.tokens import count_tokens

    if not terms or token_budget <= 0:
        return []

    now = now if now is not None else time.time()
    ranked = [t for _, t in sorted(
        enumerate(terms),
        key=lambda it: (-_score(it[1], stats, now), it[0]),
    )]

    # Binary-search the longest ranked prefix that fits. Costing terms one by
    # one and summing overcounts badly — BPE merges across the ", " separators,
    # so per-term counts double-count the joins and spend the budget ~2x too
    # fast. Measuring the joined string is the only honest count, and a prefix
    # search needs ~log2(n) encodes instead of n.
    lo, hi = 0, len(ranked)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(", ".join(ranked[:mid]), model_name) <= token_budget:
            lo = mid
        else:
            hi = mid - 1

    selected = ranked[:lo]
    # Emit in ranked order — faster-whisper truncates hotwords from the tail, so
    # anything dropped downstream is the lowest-scoring term.
    if len(selected) < len(terms):
        log.info(
            "Vocabulary: %d/%d terms fit the %d-token hotword budget (%d tokens)",
            len(selected), len(terms), token_budget,
            count_tokens(", ".join(selected), model_name),
        )
    return selected


def load_vocabulary(model_cfg: ModelConfig) -> str:
    """Load vocabulary as a comma-separated string, unranked and untruncated.

    Kept for callers that just want the raw list as text. The decoder path goes
    through `select_terms` instead — this string is NOT budget-safe.
    """
    return ", ".join(load_terms(model_cfg))


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

    `get()` re-reads and re-ranks only when vocabulary.txt or vocab_stats.json
    changed; otherwise it pays two `stat` calls and returns the cached result.
    This lets the daemon pick up `voiceio correct` edits without a restart while
    keeping `_do_start` cheap — the tokenizer load (~53ms) and ranking (~17ms)
    happen on a cache miss, which is rare, never on every recording.
    """

    def __init__(self, model_cfg: ModelConfig):
        self._model_cfg = model_cfg
        self._mtime: float | None = None
        self._stats_mtime: float | None = None
        self._budget: int | None = None
        self._all: list[str] = []
        self._selected: list[str] = []
        self._loaded_all = False
        self._selection_valid = False

    @staticmethod
    def _mtime_of(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _ensure_all(self) -> None:
        mtime = self._mtime_of(resolve_vocab_path(self._model_cfg))
        if self._loaded_all and mtime == self._mtime:
            return
        self._all = load_terms(self._model_cfg)
        self._mtime = mtime
        self._loaded_all = True
        self._selection_valid = False  # terms changed → re-rank

    def get_all(self) -> list[str]:
        """Every term, untruncated — for postcorrect, which has no token budget."""
        self._ensure_all()
        return self._all

    def get_selected(self, *, token_budget: int) -> list[str]:
        """Highest-scoring terms that fit the hotword token budget.

        Re-ranks only when the vocabulary, the usage stats, or the budget
        changed — the tokenizer load and encode happen here, not per recording.
        """
        from voiceio.vocab_stats import VocabStats, _path as stats_path

        self._ensure_all()
        smtime = self._mtime_of(stats_path())
        if (self._selection_valid and smtime == self._stats_mtime
                and self._budget == token_budget):
            return self._selected

        stats = VocabStats()
        stats.load()
        self._selected = select_terms(
            self._all,
            token_budget=token_budget,
            model_name=self._model_cfg.name,
            stats=stats,
        )
        self._stats_mtime = smtime
        self._budget = token_budget
        self._selection_valid = True
        return self._selected

    def get(self) -> str:
        """Full vocabulary as a string (NOT budget-safe — see get_selected)."""
        return ", ".join(self.get_all())
