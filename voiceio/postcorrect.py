"""Constrained LLM post-correction of final transcripts.

A final-pass-only rewrite that fixes misrecognized words (wrong proper nouns,
homophones, garbled technical terms) without rephrasing. Uses the cloud
OpenAI-compatible client in ``llm_api`` (the same one autocorrect uses) — never
local Ollama, which is too weak for this task.

Every rewrite is tightly guarded: the LLM only ever *replaces* the original if
the edit is small (word-level edit ratio and word-count change both bounded).
On any failure, timeout, empty response, or over-eager rewrite, the original
text is returned unchanged.
"""
from __future__ import annotations

import dataclasses
import difflib
import logging
import threading
import time

from voiceio.config import AutocorrectConfig, Config

log = logging.getLogger(__name__)

# Guard thresholds — an ASR fix touches a handful of words, never a rewrite.
_MAX_EDIT_RATIO = 0.3    # word-level SequenceMatcher edit ratio ceiling
_MAX_WORDCOUNT_DELTA = 0.2  # allowed relative change in word count

# Disfluency-mode guards. The whole promise is "never change meaning", enforced
# structurally on the word-level diff: no INSERTIONS (can't add or rephrase),
# few REPLACEMENTS (ASR fixes only, not reword), bounded DELETIONS (can't nuke
# real content). Deletions are what disfluency removal legitimately does.
_MAX_INSERTED_WORDS = 0     # adding any word = altering meaning → reject
_MAX_REPLACE_FRAC = 0.15    # ASR word-fixes only, never wholesale rewording
_MAX_DELETE_FRAC = 0.4      # backstop against deleting real content

# Words whose deletion/replacement flips meaning — the fraction caps can't catch
# a single dropped "not". If the edit touches any of these on the original side,
# reject outright. (Contractions ending in "n't" are handled separately.)
_MEANING_CRITICAL = frozenset({
    "not", "no", "never", "none", "nor", "neither", "without", "cannot",
    "nothing", "nobody", "nowhere", "n't",
})


def _is_meaning_critical(word: str) -> bool:
    w = word.lower().strip(".,;:!?\"'()")
    return w in _MEANING_CRITICAL or w.endswith("n't")

_SYSTEM_PROMPT = (
    "You fix automatic speech recognition errors in dictated text. "
    "The user dictates about software engineering and their projects. "
    "Fix ONLY misrecognized words (wrong proper nouns, homophone errors, "
    "garbled technical terms). NEVER rephrase, summarize, add or remove "
    "content, or change style/punctuation beyond the fixed words. "
    "Return only the corrected text with no commentary."
)

# Disfluency mode: also strip spoken filler, delete-only. The strict rules
# mirror the guards — the model is told exactly what the diff check enforces.
_SYSTEM_PROMPT_CLEAN = (
    "You convert dictated speech into clean, concise written text. The user "
    "dictates about software engineering and their projects, and wants it to "
    "read like something they wrote, not spoke. Do exactly two things:\n"
    "1. Fix words the recognizer misheard (wrong proper nouns, homophones, "
    "garbled technical terms).\n"
    "2. Cut spoken disfluency so it reads tight and clear:\n"
    "   - filler sounds (um, uh, er);\n"
    "   - filler discourse markers with no content: leading 'so', 'yeah', "
    "'okay', 'well', 'now', 'right', and 'you know', 'I mean', 'like', 'sort "
    "of'/'kind of' when used as filler, 'or something'/'or whatever', a "
    "trailing 'right?' tag;\n"
    "   - false starts and self-corrections (keep only the corrected version);\n"
    "   - stray word and phrase repetitions.\n"
    "Lean toward the briefer version — remove hesitation and padding freely.\n"
    "HARD LIMITS (these protect meaning): only DELETE and FIX — NEVER add "
    "words, rephrase, reword, reorder, or summarize. NEVER drop or alter actual "
    "content, a negation ('not', 'never', \"n't\"), or a hedge that changes "
    "certainty ('I think', 'maybe', 'probably', 'might'). Keep the speaker's own "
    "wording. Return only the cleaned text, nothing else."
)

_MAX_RECENT = 3


def _strip_wrapping(text: str) -> str:
    """Strip markdown fences and a single layer of surrounding quotes."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Strip one matching pair of surrounding quotes.
    for q in ('"', "'", "`"):
        if len(text) >= 2 and text[0] == q and text[-1] == q:
            text = text[1:-1].strip()
            break
    return text


def _word_edit_ratio(a: str, b: str) -> float:
    """Fraction of words changed between two texts (0.0 = identical)."""
    aw, bw = a.split(), b.split()
    if not aw and not bw:
        return 0.0
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    return 1.0 - sm.ratio()


def _changed_words(a: str, b: str) -> list[str]:
    """Compact list of 'old→new' word changes for logging."""
    aw, bw = a.split(), b.split()
    changes: list[str] = []
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(aw[i1:i2]) or "∅"
        new = " ".join(bw[j1:j2]) or "∅"
        changes.append(f"{old}→{new}")
    return changes


class PostCorrector:
    """Final-pass LLM corrector with mandatory sanity guards."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._pc = cfg.postcorrect
        self._remove_disfluencies = cfg.output.remove_disfluencies
        # API key / base_url resolution is shared with [autocorrect].
        self._ac = cfg.autocorrect
        self._available: bool | None = None
        # Wall-clock seconds of the most recent LLM call (latency metrics).
        self.last_secs: float | None = None
        # Per-recording context, set by the app before each utterance.
        self._vocabulary = ""
        self._recent: list[str] = []
        self._context: str | None = None
        # Context actually sent with the in-progress correct() call (what
        # _record must persist — may differ from self._context).
        self._effective_ctx: str | None = None
        # A worker abandoned at the deadline may block indefinitely on a hung
        # endpoint; cap leakage at one thread/socket by skipping new calls
        # while it is still alive.
        self._abandoned: threading.Thread | None = None

    # ── availability ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when enabled and an API key resolves. Cached after first check."""
        if not self._pc.enabled:
            return False
        if self._available is None:
            from voiceio.llm_api import resolve_api_key
            self._available = bool(resolve_api_key(self._ac))
            if not self._available:
                log.debug("PostCorrector disabled: no API key resolved")
        return self._available

    # ── context ─────────────────────────────────────────────────────────

    def set_context(
        self, vocabulary: str = "", recent: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        """Set the context used for the next correction(s)."""
        self._vocabulary = vocabulary or ""
        self._recent = list(recent or [])
        self._context = title

    # ── correction ──────────────────────────────────────────────────────

    def _client_cfg(self) -> AutocorrectConfig:
        """AutocorrectConfig carrying the postcorrect model + timeout."""
        return dataclasses.replace(
            self._ac,
            model=self._pc.model or self._ac.model,
            timeout_secs=self._pc.timeout_secs,
        )

    def _build_user_message(self, text: str, vocabulary: str, recent, context) -> str:
        parts: list[str] = []
        if vocabulary:
            parts.append(
                "The user's known vocabulary (correct spellings may appear "
                f"here):\n{vocabulary}"
            )
        if recent:
            joined = "\n".join(f"- {r}" for r in recent[-_MAX_RECENT:])
            parts.append(f"Recent dictation for context:\n{joined}")
        if context:
            parts.append(f"Active window: {context}")
        parts.append(f"Transcript to correct:\n{text}")
        return "\n\n".join(parts)

    def _record(self, before: str, after: str | None, outcome: str) -> None:
        """Persist one LLM attempt (before/after/outcome) as training data.

        Pairs land in postcorrect_pairs.jsonl — unlike the rotating log they
        survive, so accepted AND rejected corrections stay available for
        tuning guards or training a local corrector later.
        """
        if not self._cfg.data.capture_intermediates:
            return
        from voiceio import retention
        from voiceio.config import POSTCORRECT_PAIRS_PATH
        retention.append_jsonl(POSTCORRECT_PAIRS_PATH, {
            "ts": time.time(),
            "before": before,
            "after": after,
            "outcome": outcome,
            "secs": round(self.last_secs, 3) if self.last_secs is not None else None,
            "model": self._pc.model or self._ac.model,
            "context": self._effective_ctx,
        })

    def correct(
        self, text: str, *, vocabulary: str = "",
        recent: list[str] | None = None, context: str | None = None,
    ) -> str:
        """Return a corrected transcript, or the original if guards reject it.

        Context args override any values set via set_context().
        """
        self.last_secs = None  # stale values must not leak into metrics
        if not text or not text.strip():
            return text
        if not self.is_available():
            return text

        if len(text.split()) < self._pc.min_words:
            log.debug("PostCorrector skip: %d words < min_words", len(text.split()))
            return text

        vocab = vocabulary or self._vocabulary
        rec = recent if recent is not None else self._recent
        ctx = context if context is not None else self._context
        self._effective_ctx = ctx

        if self._abandoned is not None:
            if self._abandoned.is_alive():
                log.warning(
                    "PostCorrector: previous request still hung — skipping this one",
                )
                self._record(text, None, "skipped_busy")
                return text
            self._abandoned = None

        from voiceio.llm_api import chat
        user_msg = self._build_user_message(text, vocab, rec, ctx)
        # timeout_secs must bound the WALL CLOCK the user waits, but urllib's
        # timeout is per-socket-read — a slowly streaming response can run
        # far past it (observed ~14s with an 8s config). Run the call in a
        # thread and abandon it at the deadline.
        t0 = time.monotonic()
        outcome: dict = {}

        system_prompt = (
            _SYSTEM_PROMPT_CLEAN if self._remove_disfluencies else _SYSTEM_PROMPT
        )

        def _call() -> None:
            try:
                outcome["response"] = chat(
                    self._client_cfg(), system_prompt, user_msg, max_tokens=1024,
                )
            except Exception as e:
                outcome["error"] = e

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        worker.join(self._pc.timeout_secs)
        self.last_secs = time.monotonic() - t0
        if worker.is_alive():
            log.debug(
                "PostCorrector deadline (%.1fs) exceeded — keeping original",
                self._pc.timeout_secs,
            )
            self._abandoned = worker
            self._record(text, None, "timeout")
            return text
        if "error" in outcome:
            log.debug("PostCorrector LLM error: %s — keeping original", outcome["error"])
            self._record(text, None, "error")
            return text
        response = outcome.get("response")

        if not response:
            log.debug("PostCorrector: empty/failed response — keeping original")
            self._record(text, None, "empty")
            return text

        corrected = _strip_wrapping(response)
        if not corrected:
            log.debug("PostCorrector: response empty after stripping — keeping original")
            self._record(text, None, "empty")
            return text

        if corrected == text:
            self._record(text, corrected, "unchanged")
            return text

        accept, reject_reason = self._guard(text, corrected)
        if not accept:
            log.debug("PostCorrector reject (%s) — keeping original", reject_reason)
            self._record(text, corrected, f"rejected_{reject_reason}")
            return text

        log.info("PostCorrector fixed: %s", ", ".join(_changed_words(text, corrected)))
        self._record(text, corrected, "applied")
        return corrected

    def _guard(self, text: str, corrected: str) -> tuple[bool, str]:
        """Decide whether the LLM's edit is within bounds.

        Returns (accept, reject_reason). In disfluency mode the promise is
        "never change meaning", enforced on the word-level diff: zero
        insertions (nothing added/rephrased), few replacements (ASR fixes, not
        rewording), bounded deletions (can't strip real content). Otherwise the
        original conservative fix-only guards apply.
        """
        aw, bw = text.split(), corrected.split()
        if not self._remove_disfluencies:
            n = len(aw)
            if n and abs(len(bw) - n) / n > _MAX_WORDCOUNT_DELTA:
                return False, "wordcount"
            if _word_edit_ratio(text, corrected) > _MAX_EDIT_RATIO:
                return False, "editratio"
            return True, ""

        inserted = replaced = deleted = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw).get_opcodes():
            if tag == "insert":
                inserted += j2 - j1
            elif tag == "replace":
                replaced += max(i2 - i1, j2 - j1)
                if any(_is_meaning_critical(w) for w in aw[i1:i2]):
                    return False, "negation"   # e.g. "not" → something else
            elif tag == "delete":
                deleted += i2 - i1
                if any(_is_meaning_critical(w) for w in aw[i1:i2]):
                    return False, "negation"   # dropping "not" inverts meaning
        n = len(aw)
        if inserted > _MAX_INSERTED_WORDS:
            return False, "inserted"          # added content — meaning changed
        if replaced > max(3, int(_MAX_REPLACE_FRAC * n)):
            return False, "reworded"           # too many substitutions = rewrite
        if n and deleted > _MAX_DELETE_FRAC * n:
            return False, "overdeleted"        # nuked real content
        return True, ""
