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

from voiceio.config import AutocorrectConfig, Config

log = logging.getLogger(__name__)

# Guard thresholds — an ASR fix touches a handful of words, never a rewrite.
_MAX_EDIT_RATIO = 0.3    # word-level SequenceMatcher edit ratio ceiling
_MAX_WORDCOUNT_DELTA = 0.2  # allowed relative change in word count

_SYSTEM_PROMPT = (
    "You fix automatic speech recognition errors in dictated text. "
    "The user dictates about software engineering and their projects. "
    "Fix ONLY misrecognized words (wrong proper nouns, homophone errors, "
    "garbled technical terms). NEVER rephrase, summarize, add or remove "
    "content, or change style/punctuation beyond the fixed words. "
    "Return only the corrected text with no commentary."
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
        # API key / base_url resolution is shared with [autocorrect].
        self._ac = cfg.autocorrect
        self._available: bool | None = None
        # Per-recording context, set by the app before each utterance.
        self._vocabulary = ""
        self._recent: list[str] = []
        self._context: str | None = None

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

    def correct(
        self, text: str, *, vocabulary: str = "",
        recent: list[str] | None = None, context: str | None = None,
    ) -> str:
        """Return a corrected transcript, or the original if guards reject it.

        Context args override any values set via set_context().
        """
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

        from voiceio.llm_api import chat
        user_msg = self._build_user_message(text, vocab, rec, ctx)
        try:
            response = chat(self._client_cfg(), _SYSTEM_PROMPT, user_msg, max_tokens=1024)
        except Exception as e:
            log.debug("PostCorrector LLM error: %s — keeping original", e)
            return text

        if not response:
            log.debug("PostCorrector: empty/failed response — keeping original")
            return text

        corrected = _strip_wrapping(response)
        if not corrected:
            log.debug("PostCorrector: response empty after stripping — keeping original")
            return text

        if corrected == text:
            return text

        # Guard: word-count must not change materially.
        orig_wc, new_wc = len(text.split()), len(corrected.split())
        if orig_wc and abs(new_wc - orig_wc) / orig_wc > _MAX_WORDCOUNT_DELTA:
            log.debug(
                "PostCorrector reject: word count %d→%d (>%.0f%%) — keeping original",
                orig_wc, new_wc, _MAX_WORDCOUNT_DELTA * 100,
            )
            return text

        # Guard: only a small fraction of words may change.
        ratio = _word_edit_ratio(text, corrected)
        if ratio > _MAX_EDIT_RATIO:
            log.debug(
                "PostCorrector reject: edit ratio %.2f > %.2f — keeping original",
                ratio, _MAX_EDIT_RATIO,
            )
            return text

        log.info("PostCorrector fixed: %s", ", ".join(_changed_words(text, corrected)))
        return corrected
