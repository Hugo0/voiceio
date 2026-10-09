"""Vocabulary biasing shared by every front end (desktop, server, eval).

Turns the user's vocabulary, rare correction targets and recent transcripts
into a per-request `Bias` sized to what the decoder can take, and feeds usage
back so the limited hotword budget follows what the user actually dictates.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from voiceio.core.decoders.base import Bias, BiasMode, Capabilities
from voiceio.tokens import HOTWORDS_TOKEN_BUDGET

if TYPE_CHECKING:
    from voiceio.corrections import CorrectionDict
    from voiceio.prompt import PromptBuilder
    from voiceio.vocabulary import VocabularyLoader

log = logging.getLogger(__name__)

# Used when a bounded decoder doesn't declare its own budget.
DEFAULT_HOTWORDS_TOKEN_BUDGET = HOTWORDS_TOKEN_BUDGET


class BiasProvider:
    def __init__(
        self,
        vocabulary: VocabularyLoader,
        corrections: CorrectionDict | None = None,
        prompts: PromptBuilder | None = None,
        language: str = "en",
    ):
        self._vocab = vocabulary
        self._corrections = corrections
        self._prompts = prompts
        self._language = language
        self._rare_targets: list[str] = []
        self._rare_targets_mtime: object = object()

    def bias_for(self, caps: Capabilities) -> Bias:
        """A fresh Bias for one recording/request. Never raises: biasing is a
        bonus and must not be able to block a dictation."""
        hotwords = None
        if caps.bias & (BiasMode.HOTWORDS | BiasMode.UNBOUNDED):
            try:
                hotwords = self._hotwords(caps) or None
            except Exception:  # noqa: BLE001
                log.debug("hotword selection failed", exc_info=True)
        prompt = None
        if caps.bias & BiasMode.PROMPT and self._prompts is not None:
            prompt = self._prompts.build()
        return Bias(hotwords=hotwords, prompt=prompt)

    def record_usage(self, text: str) -> None:
        """Count which vocabulary terms this utterance used (finalize path)."""
        if not text:
            return
        if self._prompts is not None:
            self._prompts.add_transcript(text)
        try:
            from voiceio import vocab_stats
            vocab_stats.update_from_text(text, self._vocab.get_all())
        except Exception:  # noqa: BLE001 — a counter must never break dictation
            log.debug("vocab usage update failed", exc_info=True)

    def _hotwords(self, caps: Capabilities) -> str:
        if caps.bias & BiasMode.UNBOUNDED:
            terms = self._vocab.get_all()
        else:
            budget = caps.max_bias_tokens or DEFAULT_HOTWORDS_TOKEN_BUDGET
            terms = self._vocab.get_selected(token_budget=budget)
        # Only RARE correction targets: common words add no bias and hundreds
        # of them blow Whisper's shared prompt+output budget.
        extra = [t for t in self._rare_correction_targets() if t not in terms]
        return ", ".join(terms + extra)

    def _rare_correction_targets(self) -> list[str]:
        """Cached against corrections.json's mtime (it rarely changes)."""
        if self._corrections is None:
            return []
        mtime = getattr(self._corrections, "_mtime", None)
        if self._rare_targets_mtime != mtime:
            from voiceio.wordfreq import is_common
            self._rare_targets = [
                t for t in self._corrections.vocabulary_terms()
                if not is_common(t, self._language)
            ]
            self._rare_targets_mtime = mtime
        return self._rare_targets
