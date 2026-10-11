"""Vocabulary biasing shared by every front end (desktop, server, eval).

Turns the user's vocabulary, rare correction targets and recent transcripts
into a per-request `Bias` sized to what the decoder can take, and feeds usage
back so the limited hotword budget follows what the user actually dictates.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from voiceio.core.decoders.base import Bias, BiasMode, Capabilities
from voiceio.tokens import HOTWORDS_TOKEN_BUDGET, count_tokens, fit_prefix

if TYPE_CHECKING:
    from voiceio.corrections import CorrectionDict
    from voiceio.prompt import PromptBuilder
    from voiceio.vocabulary import VocabularyLoader

log = logging.getLogger(__name__)

# Used when a bounded decoder doesn't declare its own budget.
DEFAULT_HOTWORDS_TOKEN_BUDGET = HOTWORDS_TOKEN_BUDGET

# Most of the bounded budget on-screen terms may take (context_terms.py). One
# proper noun costs 4-6 tokens, so this is the 4-5 terms that matter for one
# dictation, and it only displaces the static vocabulary's lowest-ranked ~20%
# (~5-7 of ~34 terms) — and only while the screen actually offers terms. The
# vocabulary + screen part stays inside the same budget, so the hotwords (and
# decode cost) barely grow: only a displaced term that is also a correction
# target comes back in the tail.
SCREEN_TOKEN_SHARE = 24


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

    @property
    def vocabulary(self) -> VocabularyLoader:
        return self._vocab

    def recent(self, n: int = 3) -> list[str]:
        """The last `n` committed utterances (oldest first)."""
        return self._prompts.recent(n) if self._prompts is not None else []

    def screen_terms(self, texts: Sequence[str], caps: Capabilities) -> list[str]:
        """Terms worth biasing from on-screen text (window title, selection),
        minus what the hotwords already carry. Never raises."""
        if not any(texts) or not caps.bias & (BiasMode.HOTWORDS | BiasMode.UNBOUNDED):
            return []
        try:
            from voiceio.context_terms import extract_terms
            return extract_terms(
                texts, language=self._language,
                exclude=self._static_terms(caps) + self._rare_correction_targets(),
                prefer=self._vocab.get_all(),
            )
        except Exception:  # noqa: BLE001
            log.debug("screen term extraction failed", exc_info=True)
            return []

    def bias_for(self, caps: Capabilities, screen_terms: Sequence[str] = ()) -> Bias:
        """A fresh Bias for one recording/request. Never raises: biasing is a
        bonus and must not be able to block a dictation."""
        hotwords = None
        if caps.bias & (BiasMode.HOTWORDS | BiasMode.UNBOUNDED):
            try:
                hotwords = self._hotwords(caps, screen_terms) or None
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

    def _budget(self, caps: Capabilities) -> int:
        return caps.max_bias_tokens or DEFAULT_HOTWORDS_TOKEN_BUDGET

    def _static_terms(self, caps: Capabilities) -> list[str]:
        if caps.bias & BiasMode.UNBOUNDED:
            return self._vocab.get_all()
        return self._vocab.get_selected(token_budget=self._budget(caps))

    def _hotwords(self, caps: Capabilities, screen_terms: Sequence[str] = ()) -> str:
        terms = self._static_terms(caps)
        taken = {t.lower() for t in terms}
        screen = [t for t in screen_terms if t.lower() not in taken]
        if screen and not caps.bias & BiasMode.UNBOUNDED:
            # Screen terms take their share off the TAIL of the ranked
            # vocabulary, so the total stays inside the budget.
            model = self._vocab.model_name
            screen = fit_prefix(screen, SCREEN_TOKEN_SHARE, model)
            if screen:
                used = count_tokens(", ".join(screen), model)
                terms = fit_prefix(terms, self._budget(caps) - used, model)
        # Only RARE correction targets: common words add no bias and hundreds
        # of them blow Whisper's shared prompt+output budget.
        extra = [t for t in self._rare_correction_targets()
                 if t not in terms and t not in screen]
        return ", ".join(terms + screen + extra)

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

