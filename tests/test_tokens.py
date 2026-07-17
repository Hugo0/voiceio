"""Tests for exact Whisper token counting and the shared 448-token budget.

Every budget here used to be a character count with a guessed chars/token ratio.
The guess was wrong in the direction that mattered: proper nouns tokenize at
~2.6 chars/token vs prose's ~4.0, so the old 600-char hotwords cap was really
238 tokens — past faster-whisper's 223 cap, which then silently dropped terms
and starved the output budget.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from voiceio.tokens import (
    HOTWORDS_TOKEN_CAP,
    MAX_SEQUENCE_TOKENS,
    OUTPUT_RESERVE_TOKENS,
    PROMPT_TOKEN_BUDGET,
    _tokenizer,
    count_tokens,
    truncate_to_tokens,
)

# Most tests here are self-consistent — they count and assert with the same
# function, so the estimate fallback is fine. A few assert properties of the
# REAL tokenizer and are meaningless against a fixed chars/token constant; those
# need a cached model, which CI runners don't have.
needs_tokenizer = pytest.mark.skipif(
    _tokenizer("small") is None,
    reason="no cached whisper tokenizer (CI runners have no model cache)",
)


class TestCountTokens:
    def test_empty(self):
        assert count_tokens("", "small") == 0

    def test_counts_real_tokens(self):
        n = count_tokens("Grafana, Hetzner, OpenRouter", "small")
        assert 5 < n < 30

    @needs_tokenizer
    def test_proper_nouns_cost_more_than_prose_per_char(self):
        """The measured fact the whole redesign rests on.

        Only meaningful against the real tokenizer: the fallback is a fixed
        chars/token constant, which makes both ratios equal by construction.
        """
        nouns = "Kalshi, Metaculus, HyperNEAT, Kubernetes, Hetzner, Grafana"
        prose = "the quick brown fox jumps over the lazy dog and then it runs"
        nouns_ratio = len(nouns) / count_tokens(nouns, "small")
        prose_ratio = len(prose) / count_tokens(prose, "small")
        assert nouns_ratio < prose_ratio

    def test_falls_back_to_estimate_without_tokenizer(self):
        """No tokenizer must degrade to an estimate, never break dictation."""
        import voiceio.tokens as t
        with patch.dict(t._cache, {"bogus-model": None}, clear=False):
            n = count_tokens("hello world", "bogus-model")
        assert n > 0

    def test_survives_a_broken_tokenizer(self):
        import voiceio.tokens as t

        class _Boom:
            def encode(self, _):
                raise RuntimeError("corrupt")

        with patch.dict(t._cache, {"broken": _Boom()}, clear=False):
            assert count_tokens("hello world", "broken") > 0


class TestTruncateToTokens:
    def test_noop_when_within_budget(self):
        assert truncate_to_tokens("short text", 100, "small") == "short text"

    def test_trims_to_budget(self):
        text = "word " * 500
        out = truncate_to_tokens(text, 50, "small")
        assert count_tokens(out, "small") <= 50
        assert out  # not emptied

    def test_keeps_the_end(self):
        """Prompts condition the audio that follows, so the OLDEST context goes."""
        text = "alpha beta gamma delta " * 50 + "FINALWORD"
        out = truncate_to_tokens(text, 20, "small")
        assert "FINALWORD" in out

    def test_zero_budget_is_empty(self):
        assert truncate_to_tokens("anything", 0, "small") == ""


class TestSharedBudget:
    def test_budgets_leave_room_to_transcribe(self):
        """hotwords + prompt + output must fit 448, with output usable.

        Before this, the live config spent 223 (hotwords) + 171 (prompt) and
        left ~50 tokens — about 37 words — to transcribe a freeze chunk holding
        45-60. Worst case it went negative and faster-whisper raised
        "The maximum decoding length must be > 0".
        """
        from voiceio.app import _HOTWORDS_TOKEN_BUDGET

        total = _HOTWORDS_TOKEN_BUDGET + PROMPT_TOKEN_BUDGET + OUTPUT_RESERVE_TOKENS
        assert total + 4 <= MAX_SEQUENCE_TOKENS
        assert _HOTWORDS_TOKEN_BUDGET <= HOTWORDS_TOKEN_CAP
