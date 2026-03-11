"""Tests for voiceio.wordfreq — word frequency lookup."""
from __future__ import annotations

from voiceio.wordfreq import extract_words, is_common, is_known, rank


def test_is_common_basic():
    """Top words should be common."""
    assert is_common("the") is True
    assert is_common("and") is True
    assert is_common("is") is True


def test_is_common_case_insensitive():
    assert is_common("The") is True
    assert is_common("THE") is True


def test_is_common_high_threshold():
    # "the" is the most common word, should pass any reasonable threshold
    assert is_common("the", threshold=7.0) is True


def test_is_common_uncommon_word():
    # A gibberish word should not be common
    assert is_common("xyzzyplugh") is False


def test_rank_top_words():
    r = rank("the")
    assert r is not None
    assert r < 10  # "the" should be near rank 0


def test_rank_unknown():
    assert rank("xyzzyplugh") is None


def test_is_known_common():
    assert is_known("the") is True
    assert is_known("computer") is True


def test_is_known_valid_english():
    """Words that are real English but outside top 10K should still be known."""
    assert is_known("simplify") is True
    assert is_known("brainstorm") is True
    assert is_known("credentials") is True
    assert is_known("hedgehog") is True
    assert is_known("cognizant") is True


def test_is_known_gibberish():
    assert is_known("xyzzyplugh") is False
    assert is_known("olamma") is False
    assert is_known("gteu") is False


def test_is_known_contractions():
    assert is_known("don't") is True
    assert is_known("I've") is True
    assert is_known("let's") is True


def test_extract_words():
    words = extract_words("Hello, world! How's it going?")
    assert "hello" in words
    assert "world" in words
    assert "how's" in words
    assert "going" in words


def test_extract_words_empty():
    assert extract_words("") == []
    assert extract_words("123 456") == []


def test_is_known_case_insensitive():
    assert is_known("Computer") is True
    assert is_known("COMPUTER") is True
