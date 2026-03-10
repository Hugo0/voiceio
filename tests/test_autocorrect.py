"""Tests for voiceio.autocorrect — frequency analysis + LLM review."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from voiceio.autocorrect import (
    ReviewResult,
    SuspiciousWord,
    _find_similar_common,
    _levenshtein,
    _parse_review_response,
    cluster_variants,
    find_suspicious_words,
    review_suspicious,
)
from voiceio.config import AutocorrectConfig, Config, LLMConfig


# ── Levenshtein distance tests ────────────────────────────────────────────


def test_levenshtein_identical():
    assert _levenshtein("hello", "hello") == 0


def test_levenshtein_one_insert():
    assert _levenshtein("cat", "cats") == 1


def test_levenshtein_one_replace():
    assert _levenshtein("cat", "car") == 1


def test_levenshtein_two_edits():
    assert _levenshtein("kitten", "sitting") == 3


def test_levenshtein_empty():
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "") == 3


# ── find_suspicious_words tests ───────────────────────────────────────────


def test_find_suspicious_common_words_excluded():
    """Common words like 'the', 'and', 'for' should never be flagged."""
    entries = [{"text": "the quick brown fox jumps over the lazy dog"}]
    result = find_suspicious_words(entries)
    words = {s.word for s in result}
    assert "the" not in words
    assert "over" not in words
    assert "quick" not in words


def test_find_suspicious_detects_uncommon():
    """Words not in the common list should be flagged."""
    entries = [{"text": "the configuraton needs fixing"}]
    result = find_suspicious_words(entries)
    words = {s.word for s in result}
    assert "configuraton" in words


def test_find_suspicious_skips_existing_corrections():
    entries = [{"text": "the configuraton needs fixing"}]
    result = find_suspicious_words(entries, existing_corrections={"configuraton"})
    words = {s.word for s in result}
    assert "configuraton" not in words


def test_find_suspicious_skips_vocabulary():
    entries = [{"text": "the kubernetes cluster is running"}]
    result = find_suspicious_words(entries, vocabulary={"kubernetes"})
    words = {s.word for s in result}
    assert "kubernetes" not in words


def test_find_suspicious_skips_short_words():
    entries = [{"text": "xyz abc qrs"}]
    result = find_suspicious_words(entries, min_word_length=4)
    words = {s.word for s in result}
    assert "xyz" not in words
    assert "abc" not in words


def test_find_suspicious_counts():
    entries = [
        {"text": "the configuraton file"},
        {"text": "update configuraton please"},
        {"text": "check configuraton again"},
    ]
    result = find_suspicious_words(entries)
    for sw in result:
        if sw.word == "configuraton":
            assert sw.count == 3
            assert len(sw.contexts) == 3
            break


def test_find_suspicious_similar_common():
    """Misspellings should find their correct common-word neighbors."""
    entries = [{"text": "the configuraton needs fixing"}]
    result = find_suspicious_words(entries)
    for sw in result:
        if sw.word == "configuraton":
            assert any("configur" in w for w in sw.similar_common)
            break


# ── cluster_variants tests ────────────────────────────────────────────────


def test_cluster_identical_words():
    words = [
        SuspiciousWord(word="configuraton", count=3),
        SuspiciousWord(word="configuation", count=1),
    ]
    clusters = cluster_variants(words)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_distant_words():
    words = [
        SuspiciousWord(word="configuraton", count=3),
        SuspiciousWord(word="kubernetes", count=1),
    ]
    clusters = cluster_variants(words)
    assert len(clusters) == 2


def test_cluster_empty():
    assert cluster_variants([]) == []


# ── _find_similar_common tests ────────────────────────────────────────────


def test_find_similar_common_basic():
    common = {"configuration": 500, "cat": 100, "the": 1}
    result = _find_similar_common("configuraton", common, max_distance=2)
    assert "configuration" in result


def test_find_similar_common_no_match():
    common = {"cat": 100, "dog": 200}
    result = _find_similar_common("zzzzzzzzz", common, max_distance=2)
    assert result == []


# ── _parse_review_response tests ──────────────────────────────────────────


def test_parse_valid_3bucket():
    response = json.dumps({
        "auto_fix": [{"wrong": "olamma", "right": "Ollama"}],
        "ask_user": [{"wrong": "pinat", "right": "Peanut", "reason": "brand name"}],
        "vocabulary": ["grafana", "postgres"],
    })
    result = _parse_review_response(response)
    assert len(result.auto_fix) == 1
    assert result.auto_fix[0]["wrong"] == "olamma"
    assert len(result.ask_user) == 1
    assert result.ask_user[0]["reason"] == "brand name"
    assert result.vocabulary == ["grafana", "postgres"]


def test_parse_code_fenced():
    inner = json.dumps({
        "auto_fix": [{"wrong": "foo", "right": "bar"}],
        "ask_user": [],
        "vocabulary": [],
    })
    response = f"```json\n{inner}\n```"
    result = _parse_review_response(response)
    assert len(result.auto_fix) == 1


def test_parse_empty_object():
    result = _parse_review_response("{}")
    assert result.auto_fix == []
    assert result.ask_user == []
    assert result.vocabulary == []


def test_parse_garbage():
    result = _parse_review_response("no json here")
    assert result == ReviewResult()


def test_parse_filters_same_word():
    response = json.dumps({
        "auto_fix": [{"wrong": "hello", "right": "Hello"}],
        "ask_user": [],
        "vocabulary": [],
    })
    result = _parse_review_response(response)
    assert result.auto_fix == []  # case-only change filtered


def test_parse_embedded_json():
    inner = json.dumps({
        "auto_fix": [{"wrong": "foo", "right": "bar"}],
        "ask_user": [],
        "vocabulary": [],
    })
    response = f"Here are my findings:\n{inner}\nDone."
    result = _parse_review_response(response)
    assert len(result.auto_fix) == 1


def test_parse_flat_array_fallback():
    """Old-style flat array (Ollama) should go into ask_user bucket."""
    response = json.dumps([
        {"wrong": "foo", "right": "bar", "confidence": "high"},
    ])
    result = _parse_review_response(response)
    assert len(result.ask_user) == 1
    assert result.ask_user[0]["wrong"] == "foo"


def test_parse_partial_buckets():
    """Missing buckets should default to empty."""
    response = json.dumps({"auto_fix": [{"wrong": "a", "right": "b"}]})
    result = _parse_review_response(response)
    assert len(result.auto_fix) == 1
    assert result.ask_user == []
    assert result.vocabulary == []


# ── review_suspicious tests ───────────────────────────────────────────────


@patch("voiceio.llm_api.chat")
def test_review_with_api_key(mock_chat):
    response = json.dumps({
        "auto_fix": [{"wrong": "olamma", "right": "Ollama"}],
        "ask_user": [],
        "vocabulary": ["grafana"],
    })
    mock_chat.return_value = response

    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))
    words = [SuspiciousWord(word="olamma", count=3, contexts=["install olamma"])]
    result = review_suspicious(cfg, words)
    assert len(result.auto_fix) == 1
    assert result.vocabulary == ["grafana"]


def test_review_empty():
    cfg = Config()
    result = review_suspicious(cfg, [])
    assert result == ReviewResult()


@patch("voiceio.llm_api.chat")
def test_review_api_failure_falls_back(mock_chat):
    """If cloud API fails and no Ollama, returns empty result."""
    mock_chat.return_value = None
    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))
    words = [SuspiciousWord(word="foo", count=1)]
    result = review_suspicious(cfg, words)
    assert result == ReviewResult()
