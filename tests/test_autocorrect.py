"""Tests for voiceio.autocorrect — offline suspicious-word finder and rule gate."""
from __future__ import annotations


from voiceio.autocorrect import (
    _find_similar_common,
    _levenshtein,
    find_suspicious_words,
)


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


def test_find_suspicious_prefers_raw_field():
    """History v2 entries: mine the pre-correction `raw`, not corrected `text`."""
    entries = [{
        "text": "the configuration needs fixing",   # already corrected
        "raw": "the configuraton needs fixing",       # what Whisper heard
    }]
    words = {s.word for s in find_suspicious_words(entries)}
    assert "configuraton" in words
    assert "configuration" not in words


def test_find_suspicious_v1_entry_without_raw():
    """v1 entries (only `text`) still work."""
    entries = [{"text": "the configuraton needs fixing"}]
    words = {s.word for s in find_suspicious_words(entries)}
    assert "configuraton" in words


def test_find_suspicious_skips_dismissed():
    entries = [{"text": "the configuraton needs fixing"}]
    result = find_suspicious_words(entries, dismissed={"configuraton"})
    assert "configuraton" not in {s.word for s in result}


# ── Safety gate tests ─────────────────────────────────────────────────────


def test_gate_blocks_real_word_as_wrong():
    """A common real word must never become a correction source."""
    from voiceio.autocorrect import gate_correction
    reason = gate_correction("their", "there")
    assert reason is not None


def test_gate_blocks_junk_target():
    """A non-word target that isn't in vocabulary is rejected."""
    from voiceio.autocorrect import gate_correction
    reason = gate_correction("manteka", "wordall")
    assert reason is not None


def test_gate_passes_nonword_to_common():
    """Non-word wrong + common right passes cleanly."""
    from voiceio.autocorrect import gate_correction
    assert gate_correction("configuraton", "configuration") is None


def test_gate_passes_target_in_vocabulary():
    """A non-common target that's in the user's vocabulary is allowed."""
    from voiceio.autocorrect import gate_correction
    assert gate_correction("olamma", "Ollama", vocabulary={"Ollama"}) is None
    assert gate_correction("olamma", "Ollama") is not None


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



def test_find_similar_common_basic():
    common = {"configuration": 500, "cat": 100, "the": 1}
    result = _find_similar_common("configuraton", common, max_distance=2)
    assert "configuration" in result


def test_find_similar_common_no_match():
    common = {"cat": 100, "dog": 200}
    result = _find_similar_common("zzzzzzzzz", common, max_distance=2)
    assert result == []


# ── _parse_review_response tests ──────────────────────────────────────────
