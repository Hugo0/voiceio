"""Tests for voiceio.autocorrect — frequency analysis + LLM review."""
from __future__ import annotations

import json
from unittest.mock import patch

from voiceio.autocorrect import (
    _REVIEW_BATCH_SIZE,
    ReviewResult,
    SuspiciousWord,
    _find_similar_common,
    _levenshtein,
    _parse_review_response,
    cluster_variants,
    find_suspicious_words,
    rank_review_items,
    rank_review_score,
    review_suspicious,
)
from voiceio.config import AutocorrectConfig, Config


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


def test_parse_ask_user_keeps_empty_right_with_reason():
    """ask_user items must survive even when LLM has no clean correction."""
    response = json.dumps({
        "auto_fix": [],
        "ask_user": [
            {"wrong": "tridle", "right": "", "reason": "unclear"},
            {"wrong": "kaiki", "right": "kaiki", "reason": "uncertain"},
        ],
        "vocabulary": [],
    })
    result = _parse_review_response(response)
    assert len(result.ask_user) == 2
    assert result.ask_user[0]["wrong"] == "tridle"
    assert result.ask_user[0]["reason"] == "unclear"
    # right == wrong gets normalized to empty string (no useful suggestion)
    assert result.ask_user[1]["wrong"] == "kaiki"
    assert result.ask_user[1]["right"] == ""
    assert result.ask_user[1]["reason"] == "uncertain"


def test_parse_ask_user_drops_empty_wrong():
    """ask_user items with no `wrong` are still useless and get dropped."""
    response = json.dumps({
        "auto_fix": [],
        "ask_user": [{"wrong": "", "right": "something", "reason": "noise"}],
        "vocabulary": [],
    })
    result = _parse_review_response(response)
    assert result.ask_user == []


def test_parse_auto_fix_still_strict():
    """auto_fix bucket still requires `right != wrong` — case-only change dropped."""
    response = json.dumps({
        "auto_fix": [
            {"wrong": "hello", "right": "Hello"},        # case-only → drop
            {"wrong": "olamma", "right": "Ollama"},       # legit → keep
            {"wrong": "foo", "right": "", "reason": "?"}, # missing right → drop
        ],
        "ask_user": [],
        "vocabulary": [],
    })
    result = _parse_review_response(response)
    assert len(result.auto_fix) == 1
    assert result.auto_fix[0]["wrong"] == "olamma"


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


@patch("voiceio.llm_api.chat")
def test_review_batches_large_input(mock_chat):
    """Large suspicious lists are split across multiple chat() calls."""
    n = _REVIEW_BATCH_SIZE * 3 + 7  # forces 4 batches
    words = [SuspiciousWord(word=f"word{i}", count=1) for i in range(n)]
    # Each call returns a unique vocab entry so we can verify all batches ran.
    call_counter = {"i": 0}

    def fake_chat(*_a, **_kw):
        call_counter["i"] += 1
        return json.dumps({
            "auto_fix": [], "ask_user": [],
            "vocabulary": [f"batch{call_counter['i']}"],
        })

    mock_chat.side_effect = fake_chat
    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))
    result = review_suspicious(cfg, words)

    assert mock_chat.call_count == 4
    # Batches run in parallel — order is non-deterministic, so compare as set.
    assert set(result.vocabulary) == {"batch1", "batch2", "batch3", "batch4"}


@patch("voiceio.llm_api.chat")
def test_review_progress_callback(mock_chat):
    """on_progress is invoked once per batch and tops out at the total."""
    # Vocab keeps the result non-empty so review_suspicious doesn't fall back.
    mock_chat.return_value = json.dumps(
        {"auto_fix": [], "ask_user": [], "vocabulary": ["x"]},
    )
    n = _REVIEW_BATCH_SIZE * 2 + 5
    words = [SuspiciousWord(word=f"w{i}", count=1) for i in range(n)]
    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))

    progress: list[tuple[int, int]] = []
    review_suspicious(cfg, words, on_progress=lambda d, t: progress.append((d, t)))

    # Three batches → three progress calls. Order may vary (parallel),
    # but the final call must report `total = n` and totals must be
    # monotonically non-decreasing.
    assert len(progress) == 3
    assert all(t == n for _, t in progress)
    assert progress[-1][0] == n
    assert all(progress[i][0] <= progress[i + 1][0] for i in range(len(progress) - 1))


@patch("voiceio.autocorrect._REVIEW_OVERALL_TIMEOUT_PER_BATCH", 0.2)
@patch("voiceio.autocorrect._REVIEW_MAX_WORKERS", 2)
@patch("voiceio.llm_api.chat")
def test_review_overall_timeout_returns_partial(mock_chat):
    """A hung batch beyond the overall deadline doesn't stall the whole review."""
    import time as _t

    good_response = json.dumps({
        "auto_fix": [], "ask_user": [],
        "vocabulary": ["fast"],
    })
    call_count = {"i": 0}

    def slow_or_fast(*_a, **_kw):
        call_count["i"] += 1
        # First call returns immediately; second one hangs past the overall budget.
        if call_count["i"] == 1:
            return good_response
        _t.sleep(5.0)
        return good_response

    mock_chat.side_effect = slow_or_fast
    n = _REVIEW_BATCH_SIZE * 2  # forces 2 batches
    words = [SuspiciousWord(word=f"w{i}", count=1) for i in range(n)]
    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))

    started = _t.monotonic()
    result = review_suspicious(cfg, words)
    elapsed = _t.monotonic() - started

    # Fast batch landed; slow batch was abandoned.
    assert result.vocabulary == ["fast"]
    # Overall budget is rounds(1) * 0.2 * 2 = 0.4s — must terminate well below 5s.
    assert elapsed < 2.0


@patch("voiceio.llm_api.chat")
def test_review_partial_batch_failure_keeps_others(mock_chat):
    """One failed batch doesn't lose results from successful batches."""
    good = json.dumps({
        "auto_fix": [{"wrong": "olamma", "right": "Ollama"}],
        "ask_user": [], "vocabulary": [],
    })
    mock_chat.side_effect = [good, None, good]
    n = _REVIEW_BATCH_SIZE * 2 + 1
    words = [SuspiciousWord(word=f"w{i}", count=1) for i in range(n)]
    cfg = Config(autocorrect=AutocorrectConfig(api_key="test-key"))

    result = review_suspicious(cfg, words)
    # Two successful batches × one auto_fix each
    assert len(result.auto_fix) == 2


# ── ranking tests ─────────────────────────────────────────────────────────


def test_rank_score_llm_suggestion_dominates():
    """An LLM-suggested correction outranks anything without one."""
    sw = SuspiciousWord(word="foo", count=50, similar_common=["food"])
    item_with = {"wrong": "bar", "right": "barn"}
    item_without = {"wrong": "foo", "right": ""}
    assert rank_review_score(item_with, None) > rank_review_score(item_without, sw)


def test_rank_score_similar_common_boosts():
    """Words near a real dictionary word score higher than isolated ones."""
    sw_near = SuspiciousWord(word="wordal", count=5, similar_common=["wordle", "word"])
    sw_alone = SuspiciousWord(word="wordal", count=5, similar_common=[])
    item = {"wrong": "wordal", "right": ""}
    assert rank_review_score(item, sw_near) > rank_review_score(item, sw_alone)


def test_rank_score_short_acronym_demoted():
    """Short all-lowercase ASCII words with no similar common get pushed down."""
    sw_acronym = SuspiciousWord(word="yaml", count=10, similar_common=[])
    sw_typo = SuspiciousWord(word="wordal", count=10, similar_common=["wordle"])
    item_a = {"wrong": "yaml", "right": ""}
    item_t = {"wrong": "wordal", "right": ""}
    assert rank_review_score(item_t, sw_typo) > rank_review_score(item_a, sw_acronym)


def test_rank_review_items_orders_correctly():
    """rank_review_items: LLM-suggested first, acronyms last."""
    sw_by_word = {
        "yaml":   SuspiciousWord(word="yaml", count=10, similar_common=[]),
        "wordal": SuspiciousWord(word="wordal", count=8, similar_common=["wordle"]),
        "ctas":   SuspiciousWord(word="ctas", count=14, similar_common=[]),
        "tarnsl": SuspiciousWord(word="tarnsl", count=2, similar_common=["transl", "trans"]),
    }
    items = [
        {"wrong": "yaml",   "right": ""},
        {"wrong": "wordal", "right": ""},
        {"wrong": "ctas",   "right": ""},
        {"wrong": "tarnsl", "right": "transl"},  # has LLM suggestion
    ]
    ranked = rank_review_items(items, sw_by_word)
    order = [it["wrong"] for it in ranked]
    # LLM-suggested first; then word with similar_common; acronyms last.
    assert order[0] == "tarnsl"
    assert order[1] == "wordal"
    assert set(order[2:]) == {"yaml", "ctas"}


def test_rank_review_items_handles_missing_metadata():
    """Items without a SuspiciousWord entry still rank (lower)."""
    items = [
        {"wrong": "foo", "right": ""},
        {"wrong": "bar", "right": "baz"},
    ]
    ranked = rank_review_items(items, {})
    assert ranked[0]["wrong"] == "bar"
