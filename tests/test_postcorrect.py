"""Tests for voiceio.postcorrect — constrained LLM post-correction."""
from __future__ import annotations

from unittest.mock import patch

from voiceio.config import Config, PostCorrectConfig
from voiceio.postcorrect import PostCorrector


def _cfg(*, enabled=True, api_key="test-key", min_words=4, model="",
         remove_disfluencies=False) -> Config:
    cfg = Config()
    cfg.postcorrect = PostCorrectConfig(
        enabled=enabled, min_words=min_words, model=model,
    )
    cfg.output.remove_disfluencies = remove_disfluencies
    cfg.autocorrect.api_key = api_key
    return cfg


# ── availability / passthrough ───────────────────────────────────────────


def test_disabled_passthrough():
    pc = PostCorrector(_cfg(enabled=False))
    with patch("voiceio.llm_api.chat") as mock_chat:
        assert pc.correct("Chris chat is the tool") == "Chris chat is the tool"
        mock_chat.assert_not_called()


def test_no_api_key_passthrough():
    import os
    with patch.dict(os.environ, {}, clear=True):
        pc = PostCorrector(_cfg(api_key=""))
        with patch("voiceio.llm_api.chat") as mock_chat:
            assert pc.correct("some transcript text here") == "some transcript text here"
            mock_chat.assert_not_called()


def test_short_utterance_skipped():
    pc = PostCorrector(_cfg(min_words=4))
    with patch("voiceio.llm_api.chat") as mock_chat:
        assert pc.correct("hello there") == "hello there"
        mock_chat.assert_not_called()


# ── successful correction ────────────────────────────────────────────────


def test_successful_correction_applied():
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the tool"):
        assert pc.correct("Chris chat is the tool") == "Crisp chat is the tool"


def test_unchanged_response_returns_original():
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value="the text is fine here"):
        assert pc.correct("the text is fine here") == "the text is fine here"


# ── guards ───────────────────────────────────────────────────────────────


def test_edit_distance_guard_rejects_rewrite():
    """An over-eager rephrase (many words changed) must be rejected."""
    pc = PostCorrector(_cfg())
    original = "we deployed the crons on the server last night"
    rewrite = "the cron jobs were successfully rolled out yesterday evening"
    with patch("voiceio.llm_api.chat", return_value=rewrite):
        assert pc.correct(original) == original


def test_word_count_guard_rejects_length_change():
    """A response that adds/removes too many words is rejected."""
    pc = PostCorrector(_cfg())
    original = "run the crons now please"
    # Doubling the word count exceeds the 20% delta ceiling.
    padded = "run the crons now please and also do many other extra things"
    with patch("voiceio.llm_api.chat", return_value=padded):
        assert pc.correct(original) == original


# ── disfluency mode: delete-only, never change meaning ───────────────────


def test_disfluency_mode_allows_deletion():
    """Removing filler shrinks the text well past the 20% fix-mode ceiling —
    the disfluency guard must allow it (deletions only)."""
    pc = PostCorrector(_cfg(remove_disfluencies=True))
    original = "so um we should like ship the thing today"
    cleaned = "so we should ship the thing today"  # dropped 'um' and 'like'
    with patch("voiceio.llm_api.chat", return_value=cleaned):
        assert pc.correct(original) == cleaned


def test_disfluency_mode_rejects_insertion():
    """Adding any word could alter meaning — reject, keep the original."""
    pc = PostCorrector(_cfg(remove_disfluencies=True))
    original = "we should ship the thing today"
    added = "we should ship the thing today and also test everything first"
    with patch("voiceio.llm_api.chat", return_value=added):
        assert pc.correct(original) == original


def test_disfluency_mode_rejects_overdeletion():
    """Deleting more than the cap means real content was stripped — reject."""
    pc = PostCorrector(_cfg(remove_disfluencies=True))
    original = "please run the full test suite before you deploy tonight okay"
    gutted = "please run test deploy"  # >40% of words gone
    with patch("voiceio.llm_api.chat", return_value=gutted):
        assert pc.correct(original) == original


def test_disfluency_mode_rejects_rewording():
    """Many substitutions = a rewrite, not filler removal — reject."""
    pc = PostCorrector(_cfg(remove_disfluencies=True))
    original = "the server crashed at midnight during the backup"
    reworded = "the machine went down at noon while copying files"
    with patch("voiceio.llm_api.chat", return_value=reworded):
        assert pc.correct(original) == original


def test_disfluency_mode_allows_trailing_shrug_with_negation():
    """No mechanical negation guard — the LLM decides. Dropping a trailing
    'I don't know' shrug (plus other filler) is accepted; the model, not a
    regex, is trusted to avoid genuine polarity flips (see the prompt)."""
    pc = PostCorrector(_cfg(remove_disfluencies=True))
    original = "the build passed but I don't know we should still check it carefully today"
    cleaned = "the build passed we should still check it carefully today"
    with patch("voiceio.llm_api.chat", return_value=cleaned):
        assert pc.correct(original) == cleaned


def test_fix_mode_still_rejects_big_deletion():
    """With disfluency mode OFF, the original conservative guards stand: a big
    deletion is a length change and must be rejected."""
    pc = PostCorrector(_cfg(remove_disfluencies=False))
    original = "so um we should like ship the thing today"
    cleaned = "so we should ship the thing today"
    with patch("voiceio.llm_api.chat", return_value=cleaned):
        assert pc.correct(original) == original


def test_llm_error_returns_original():
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", side_effect=RuntimeError("boom")):
        assert pc.correct("run the crons on the box") == "run the crons on the box"


def test_empty_response_returns_original():
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value=None):
        assert pc.correct("run the crons on the box") == "run the crons on the box"


# ── response cleanup ─────────────────────────────────────────────────────


def test_strips_markdown_fence():
    pc = PostCorrector(_cfg())
    fenced = "```\nCrisp chat is the tool\n```"
    with patch("voiceio.llm_api.chat", return_value=fenced):
        assert pc.correct("Chris chat is the tool") == "Crisp chat is the tool"


def test_strips_surrounding_quotes():
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value='"Crisp chat is the tool"'):
        assert pc.correct("Chris chat is the tool") == "Crisp chat is the tool"


# ── context wiring ───────────────────────────────────────────────────────


def test_context_included_in_prompt():
    pc = PostCorrector(_cfg())
    pc.set_context(vocabulary="Crisp, crons", recent=["earlier text"], title="editor")
    captured = {}

    def fake_chat(cfg, system, user_msg, **kw):
        captured["user"] = user_msg
        return "Crisp chat is the tool"

    with patch("voiceio.llm_api.chat", side_effect=fake_chat):
        pc.correct("Chris chat is the tool")
    assert "Crisp, crons" in captured["user"]
    assert "earlier text" in captured["user"]
    assert "editor" in captured["user"]
    assert "Chris chat is the tool" in captured["user"]


def test_uses_postcorrect_model_override():
    pc = PostCorrector(_cfg(model="my/override-model"))
    captured = {}

    def fake_chat(cfg, system, user_msg, **kw):
        captured["model"] = cfg.model
        return "Crisp chat is the tool"

    with patch("voiceio.llm_api.chat", side_effect=fake_chat):
        pc.correct("Chris chat is the tool")
    assert captured["model"] == "my/override-model"


# ── pipeline ordering ────────────────────────────────────────────────────


def test_pipeline_runs_postcorrect_only_on_final():
    from voiceio.postprocess import apply_pipeline

    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the best tool"):
        # Non-final pass: postcorrect must NOT run.
        text, abort = apply_pipeline(
            "Chris chat is the best tool", postcorrect=pc, final=False,
        )
        assert text == "Chris chat is the best tool"
        assert not abort

        # Final pass: postcorrect runs.
        text, abort = apply_pipeline(
            "Chris chat is the best tool", postcorrect=pc, final=True,
        )
        assert text == "Crisp chat is the best tool"
        assert not abort


# ── wall-clock deadline ──────────────────────────────────────────────────


def test_deadline_exceeded_keeps_original():
    """A chat call slower than timeout_secs is abandoned at the deadline."""
    import time as _time
    cfg = _cfg()
    cfg.postcorrect.timeout_secs = 0.2

    def slow_chat(*a, **kw):
        _time.sleep(2.0)
        return "Crisp chat is the tool"

    pc = PostCorrector(cfg)
    with patch("voiceio.llm_api.chat", side_effect=slow_chat):
        t0 = _time.monotonic()
        assert pc.correct("Chris chat is the tool") == "Chris chat is the tool"
        elapsed = _time.monotonic() - t0
    assert elapsed < 1.0  # returned at the deadline, not after the full call
    assert pc.last_secs is not None


def test_within_deadline_applies_correction():
    cfg = _cfg()
    cfg.postcorrect.timeout_secs = 5.0
    pc = PostCorrector(cfg)
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the tool"):
        assert pc.correct("Chris chat is the tool") == "Crisp chat is the tool"


# ── before/after pair persistence ────────────────────────────────────────


def _read_pairs(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_applied_correction_recorded(tmp_path, monkeypatch):
    import voiceio.postcorrect as pcmod
    from voiceio import config as cfgmod
    pairs = tmp_path / "pairs.jsonl"
    monkeypatch.setattr(cfgmod, "POSTCORRECT_PAIRS_PATH", pairs)
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the tool"):
        pc.correct("Chris chat is the tool")
    entries = _read_pairs(pairs)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "applied"
    assert entries[0]["before"] == "Chris chat is the tool"
    assert entries[0]["after"] == "Crisp chat is the tool"
    assert entries[0]["secs"] is not None


def test_rejected_rewrite_recorded(tmp_path, monkeypatch):
    from voiceio import config as cfgmod
    pairs = tmp_path / "pairs.jsonl"
    monkeypatch.setattr(cfgmod, "POSTCORRECT_PAIRS_PATH", pairs)
    pc = PostCorrector(_cfg())
    with patch("voiceio.llm_api.chat", return_value="Completely different rewritten sentence with many other words entirely"):
        out = pc.correct("Chris chat is the tool")
    assert out == "Chris chat is the tool"
    entries = _read_pairs(pairs)
    assert len(entries) == 1
    assert entries[0]["outcome"].startswith("rejected")


def test_no_recording_when_disabled(tmp_path, monkeypatch):
    from voiceio import config as cfgmod
    pairs = tmp_path / "pairs.jsonl"
    monkeypatch.setattr(cfgmod, "POSTCORRECT_PAIRS_PATH", pairs)
    cfg = _cfg()
    cfg.data.capture_intermediates = False
    pc = PostCorrector(cfg)
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the tool"):
        pc.correct("Chris chat is the tool")
    assert not pairs.exists()


def test_hung_request_blocks_next_call_not_thread_pileup(tmp_path, monkeypatch):
    """A worker abandoned at the deadline makes the NEXT call skip the LLM."""
    import threading as _threading
    import time as _time
    from voiceio import config as cfgmod
    monkeypatch.setattr(cfgmod, "POSTCORRECT_PAIRS_PATH", tmp_path / "pairs.jsonl")
    cfg = _cfg()
    cfg.postcorrect.timeout_secs = 0.1
    pc = PostCorrector(cfg)
    release = _threading.Event()

    def hung_chat(*a, **kw):
        release.wait(5)
        return "x"

    with patch("voiceio.llm_api.chat", side_effect=hung_chat) as mock_chat:
        assert pc.correct("Chris chat is the tool") == "Chris chat is the tool"
        assert mock_chat.call_count == 1
        # Second call while the first worker is still hung: no new thread
        assert pc.correct("Chris chat is the tool") == "Chris chat is the tool"
        assert mock_chat.call_count == 1
    release.set()
    entries = _read_pairs(tmp_path / "pairs.jsonl")
    assert [e["outcome"] for e in entries] == ["timeout", "skipped_busy"]


def test_recorded_context_is_effective_context(tmp_path, monkeypatch):
    from voiceio import config as cfgmod
    monkeypatch.setattr(cfgmod, "POSTCORRECT_PAIRS_PATH", tmp_path / "pairs.jsonl")
    pc = PostCorrector(_cfg())
    pc.set_context(title="Stale Window")
    with patch("voiceio.llm_api.chat", return_value="Crisp chat is the tool"):
        pc.correct("Chris chat is the tool", context="Fresh Window")
    entries = _read_pairs(tmp_path / "pairs.jsonl")
    assert entries[0]["context"] == "Fresh Window"
