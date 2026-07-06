"""Tests for voiceio.postcorrect — constrained LLM post-correction."""
from __future__ import annotations

from unittest.mock import patch

from voiceio.config import Config, PostCorrectConfig
from voiceio.postcorrect import PostCorrector


def _cfg(*, enabled=True, api_key="test-key", min_words=4, model="") -> Config:
    cfg = Config()
    cfg.postcorrect = PostCorrectConfig(
        enabled=enabled, min_words=min_words, model=model,
    )
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
