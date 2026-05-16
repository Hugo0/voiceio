"""Tests for CLI helpers in voiceio.cli."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from voiceio.autocorrect import SuspiciousWord
from voiceio.cli import _offer_cluster_apply, _save_api_key


@pytest.fixture
def patch_config_path(monkeypatch, tmp_path):
    """Redirect voiceio.config.CONFIG_PATH to a temp file."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("voiceio.config.CONFIG_PATH", config_path)
    return config_path


def test_save_api_key_empty_config_openrouter(patch_config_path):
    """Saving to a fresh config writes a new [autocorrect] block."""
    cfg = MagicMock()
    _save_api_key(cfg, "sk-or-v1-abc123")

    content = patch_config_path.read_text(encoding="utf-8")
    assert "[autocorrect]" in content
    assert 'api_key = "sk-or-v1-abc123"' in content
    assert 'base_url = "https://openrouter.ai/api/v1"' in content
    assert "moonshotai/kimi-k2-0905" in content


def test_save_api_key_detects_anthropic(patch_config_path):
    """sk-ant- prefix routes to Anthropic provider."""
    _save_api_key(MagicMock(), "sk-ant-api03-xyz")

    content = patch_config_path.read_text(encoding="utf-8")
    assert 'api_key = "sk-ant-api03-xyz"' in content
    assert "api.anthropic.com" in content


def test_save_api_key_detects_openai(patch_config_path):
    """sk-proj- prefix routes to OpenAI provider."""
    _save_api_key(MagicMock(), "sk-proj-foo")

    content = patch_config_path.read_text(encoding="utf-8")
    assert 'api_key = "sk-proj-foo"' in content
    assert "api.openai.com" in content


def test_save_api_key_replaces_existing_in_autocorrect(patch_config_path):
    """Existing api_key under [autocorrect] is overwritten, not duplicated."""
    patch_config_path.write_text(
        '[autocorrect]\n'
        'api_key = "old-key"\n'
        'base_url = "https://example.com/v1"\n'
        'model = "old-model"\n',
        encoding="utf-8",
    )
    _save_api_key(MagicMock(), "sk-or-v1-new")

    content = patch_config_path.read_text(encoding="utf-8")
    assert content.count("api_key") == 1
    assert 'api_key = "sk-or-v1-new"' in content
    assert "old-key" not in content
    assert "old-model" not in content


def test_save_api_key_replaces_commented_field(patch_config_path):
    """A commented-out api_key line gets replaced."""
    patch_config_path.write_text(
        '[autocorrect]\n'
        '# api_key = ""\n'
        '# base_url = ""\n'
        '# model = ""\n',
        encoding="utf-8",
    )
    _save_api_key(MagicMock(), "sk-or-v1-new")

    content = patch_config_path.read_text(encoding="utf-8")
    assert 'api_key = "sk-or-v1-new"' in content
    assert "# api_key" not in content


def test_save_api_key_preserves_other_sections(patch_config_path):
    """Sections other than [autocorrect] are untouched."""
    patch_config_path.write_text(
        '[model]\n'
        'language = "en"\n'
        '\n'
        '[autocorrect]\n'
        'api_key = "old"\n'
        '\n'
        '[tts]\n'
        'engine = "piper"\n',
        encoding="utf-8",
    )
    _save_api_key(MagicMock(), "sk-or-v1-new")

    content = patch_config_path.read_text(encoding="utf-8")
    assert '[model]' in content
    assert 'language = "en"' in content
    assert '[tts]' in content
    assert 'engine = "piper"' in content
    assert 'api_key = "sk-or-v1-new"' in content


def test_save_api_key_appends_section_when_missing(patch_config_path):
    """If [autocorrect] doesn't exist yet, it's appended."""
    patch_config_path.write_text(
        '[model]\n'
        'language = "en"\n',
        encoding="utf-8",
    )
    _save_api_key(MagicMock(), "sk-or-v1-fresh")

    content = patch_config_path.read_text(encoding="utf-8")
    assert '[model]' in content
    assert '[autocorrect]' in content
    assert 'api_key = "sk-or-v1-fresh"' in content


def test_save_api_key_creates_parent_dir(monkeypatch, tmp_path):
    """Parent directory is created if missing."""
    nested = tmp_path / "deep" / "nested" / "config.toml"
    monkeypatch.setattr("voiceio.config.CONFIG_PATH", nested)

    _save_api_key(MagicMock(), "sk-or-v1-abc")

    assert nested.exists()
    assert 'api_key = "sk-or-v1-abc"' in nested.read_text(encoding="utf-8")


# ── _offer_cluster_apply ──────────────────────────────────────────────────


def _rl_prompt(s):  # passthrough — strip ANSI is done by readline IRL
    return s


def test_cluster_apply_no_close_variants_returns_zero(monkeypatch):
    """No remaining items within Levenshtein 2 → no prompt, no extra fixes."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnit", "right": ""},
        {"wrong": "completely_different", "right": ""},
    ]
    sw_by_word = {
        "pnit": SuspiciousWord(word="pnit", count=6),
        "completely_different": SuspiciousWord(word="completely_different", count=1),
    }
    # Should not call input() at all
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: pytest.fail("prompted"))
    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=1,
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    assert n == 0
    assert cd.add.call_count == 0
    assert len(to_review) == 2


def test_cluster_apply_yes_applies_to_all_close_variants(monkeypatch):
    """Y reply applies the same correction to every Levenshtein-≤2 remaining item."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnit", "right": ""},      # current (already corrected)
        {"wrong": "pnat", "right": "PNET"},  # close — should be batch-fixed
        {"wrong": "pnut", "right": "PNET"},  # close — should be batch-fixed
        {"wrong": "yaml", "right": ""},      # not close — left alone
        {"wrong": "pinat", "right": "PNET"}, # close (distance 2) — batch-fixed
    ]
    sw_by_word = {
        w["wrong"]: SuspiciousWord(word=w["wrong"], count=2) for w in to_review
    }
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "y")

    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=1,
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    assert n == 3
    added = {call.args for call in cd.add.call_args_list}
    assert added == {("pnat", "peanut"), ("pnut", "peanut"), ("pinat", "peanut")}
    # yaml stays in to_review; the three variants are removed
    assert [it["wrong"] for it in to_review] == ["pnit", "yaml"]


def test_cluster_apply_no_reply_skips(monkeypatch):
    """Replying 'n' applies nothing and leaves to_review intact."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnit", "right": ""},
        {"wrong": "pnat", "right": ""},
        {"wrong": "pnut", "right": ""},
    ]
    sw_by_word = {
        w["wrong"]: SuspiciousWord(word=w["wrong"], count=2) for w in to_review
    }
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=1,
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    assert n == 0
    assert cd.add.call_count == 0
    assert len(to_review) == 3


def test_cluster_apply_default_yes_on_empty_input(monkeypatch):
    """Empty input (just Enter) is treated as Yes."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnit", "right": ""},
        {"wrong": "pnat", "right": ""},
    ]
    sw_by_word = {w["wrong"]: SuspiciousWord(word=w["wrong"], count=2) for w in to_review}
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "")
    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=1,
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    assert n == 1
    cd.add.assert_called_once_with("pnat", "peanut")


def test_cluster_apply_skips_already_reviewed_indices(monkeypatch):
    """Items at indices < current_i (already reviewed) are not touched."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnut", "right": ""},   # already reviewed (index 0)
        {"wrong": "pnit", "right": ""},   # current
        {"wrong": "pnat", "right": ""},   # remaining — eligible
    ]
    sw_by_word = {w["wrong"]: SuspiciousWord(word=w["wrong"], count=2) for w in to_review}
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "y")
    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=2,  # skip past current word too
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    # Only pnat (at index 2) is eligible — pnut is at index 0 (< current_i).
    assert n == 1
    cd.add.assert_called_once_with("pnat", "peanut")


def test_cluster_apply_eof_aborts_safely(monkeypatch):
    """Ctrl-D / EOF on the prompt does not crash and applies nothing."""
    cd = MagicMock()
    to_review = [
        {"wrong": "pnit", "right": ""},
        {"wrong": "pnat", "right": ""},
    ]
    sw_by_word = {w["wrong"]: SuspiciousWord(word=w["wrong"], count=2) for w in to_review}

    def raise_eof(*_a, **_k):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    n = _offer_cluster_apply(
        cd, "pnit", "peanut", to_review, current_i=1,
        sw_by_word=sw_by_word, rl_prompt=_rl_prompt,
    )
    assert n == 0
    assert cd.add.call_count == 0
