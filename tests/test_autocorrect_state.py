"""Tests for the autocorrect scan cursor + dismissed-set state."""
from __future__ import annotations

from voiceio.autocorrect_state import (
    AutocorrectState, load_state, save_state,
)


def test_load_missing_returns_defaults(tmp_path):
    state = load_state(tmp_path / "nope.json")
    assert state.last_scan_ts == 0.0
    assert state.dismissed == set()


def test_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    state = AutocorrectState(last_scan_ts=123.5)
    state.dismiss("Manteka")
    state.dismiss("wordall")
    save_state(state, p)

    loaded = load_state(p)
    assert loaded.last_scan_ts == 123.5
    assert loaded.is_dismissed("manteka")   # case-insensitive
    assert loaded.is_dismissed("WORDALL")
    assert not loaded.is_dismissed("kubernetes")


def test_malformed_file_tolerated(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ not valid json")
    state = load_state(p)
    assert state.last_scan_ts == 0.0
    assert state.dismissed == set()


def test_dismiss_empty_ignored():
    state = AutocorrectState()
    state.dismiss("")
    assert state.dismissed == set()


def test_default_path_isolated_by_conftest(tmp_path, monkeypatch):
    """save/load with no explicit path uses the (isolated) module STATE_PATH."""
    import voiceio.autocorrect_state as mod
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "s.json")
    s = AutocorrectState(last_scan_ts=9.0)
    s.dismiss("foo")
    save_state(s)
    assert load_state().is_dismissed("foo")
    assert load_state().last_scan_ts == 9.0
