"""Tests for voiceio.hints — contextual CLI hints."""
from __future__ import annotations

import json
from unittest.mock import patch

import voiceio.hints as hints_mod
from voiceio.hints import hint


def test_hint_shows_on_tty(tmp_path, capsys):
    state_path = tmp_path / "hints.json"
    with (
        patch.object(hints_mod, "_HINTS_PATH", state_path),
        patch("sys.stderr") as mock_stderr,
        patch.dict("os.environ", {}, clear=False),
    ):
        mock_stderr.isatty.return_value = True
        mock_stderr.write = lambda s: None  # print goes through
        # Use capsys won't capture stderr mock, so check state file instead
        hint("test_id", "try this")
        state = json.loads(state_path.read_text())
        assert state["test_id"] == 1


def test_hint_skipped_when_not_tty(tmp_path):
    state_path = tmp_path / "hints.json"
    with (
        patch.object(hints_mod, "_HINTS_PATH", state_path),
        patch("sys.stderr") as mock_stderr,
    ):
        mock_stderr.isatty.return_value = False
        hint("test_id", "try this")
        assert not state_path.exists()


def test_hint_skipped_when_env_set(tmp_path):
    state_path = tmp_path / "hints.json"
    with (
        patch.object(hints_mod, "_HINTS_PATH", state_path),
        patch("sys.stderr") as mock_stderr,
        patch.dict("os.environ", {"VOICEIO_NO_HINTS": "1"}),
    ):
        mock_stderr.isatty.return_value = True
        hint("test_id", "try this")
        assert not state_path.exists()


def test_hint_stops_after_max_shows(tmp_path):
    state_path = tmp_path / "hints.json"
    state_path.write_text(json.dumps({"test_id": 3}))
    with (
        patch.object(hints_mod, "_HINTS_PATH", state_path),
        patch("sys.stderr") as mock_stderr,
        patch.dict("os.environ", {}, clear=False),
    ):
        mock_stderr.isatty.return_value = True
        hint("test_id", "try this")
        state = json.loads(state_path.read_text())
        assert state["test_id"] == 3  # unchanged


def test_hint_increments_count(tmp_path):
    state_path = tmp_path / "hints.json"
    state_path.write_text(json.dumps({"test_id": 1}))
    with (
        patch.object(hints_mod, "_HINTS_PATH", state_path),
        patch("sys.stderr") as mock_stderr,
        patch.dict("os.environ", {}, clear=False),
    ):
        mock_stderr.isatty.return_value = True
        hint("test_id", "try this")
        state = json.loads(state_path.read_text())
        assert state["test_id"] == 2
