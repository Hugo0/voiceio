"""Tests for the IBus text size limits.

The bug these guard against: a preedit or commit string above libwayland's
4096-byte per-message buffer makes the compositor drop the focused
application's connection, killing it mid-dictation.
"""
from __future__ import annotations

import pytest

from voiceio.ibus.textlimit import (
    MAX_TEXT_BYTES,
    TRUNCATION_MARK,
    split_utf8,
    tail_utf8,
)

# The real 4096-byte ceiling every chunk has to clear, with room for the
# neighbouring events GNOME Shell may pack into the same buffer flush.
WAYLAND_BUFFER_BYTES = 4096


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


@pytest.fixture
def long_note() -> str:
    """A transcript the size of the ones that crashed Ghostty and Obsidian."""
    return " ".join(f"word{i}" for i in range(900))


def test_short_text_is_one_chunk():
    assert split_utf8("hello there") == ["hello there"]


def test_empty_text_yields_no_chunks():
    assert split_utf8("") == []


def test_chunks_rejoin_to_the_original(long_note):
    assert "".join(split_utf8(long_note)) == long_note


def test_every_chunk_fits_the_wayland_buffer(long_note):
    chunks = split_utf8(long_note)
    assert len(chunks) > 1
    assert all(_nbytes(c) <= MAX_TEXT_BYTES for c in chunks)
    assert all(_nbytes(c) < WAYLAND_BUFFER_BYTES for c in chunks)


def test_multibyte_characters_are_never_cut_in_half():
    # No spaces, so splitting has to fall back to character boundaries.
    text = "é€𝄞" * 2000
    chunks = split_utf8(text)
    assert "".join(chunks) == text
    assert all(_nbytes(c) <= MAX_TEXT_BYTES for c in chunks)


def test_splits_prefer_word_boundaries(long_note):
    # Every chunk but the last should end at a space, so commits do not land
    # mid-word in editors that treat each commit as a unit.
    chunks = split_utf8(long_note)
    assert all(c.endswith(" ") for c in chunks[:-1])


def test_a_run_without_spaces_still_splits():
    text = "x" * (MAX_TEXT_BYTES * 3)
    chunks = split_utf8(text)
    assert "".join(chunks) == text
    assert all(_nbytes(c) <= MAX_TEXT_BYTES for c in chunks)


def test_short_preedit_is_untouched():
    assert tail_utf8("hello there") == "hello there"


def test_long_preedit_is_bounded_and_marked(long_note):
    shown = tail_utf8(long_note)
    assert _nbytes(shown) <= MAX_TEXT_BYTES
    assert shown.startswith(TRUNCATION_MARK)


def test_preedit_keeps_the_most_recent_words(long_note):
    # The tail is what the speaker is saying right now — that is the half worth
    # showing when the preview has to be trimmed.
    assert tail_utf8(long_note).endswith(long_note[-200:])


def test_preedit_tail_survives_multibyte_text():
    text = "é" * 4000
    shown = tail_utf8(text)
    assert _nbytes(shown) <= MAX_TEXT_BYTES
    assert shown == TRUNCATION_MARK + "é" * ((MAX_TEXT_BYTES - _nbytes(TRUNCATION_MARK)) // 2)


def test_commit_secs_zero_for_one_chunk():
    from voiceio.ibus.textlimit import commit_secs
    assert commit_secs("short note") == 0.0


def test_commit_secs_covers_every_chunk_gap(long_note):
    from voiceio.ibus.textlimit import COMMIT_GAP_MS, commit_secs
    n = len(split_utf8(long_note))
    assert n > 1
    assert commit_secs(long_note) == n * COMMIT_GAP_MS / 1000


def test_ibus_typer_holds_a_chunked_commit_until_delivered(long_note, monkeypatch):
    """Returning early let the input source be restored while tail chunks
    were still pending, so a long note lost its end."""
    from voiceio.typers import ibus as ibus_typer
    t = ibus_typer.IBusTyper.__new__(ibus_typer.IBusTyper)
    sent, slept = [], []
    monkeypatch.setattr(t, "_send", sent.append, raising=False)
    monkeypatch.setattr(t, "_copy_to_clipboard", lambda _t: None, raising=False)
    monkeypatch.setattr(t, "_engine_has_focus", lambda: True, raising=False)
    monkeypatch.setattr(ibus_typer.time, "sleep", slept.append)
    t.commit_text(long_note)
    assert sent == [f"commit:{long_note}"]
    assert slept and slept[-1] >= 0.3
