"""Regression guard: the engine must never hand IBus an oversized string.

``test_ibus_textlimit`` proves the limiter itself is correct. This pins the
engine to actually routing through it, so the crash cannot come back via an
edit that bypasses the limiter. A stub stands in for the IBus GObject bindings,
which are a system package and are not installed in the test venv.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from voiceio.ibus.textlimit import MAX_TEXT_BYTES


class _Text:
    """Stand-in for IBus.Text that just remembers its string."""

    def __init__(self, string: str):
        self.string = string

    @staticmethod
    def new_from_string(string: str) -> "_Text":
        return _Text(string)

    def append_attribute(self, *args) -> None:
        pass


class _Recorder:
    """Stands in for ``self``, recording what the engine sends to IBus."""

    def __init__(self):
        self.committed: list[str] = []
        self.preedits: list[str] = []
        self.hides = 0

    def hide_preedit_text(self) -> None:
        self.hides += 1

    def commit_text(self, text: _Text) -> None:
        self.committed.append(text.string)

    def update_preedit_text(self, text: _Text, cursor: int, visible: bool) -> None:
        self.preedits.append(text.string)


@pytest.fixture
def engine(monkeypatch):
    ibus = types.SimpleNamespace(
        Engine=type("Engine", (), {}),
        Factory=type("Factory", (), {}),
        Text=_Text,
        AttrType=types.SimpleNamespace(UNDERLINE=1),
        AttrUnderline=types.SimpleNamespace(SINGLE=1),
    )
    repository = types.ModuleType("gi.repository")
    repository.IBus = ibus
    repository.GLib = types.SimpleNamespace()
    repository.GObject = types.SimpleNamespace()
    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    gi.repository = repository

    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    sys.modules.pop("voiceio.ibus.engine", None)
    module = importlib.import_module("voiceio.ibus.engine")
    yield module.VoiceIOEngine
    sys.modules.pop("voiceio.ibus.engine", None)


@pytest.fixture
def long_note() -> str:
    """A transcript the size of the ones that crashed Ghostty and Obsidian."""
    return " ".join(f"word{i}" for i in range(900))


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def test_commit_chunks_stay_under_the_limit(engine, long_note):
    rec = _Recorder()
    engine.commit(rec, long_note)
    assert len(rec.committed) > 1
    assert all(_nbytes(c) <= MAX_TEXT_BYTES for c in rec.committed)


def test_commit_delivers_the_whole_note(engine, long_note):
    rec = _Recorder()
    engine.commit(rec, long_note)
    assert "".join(rec.committed) == long_note


def test_commit_hides_preedit_first(engine, long_note):
    rec = _Recorder()
    engine.commit(rec, long_note)
    assert rec.hides == 1


def test_empty_commit_sends_nothing(engine):
    rec = _Recorder()
    engine.commit(rec, "")
    assert rec.committed == []
    assert rec.hides == 1


def test_preedit_stays_under_the_limit(engine, long_note):
    rec = _Recorder()
    engine.preedit(rec, long_note)
    assert len(rec.preedits) == 1
    assert _nbytes(rec.preedits[0]) <= MAX_TEXT_BYTES


def test_short_preedit_is_shown_verbatim(engine):
    rec = _Recorder()
    engine.preedit(rec, "hello there")
    assert rec.preedits == ["hello there"]


def test_empty_preedit_hides_instead(engine):
    rec = _Recorder()
    engine.preedit(rec, "")
    assert rec.preedits == []
    assert rec.hides == 1
