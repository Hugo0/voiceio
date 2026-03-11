"""Tests for the corrections dictionary."""
from __future__ import annotations

import json

import pytest

from voiceio.corrections import CorrectionDict


@pytest.fixture
def cd(tmp_path):
    """CorrectionDict backed by a temp file."""
    return CorrectionDict(path=tmp_path / "corrections.json")


class TestApply:
    def test_add_and_apply(self, cd):
        cd.add("Kosagra", "Kusagra")
        assert cd.apply("Hey Kosagra") == "Hey Kusagra"

    def test_case_insensitive(self, cd):
        cd.add("Kosagra", "Kusagra")
        assert cd.apply("hey kosagra") == "hey Kusagra"
        assert cd.apply("hey KOSAGRA") == "hey Kusagra"

    def test_whole_word_only(self, cd):
        cd.add("the", "they")
        assert cd.apply("there is the cat") == "there is they cat"
        assert "there" in cd.apply("there is the cat")

    def test_multi_word_key(self, cd):
        cd.add("machine learning", "ML")
        assert cd.apply("I like machine learning a lot") == "I like ML a lot"

    def test_multi_word_value(self, cd):
        cd.add("ML", "machine learning")
        assert cd.apply("I like ML") == "I like machine learning"

    def test_multiple_occurrences(self, cd):
        cd.add("foo", "bar")
        assert cd.apply("foo and foo") == "bar and bar"

    def test_empty_dict_passthrough(self, cd):
        assert cd.apply("hello world") == "hello world"

    def test_empty_text(self, cd):
        cd.add("foo", "bar")
        assert cd.apply("") == ""

    def test_longer_match_wins(self, cd):
        cd.add("new", "NEW")
        cd.add("new york", "NYC")
        assert cd.apply("I love new york") == "I love NYC"


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "corrections.json"
        cd1 = CorrectionDict(path=path)
        cd1.add("foo", "bar")
        cd1.add("baz", "qux")

        cd2 = CorrectionDict(path=path)
        assert cd2.apply("foo baz") == "bar qux"

    def test_remove(self, cd):
        cd.add("foo", "bar")
        assert cd.apply("foo") == "bar"
        assert cd.remove("foo") is True
        assert cd.apply("foo") == "foo"

    def test_remove_not_found(self, cd):
        assert cd.remove("nonexistent") is False

    def test_list_all(self, cd):
        cd.add("foo", "bar")
        cd.add("baz", "qux")
        result = cd.list_all()
        assert result == {"foo": "bar", "baz": "qux"}

    def test_atomic_save(self, tmp_path):
        path = tmp_path / "corrections.json"
        cd = CorrectionDict(path=path)
        cd.add("test", "value")
        # No .tmp file should remain
        assert not (tmp_path / "corrections.tmp").exists()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"test": "value"}


class TestFlagging:
    def test_flag_and_list(self, cd):
        cd.flag_word("hello")
        cd.flag_word("world")
        assert cd.list_flagged() == ["hello", "world"]

    def test_clear_flagged(self, cd):
        cd.flag_word("hello")
        cd.clear_flagged()
        assert cd.list_flagged() == []

    def test_flag_empty_word_ignored(self, cd):
        cd.flag_word("")
        cd.flag_word("  ")
        assert cd.list_flagged() == []

    def test_flagged_persistence(self, tmp_path):
        path = tmp_path / "corrections.json"
        cd1 = CorrectionDict(path=path)
        cd1.flag_word("test")

        cd2 = CorrectionDict(path=path)
        assert cd2.list_flagged() == ["test"]


class TestVocabulary:
    def test_vocabulary_terms(self, cd):
        cd.add("foo", "bar")
        cd.add("baz", "qux")
        terms = cd.vocabulary_terms()
        assert sorted(terms) == ["bar", "qux"]

    def test_vocabulary_terms_empty(self, cd):
        assert cd.vocabulary_terms() == []


class TestEdgeCases:
    def test_unicode_corrections(self, cd):
        cd.add("cafe", "caf\u00e9")
        assert cd.apply("I went to the cafe") == "I went to the caf\u00e9"

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "corrections.json"
        path.write_text("not json")
        cd = CorrectionDict(path=path)
        assert cd.apply("test") == "test"

    def test_load_non_dict_json(self, tmp_path):
        path = tmp_path / "corrections.json"
        path.write_text('["list", "not", "dict"]')
        cd = CorrectionDict(path=path)
        assert cd.apply("test") == "test"

    def test_missing_file(self, tmp_path):
        cd = CorrectionDict(path=tmp_path / "nonexistent.json")
        assert cd.apply("test") == "test"
