"""Tests for transcription history."""
from __future__ import annotations

import json

from voiceio import history


class TestAppend:
    def test_append_creates_file(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("hello world", path=p)
        assert p.exists()
        entries = p.read_text().strip().split("\n")
        assert len(entries) == 1
        data = json.loads(entries[0])
        assert data["text"] == "hello world"
        assert "ts" in data

    def test_append_multiple(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("first", path=p)
        history.append("second", path=p)
        entries = p.read_text().strip().split("\n")
        assert len(entries) == 2

    def test_append_empty_ignored(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("", path=p)
        history.append("  ", path=p)
        assert not p.exists()

    def test_append_strips_whitespace(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("  hello  ", path=p)
        data = json.loads(p.read_text().strip())
        assert data["text"] == "hello"


class TestRead:
    def test_read_empty(self, tmp_path):
        assert history.read(path=tmp_path / "nope.jsonl") == []

    def test_read_newest_first(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("first", path=p)
        history.append("second", path=p)
        entries = history.read(path=p)
        assert entries[0]["text"] == "second"
        assert entries[1]["text"] == "first"

    def test_read_with_limit(self, tmp_path):
        p = tmp_path / "history.jsonl"
        for i in range(10):
            history.append(f"entry {i}", path=p)
        entries = history.read(path=p, limit=3)
        assert len(entries) == 3
        assert entries[0]["text"] == "entry 9"


class TestSearch:
    def test_search_found(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("hello world", path=p)
        history.append("goodbye world", path=p)
        history.append("hello there", path=p)
        results = history.search("hello", path=p)
        assert len(results) == 2

    def test_search_case_insensitive(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("Hello World", path=p)
        results = history.search("hello", path=p)
        assert len(results) == 1

    def test_search_no_results(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("hello", path=p)
        assert history.search("xyz", path=p) == []


class TestClear:
    def test_clear(self, tmp_path):
        p = tmp_path / "history.jsonl"
        history.append("hello", path=p)
        history.clear(path=p)
        assert not p.exists()

    def test_clear_nonexistent(self, tmp_path):
        history.clear(path=tmp_path / "nope.jsonl")  # no error
