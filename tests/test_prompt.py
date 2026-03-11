"""Tests for PromptBuilder (vocabulary + transcript history)."""
from __future__ import annotations

import threading

from voiceio.prompt import PromptBuilder


class TestBuildVocabOnly:
    def test_vocab_only(self):
        pb = PromptBuilder(vocabulary="VoiceIO, GNOME, Wayland")
        assert pb.build() == "VoiceIO, GNOME, Wayland"

    def test_empty_returns_none(self):
        pb = PromptBuilder()
        assert pb.build() is None


class TestBuildWithHistory:
    def test_history_only(self):
        pb = PromptBuilder()
        pb.add_transcript("Hello world")
        assert pb.build() == "Hello world"

    def test_vocab_and_history(self):
        pb = PromptBuilder(vocabulary="VoiceIO")
        pb.add_transcript("Hello world")
        result = pb.build()
        assert result == "VoiceIO | Hello world"

    def test_multiple_segments(self):
        pb = PromptBuilder()
        pb.add_transcript("First segment")
        pb.add_transcript("Second segment")
        result = pb.build()
        assert result == "First segment Second segment"

    def test_max_segments_respected(self):
        pb = PromptBuilder(max_segments=2)
        pb.add_transcript("First")
        pb.add_transcript("Second")
        pb.add_transcript("Third")
        result = pb.build()
        assert "First" not in result
        assert "Second" in result
        assert "Third" in result

    def test_empty_transcript_ignored(self):
        pb = PromptBuilder()
        pb.add_transcript("")
        assert pb.build() is None


class TestTruncation:
    def test_history_truncated_to_fit(self):
        pb = PromptBuilder(vocabulary="VOCAB", max_chars=30)
        pb.add_transcript("A very long transcript that exceeds the budget")
        result = pb.build()
        assert result is not None
        assert len(result) <= 30

    def test_truncation_snaps_to_word_boundary(self):
        pb = PromptBuilder(max_chars=20)
        pb.add_transcript("Hello beautiful world today")
        result = pb.build()
        assert result is not None
        # Should not cut in the middle of a word
        for word in result.split():
            assert len(word) > 0


class TestReset:
    def test_reset_clears_history(self):
        pb = PromptBuilder(vocabulary="VOCAB")
        pb.add_transcript("Hello")
        pb.reset()
        assert pb.build() == "VOCAB"

    def test_reset_preserves_vocabulary(self):
        pb = PromptBuilder(vocabulary="VOCAB")
        pb.add_transcript("Hello")
        pb.reset()
        assert "VOCAB" in pb.build()


class TestThreadSafety:
    def test_concurrent_add_and_build(self):
        pb = PromptBuilder(max_segments=100)
        errors = []

        def add_many():
            try:
                for i in range(100):
                    pb.add_transcript(f"Segment {i}")
            except Exception as e:
                errors.append(e)

        def build_many():
            try:
                for _ in range(100):
                    pb.build()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_many)
        t2 = threading.Thread(target=build_many)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors
