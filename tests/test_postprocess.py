"""Tests for text post-processing (punctuation, capitalization)."""
from __future__ import annotations

from voiceio.postprocess import cleanup
from voiceio.streaming import _word_match_len


class TestCapitalization:
    def test_first_char(self):
        assert cleanup("hello world") == "Hello world"

    def test_after_period(self):
        assert cleanup("hello. world") == "Hello. World"

    def test_after_question_mark(self):
        assert cleanup("really? yes") == "Really? Yes"

    def test_after_exclamation(self):
        assert cleanup("wow! nice") == "Wow! Nice"

    def test_already_capitalized(self):
        assert cleanup("Hello World") == "Hello World"

    def test_single_char(self):
        assert cleanup("a") == "A"


class TestSpacing:
    def test_double_spaces(self):
        assert cleanup("hello  world") == "Hello world"

    def test_triple_spaces(self):
        assert cleanup("hello   world") == "Hello world"

    def test_space_after_period(self):
        assert cleanup("hello.world") == "Hello. World"

    def test_space_before_comma(self):
        assert cleanup("hello , world") == "Hello, world"

    def test_space_before_period(self):
        assert cleanup("hello .") == "Hello."

    def test_leading_trailing_whitespace(self):
        assert cleanup("  hello world  ") == "Hello world"


class TestIdempotency:
    def test_basic(self):
        text = "hello world"
        assert cleanup(cleanup(text)) == cleanup(text)

    def test_with_punctuation(self):
        text = "hello. world? yes! ok"
        assert cleanup(cleanup(text)) == cleanup(text)

    def test_complex(self):
        text = "  hello  .world  ,  how  are  you  ?  fine  "
        assert cleanup(cleanup(text)) == cleanup(text)


class TestNonLatinLanguages:
    def test_chinese_no_capitalization(self):
        result = cleanup("你好世界", language="zh")
        assert result == "你好世界"

    def test_japanese_no_capitalization(self):
        result = cleanup("こんにちは", language="ja")
        assert result == "こんにちは"

    def test_still_normalizes_spaces(self):
        result = cleanup("  hello  world  ", language="zh")
        assert result == "hello world"


class TestEdgeCases:
    def test_empty_string(self):
        assert cleanup("") == ""

    def test_whitespace_only(self):
        assert cleanup("   ") == ""

    def test_all_punctuation(self):
        result = cleanup("...")
        assert result == "..."

    def test_unicode_accents(self):
        assert cleanup("café. résumé") == "Café. Résumé"


class TestWordMatchCompatibility:
    """Verify cleanup doesn't break streaming word-level matching."""

    def test_capitalization_invisible_to_matching(self):
        raw = ["hello", "world"]
        cleaned = ["Hello", "World"]
        assert _word_match_len(raw, cleaned) == 2

    def test_punct_spacing_invisible_to_matching(self):
        raw = ["hello,", "world"]
        cleaned = ["hello,", "world"]
        assert _word_match_len(raw, cleaned) == 2

    def test_mixed_changes_invisible(self):
        raw = ["testing,", "testing,", "hello"]
        cleaned = ["Testing,", "testing", "hello"]
        assert _word_match_len(raw, cleaned) == 3
