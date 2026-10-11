"""Tests for text post-processing (punctuation, capitalization)."""
from __future__ import annotations

from voiceio.config import Config
from voiceio.core.pipeline import Pipeline
from voiceio.postprocess import cleanup, strip_disfluencies
from voiceio.display import _word_match_len


def _run(text: str, *, final: bool = False, **output) -> tuple[str, bool]:
    """The daemon's pipeline with every optional stage off unless named."""
    cfg = Config()
    out = {"punctuation_cleanup": False, "number_conversion": False,
           "remove_disfluencies": False, "voice_input_prefix": ""} | output
    for key, value in out.items():
        setattr(cfg.output, key, value)
    result = Pipeline.from_config(cfg)(text, final=final)
    return result.text, result.abort


class TestStripDisfluencies:
    """Delete-only regex layer: filler sounds + duplicate re-decode sentences,
    never anything that could change meaning."""

    def test_removes_filler_sounds(self):
        out = strip_disfluencies("so um we should uh ship it")
        assert "um" not in out.split()
        assert "uh" not in out.split()
        assert "we should" in out and "ship it" in out

    def test_removes_comma_wrapped_filler(self):
        assert "uh" not in strip_disfluencies("we tested, uh, everything").split()

    def test_preserves_valid_word_repetition(self):
        # "had had" is valid English; regex must NOT touch word repeats — that
        # judgment belongs to the LLM layer.
        assert strip_disfluencies("I had had enough") == "I had had enough"

    def test_dedups_duplicate_sentence(self):
        # The Whisper re-decode artifact: a whole sentence repeated verbatim.
        out = strip_disfluencies("Do the deep research now. Do the deep research now.")
        assert out == "Do the deep research now."

    def test_keeps_meaningful_words(self):
        # 'like' as a real verb/preposition and content must survive.
        text = "I like the design and it works like a charm"
        assert strip_disfluencies(text) == text

    def test_never_eats_real_words_or_units(self):
        # Default-on runs on everyone's speech: filler patterns must not collide
        # with real words, units, or abbreviations. Case-sensitive matching is
        # what protects the all-caps abbreviations (ER, UM, HM).
        for text in [
            "to err is human",
            "we should err on caution",
            "the bolt is 5 mm wide",
            "set it to 10 mm please",
            "ah yes I remember now",
            "I like the ohm rating",
            "Take him to the ER right now",       # ER = emergency room
            "The UM campus in Michigan",           # UM = University of Michigan
            "The Er atom is a lanthanide",         # Er = erbium
            "we measured 3 hm across",             # hm = hectometre (bare, 1 m)
        ]:
            assert strip_disfluencies(text) == text, text

    def test_preserves_newlines_and_structure(self):
        # A filler on its own line must not swallow the paragraph break.
        assert strip_disfluencies("First para.\n\nSecond para.") == \
            "First para.\n\nSecond para."
        assert "\n\n" in strip_disfluencies("First para.\n\num\n\nSecond para.")

    def test_uh_huh_removed_whole(self):
        # Regression: ordering bug once stranded "-huh".
        assert strip_disfluencies("uh-huh right") == "right"
        assert strip_disfluencies("uh huh yes") == "yes"

    def test_preserves_emphatic_short_repeat(self):
        # Short repeats are emphasis, not a re-decode artifact — keep them.
        assert strip_disfluencies("No. No.") == "No. No."

    def test_empty(self):
        assert strip_disfluencies("") == ""


class TestPipelineDisfluencies:
    def test_flag_off_keeps_fillers(self):
        text, _ = _run("um hello there friend", punctuation_cleanup=True, final=True)
        assert "um" in text.lower().split()

    def test_flag_on_strips_fillers(self):
        text, _ = _run(
            "um hello there friend",
            punctuation_cleanup=True, remove_disfluencies=True, final=True,
        )
        assert "um" not in text.lower().split()
        assert "hello there friend" in text.lower()


class TestCapitalization:
    def test_first_char(self):
        assert cleanup("hello world") == "Hello world"

    def test_after_period(self):
        assert cleanup("hello? world") == "Hello? World"
        assert cleanup("Hello. World") == "Hello. World"

    def test_lowercase_after_a_segment_stop_is_a_comma(self):
        # Whisper ends each segment with "." and continues the sentence in
        # lowercase; capitalizing it made a false stop permanent.
        assert cleanup("a bit bigger. and remove the latch.") == "A bit bigger, and remove the latch."
        assert cleanup("hello. world") == "Hello, world"

    def test_abbreviations_and_dotted_names_keep_their_dot(self):
        assert cleanup("pears etc. and more") == "Pears etc. and more"
        assert cleanup("use e.g. this one") == "Use e.g. this one"
        assert cleanup("go to hugo0.com. and then") == "Go to hugo0.com. and then"
        assert cleanup("I paused... and then") == "I paused... and then"

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

    def test_dots_inside_tokens_survive(self):
        for text in ["see claude.ai", "open roadmap.md", "the Print.Observer class"]:
            assert cleanup(text)[1:] == text[1:]

    def test_space_after_question_mark(self):
        assert cleanup("really?yes") == "Really? Yes"

    def test_segment_continuation_ellipsis(self):
        assert cleanup("interesting... ...evolving fast") == "Interesting... evolving fast"

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
        assert cleanup("café? résumé") == "Café? Résumé"


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


class TestVoiceInputPrefix:
    """The prefix marker is prepended only when configured + text is non-empty."""

    def test_disabled_by_default(self):
        text, abort = _run("hello world", final=True)
        assert text == "hello world"
        assert abort is False

    def test_applied_when_set_final(self):
        text, abort = _run(
            "hello world", voice_input_prefix="[voice]", final=True,
        )
        assert text == "[voice] hello world"
        assert abort is False

    def test_applied_during_streaming(self):
        # Streaming passes (final=False) must also carry the prefix so the
        # marker appears from the first chunk, not only at the very end.
        text, _ = _run(
            "partial", voice_input_prefix="[voice]", final=False,
        )
        assert text == "[voice] partial"

    def test_not_applied_to_empty_text(self):
        text, abort = _run(
            "", voice_input_prefix="[voice]", final=True,
        )
        assert text == ""
        assert abort is False

    def test_custom_prefix(self):
        text, _ = _run(
            "ok", voice_input_prefix="[v]", final=True,
        )
        assert text == "[v] ok"

    def test_with_cleanup_chain(self):
        # Cleanup capitalizes, then prefix is prepended verbatim.
        text, _ = _run(
            "hello world",
            punctuation_cleanup=True,
            voice_input_prefix="[voice]",
            final=True,
        )
        assert text == "[voice] Hello world"
