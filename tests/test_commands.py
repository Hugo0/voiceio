"""Tests for voice command detection and replacement."""
from __future__ import annotations

from voiceio.commands import CommandProcessor


class TestPunctuation:
    def test_period(self):
        cp = CommandProcessor()
        assert cp.process("hello period") == "hello."

    def test_comma(self):
        cp = CommandProcessor()
        assert cp.process("hello comma world") == "hello, world"

    def test_question_mark(self):
        cp = CommandProcessor()
        assert cp.process("is that right question mark") == "is that right?"

    def test_exclamation_point(self):
        cp = CommandProcessor()
        assert cp.process("wow exclamation point") == "wow!"

    def test_exclamation_mark(self):
        cp = CommandProcessor()
        assert cp.process("wow exclamation mark") == "wow!"

    def test_colon(self):
        cp = CommandProcessor()
        assert cp.process("note colon important") == "note: important"

    def test_semicolon(self):
        cp = CommandProcessor()
        assert cp.process("hello semicolon world") == "hello; world"

    def test_full_stop(self):
        cp = CommandProcessor()
        assert cp.process("hello full stop") == "hello."

    def test_multiple_commands(self):
        cp = CommandProcessor()
        result = cp.process("hello comma how are you question mark")
        assert result == "hello, how are you?"


class TestFormatting:
    def test_new_line(self):
        cp = CommandProcessor()
        assert cp.process("hello new line world") == "hello\nworld"

    def test_newline_single_word(self):
        cp = CommandProcessor()
        assert cp.process("hello newline world") == "hello\nworld"

    def test_new_paragraph(self):
        cp = CommandProcessor()
        assert cp.process("hello new paragraph world") == "hello\n\nworld"


class TestUndo:
    def test_scratch_that(self):
        r = CommandProcessor(editing=True).parse("hello world scratch that")
        assert (r.text, r.undo) == ("hello world", True)

    def test_undo_that(self):
        r = CommandProcessor(editing=True).parse("hello world undo that")
        assert (r.text, r.undo) == ("hello world", True)

    def test_scratch_that_only(self):
        r = CommandProcessor(editing=True).parse("scratch that")
        assert (r.text, r.undo) == ("", True)

    def test_stateless_between_calls(self):
        cp = CommandProcessor(editing=True)
        assert cp.parse("scratch that").undo
        assert not cp.parse("hello world").undo

    def test_editing_off_ignores_scratch(self):
        r = CommandProcessor(editing=False).parse("hello world scratch that")
        assert (r.text, r.undo) == ("hello world scratch that", False)


class TestEdgeCases:
    def test_no_commands(self):
        cp = CommandProcessor()
        assert cp.process("hello world") == "hello world"

    def test_disabled(self):
        cp = CommandProcessor(enabled=False)
        assert cp.process("hello period") == "hello period"

    def test_empty(self):
        cp = CommandProcessor()
        assert cp.process("") == ""

    def test_periodically_not_matched(self):
        cp = CommandProcessor()
        assert cp.process("periodically") == "periodically"

    def test_case_insensitive(self):
        cp = CommandProcessor()
        assert cp.process("hello Period") == "hello."

    def test_whisper_trailing_punct(self):
        """Whisper sometimes adds punctuation after command words."""
        cp = CommandProcessor()
        assert cp.process("hello period.") == "hello."

    def test_command_at_start(self):
        cp = CommandProcessor()
        assert cp.process("comma hello") == ", hello"

    def test_quotes(self):
        cp = CommandProcessor()
        result = cp.process("he said open quote hello close quote")
        assert result == 'he said "hello"'

    def test_parens(self):
        cp = CommandProcessor()
        result = cp.process("see open paren note close paren")
        assert result == "see (note)"


class TestCorrectThat:
    def test_correct_that_flags_previous_word(self):
        r = CommandProcessor(editing=True).parse("hello world correct that")
        assert (r.text, r.flagged_word) == ("hello", "world")

    def test_correct_that_no_preceding_word(self):
        r = CommandProcessor(editing=True).parse("correct that")
        assert (r.text, r.flagged_word) == ("", "")

    def test_correct_that_returns_text_before(self):
        r = CommandProcessor(editing=True).parse("one two three correct that")
        assert (r.text, r.flagged_word) == ("one two", "three")

    def test_no_flag_without_command(self):
        cp = CommandProcessor(editing=True)
        assert cp.parse("hello correct that").flagged_word == "hello"
        assert cp.parse("hello world").flagged_word is None
