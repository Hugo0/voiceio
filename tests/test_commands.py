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
        cp = CommandProcessor(editing=True)
        result = cp.process("hello world scratch that")
        assert result == "hello world"
        assert cp.undo_requested is True

    def test_undo_that(self):
        cp = CommandProcessor(editing=True)
        result = cp.process("hello world undo that")
        assert result == "hello world"
        assert cp.undo_requested is True

    def test_scratch_that_only(self):
        cp = CommandProcessor(editing=True)
        result = cp.process("scratch that")
        assert result == ""
        assert cp.undo_requested is True

    def test_undo_resets_between_calls(self):
        cp = CommandProcessor(editing=True)
        cp.process("scratch that")
        assert cp.undo_requested is True
        cp.process("hello world")
        assert cp.undo_requested is False

    def test_editing_off_ignores_scratch(self):
        cp = CommandProcessor(editing=False)
        result = cp.process("hello world scratch that")
        assert result == "hello world scratch that"
        assert cp.undo_requested is False


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
    def test_correct_that_sets_flag(self):
        cp = CommandProcessor(editing=True)
        result = cp.process("hello world correct that")
        assert cp.flag_requested is True
        assert cp.flagged_word == "world"
        assert result == "hello"

    def test_correct_that_no_preceding_word(self):
        cp = CommandProcessor(editing=True)
        result = cp.process("correct that")
        assert cp.flag_requested is True
        assert cp.flagged_word == ""
        assert result == ""

    def test_correct_that_returns_text_before(self):
        cp = CommandProcessor(editing=True)
        result = cp.process("one two three correct that")
        assert result == "one two"
        assert cp.flagged_word == "three"

    def test_flag_resets_between_calls(self):
        cp = CommandProcessor(editing=True)
        cp.process("hello correct that")
        assert cp.flag_requested is True
        cp.process("hello world")
        assert cp.flag_requested is False
        assert cp.flagged_word == ""

    def test_undo_not_set_on_flag(self):
        cp = CommandProcessor(editing=True)
        cp.process("hello correct that")
        assert cp.flag_requested is True
        assert cp.undo_requested is False

    def test_editing_off_ignores_correct(self):
        cp = CommandProcessor(editing=False)
        result = cp.process("hello world correct that")
        assert result == "hello world correct that"
        assert cp.flag_requested is False


class TestSpacing:
    def test_no_double_spaces(self):
        cp = CommandProcessor()
        result = cp.process("hello comma world")
        assert "  " not in result

    def test_space_after_punct(self):
        cp = CommandProcessor()
        result = cp.process("hello period world")
        assert result == "hello. world"
