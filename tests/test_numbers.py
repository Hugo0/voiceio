"""Tests for number word to digit conversion."""
from __future__ import annotations

import pytest

from voiceio.numbers import convert_numbers


class TestBasicCardinals:
    def test_single_digits(self):
        assert convert_numbers("I have five cats") == "I have 5 cats"

    def test_teens(self):
        assert convert_numbers("she is thirteen") == "she is 13"

    def test_tens(self):
        assert convert_numbers("about twenty people") == "about 20 people"

    def test_compound(self):
        assert convert_numbers("twenty five dollars") == "25 dollars"

    def test_hundred(self):
        assert convert_numbers("three hundred people") == "300 people"

    def test_hundred_and(self):
        assert convert_numbers("three hundred and forty two") == "342"

    def test_thousand(self):
        assert convert_numbers("two thousand") == "2000"

    def test_large(self):
        assert convert_numbers("two thousand five hundred") == "2500"

    def test_a_hundred(self):
        assert convert_numbers("a hundred people") == "100 people"

    def test_a_thousand(self):
        assert convert_numbers("a thousand times") == "1000 times"

    def test_zero(self):
        assert convert_numbers("zero issues") == "0 issues"


class TestPercentages:
    def test_basic_percent(self):
        assert convert_numbers("twenty five percent") == "25%"

    def test_hundred_percent(self):
        assert convert_numbers("one hundred percent") == "100%"


class TestOrdinals:
    def test_first(self):
        assert convert_numbers("the first time") == "the 1st time"

    def test_second(self):
        assert convert_numbers("the second try") == "the 2nd try"

    def test_third(self):
        assert convert_numbers("the third option") == "the 3rd option"

    def test_fifth(self):
        assert convert_numbers("the fifth element") == "the 5th element"

    def test_twentieth(self):
        assert convert_numbers("the twentieth century") == "the 20th century"


class TestEdgeCases:
    def test_no_numbers(self):
        assert convert_numbers("hello world") == "hello world"

    def test_empty(self):
        assert convert_numbers("") == ""

    def test_non_english(self):
        assert convert_numbers("trois cent", language="fr") == "trois cent"

    def test_mixed(self):
        assert convert_numbers("I ate three tacos and five burritos") == "I ate 3 tacos and 5 burritos"

    def test_number_with_trailing_punct(self):
        result = convert_numbers("I have five.")
        assert result == "I have 5."

    def test_preserves_non_number_words(self):
        assert convert_numbers("the quick brown fox") == "the quick brown fox"

    def test_and_not_consumed_alone(self):
        # "and" alone shouldn't be consumed as a number word
        assert convert_numbers("bread and butter") == "bread and butter"


class TestPipeline:
    """Test that numbers work after cleanup (capitalized text)."""
    def test_capitalized_input(self):
        assert convert_numbers("Twenty Five Percent") == "25%"

    def test_sentence(self):
        result = convert_numbers("There are three hundred people here")
        assert result == "There are 300 people here"
