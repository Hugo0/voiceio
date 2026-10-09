"""Tests for the composable text pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

from voiceio.commands import CommandProcessor
from voiceio.config import Config
from voiceio.core.pipeline import (
    Commands, Pipeline, Prefix, StageContext, StageResult,
)


class Upper:
    name = "upper"
    final_only = True

    def __call__(self, text, ctx):
        return StageResult(text.upper())


def test_final_only_stages_skip_interim_passes():
    pipe = Pipeline([Upper()])
    assert pipe("hi", final=False).text == "hi"
    assert pipe("hi", final=True).text == "HI"


def test_timings_recorded_per_stage():
    result = Pipeline([Upper(), Prefix("[v]")])("hi", final=True)
    assert set(result.timings) == {"upper", "prefix"}
    assert result.text == "[v] HI"


def test_scratch_that_aborts_and_stops_later_stages():
    later = MagicMock()
    later.final_only = False
    pipe = Pipeline([Commands(CommandProcessor(editing=True)), later])
    result = pipe("hello world scratch that", final=False)
    assert result.abort and result.text == ""
    later.assert_not_called()


def test_correct_that_flags_only_on_final_pass():
    """Regression: the flag was written to disk on every interim pass."""
    flagged = []
    pipe = Pipeline([Commands(CommandProcessor(editing=True))], on_flag=flagged.append)
    assert pipe("hello world correct that", final=False).abort
    assert flagged == []
    assert pipe("hello world correct that", final=True).abort
    assert flagged == ["world"]


def test_from_config_builds_stack_in_order():
    cfg = Config()
    cfg.output.remove_disfluencies = True
    cfg.output.punctuation_cleanup = True
    cfg.output.number_conversion = True
    cfg.output.voice_input_prefix = "[voice]"
    corrections = MagicMock()
    pipe = Pipeline.from_config(cfg, corrections=corrections,
                                commands=CommandProcessor())
    assert [s.name for s in pipe.stages] == [
        "disfluency", "cleanup", "numbers", "commands", "corrections", "prefix"]


def test_prefix_strip_roundtrip():
    p = Prefix("[voice]")
    assert p.strip(p("hello", StageContext(final=True)).text) == "hello"
