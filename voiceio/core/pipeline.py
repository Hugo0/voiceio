"""The text post-processing pipeline as composable stages.

Raw decoder text → [disfluency → cleanup → numbers → commands → corrections →
postcorrect → llm → prefix] → what gets typed. Each stage is a small object
that can be configured, tested and timed on its own; front ends pick a stack
with `Pipeline.from_config(..., profile=...)`.

Stages are pure with respect to the pipeline: they return text plus
*effects* ("the user said scratch that", "flag this word") instead of setting
flags on shared objects. The pipeline applies side effects only on the final
pass, so interim passes never write to disk and a finalizer that overlaps the
next recording can't leak state into it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from voiceio.commands import CommandProcessor
    from voiceio.config import Config
    from voiceio.corrections import CorrectionDict
    from voiceio.llm import LLMProcessor
    from voiceio.postcorrect import PostCorrector

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageContext:
    final: bool
    language: str = "en"


@dataclass(frozen=True)
class StageResult:
    text: str
    abort: bool = False                 # discard the utterance (e.g. "scratch that")
    flagged_word: str | None = None     # "correct that": queue this word for review


class Stage(Protocol):
    name: str
    final_only: bool

    def __call__(self, text: str, ctx: StageContext) -> StageResult: ...


@dataclass
class PipelineResult:
    text: str
    abort: bool = False
    timings: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, stages: Sequence[Stage], *, language: str = "en",
                 on_flag: Callable[[str], None] | None = None):
        self.stages = list(stages)
        self._language = language
        self._on_flag = on_flag

    def __call__(self, text: str, *, final: bool = False) -> PipelineResult:
        ctx = StageContext(final=final, language=self._language)
        timings: dict[str, float] = {}
        for stage in self.stages:
            if not text:
                break
            if stage.final_only and not final:
                continue
            t0 = time.monotonic()
            result = stage(text, ctx)
            timings[stage.name] = round(time.monotonic() - t0, 3)
            if result.abort:
                if final and result.flagged_word and self._on_flag is not None:
                    self._on_flag(result.flagged_word)
                return PipelineResult("", abort=True, timings=timings)
            text = result.text
        return PipelineResult(text, timings=timings)

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        corrections: CorrectionDict | None = None,
        commands: CommandProcessor | None = None,
        postcorrect: PostCorrector | None = None,
        llm: LLMProcessor | None = None,
        prefix: bool = True,
    ) -> Pipeline:
        out = cfg.output
        stages: list[Stage] = []
        if out.remove_disfluencies:
            stages.append(Disfluency())
        if out.punctuation_cleanup:
            stages.append(Cleanup())
        if out.number_conversion:
            stages.append(Numbers())
        if commands is not None:
            stages.append(Commands(commands))
        if corrections is not None:
            stages.append(Corrections(corrections))
        if postcorrect is not None:
            stages.append(PostCorrect(postcorrect))
        if llm is not None:
            stages.append(LLMCleanup(llm))
        if prefix and out.voice_input_prefix:
            stages.append(Prefix(out.voice_input_prefix))
        on_flag = corrections.flag_word if corrections is not None else None
        return cls(stages, language=cfg.model.language, on_flag=on_flag)


# ── stages ───────────────────────────────────────────────────────────────

class Disfluency:
    """Delete filler sounds and duplicated re-decode sentences. Runs before
    cleanup so cleanup repairs the spacing the deletions leave."""
    name = "disfluency"
    final_only = False

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        from voiceio.postprocess import strip_disfluencies
        return StageResult(strip_disfluencies(text))


class Cleanup:
    name = "cleanup"
    final_only = False

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        from voiceio.postprocess import cleanup
        return StageResult(cleanup(text, ctx.language))


class Numbers:
    name = "numbers"
    final_only = False

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        from voiceio.numbers import convert_numbers
        return StageResult(convert_numbers(text, ctx.language))


class Commands:
    """Spoken punctuation/formatting, and (with editing on) "scratch that" /
    "correct that"."""
    name = "commands"
    final_only = False

    def __init__(self, processor: CommandProcessor):
        self._processor = processor

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        parsed = self._processor.parse(text)
        if parsed.undo or parsed.flagged_word is not None:
            return StageResult("", abort=True, flagged_word=parsed.flagged_word or None)
        return StageResult(parsed.text)


class Corrections:
    name = "corrections"
    final_only = False

    def __init__(self, corrections: CorrectionDict):
        self._corrections = corrections

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        return StageResult(self._corrections.apply(text))


class PostCorrect:
    """Opt-in constrained cloud LLM fix of misheard words (final only)."""
    name = "postcorrect"
    final_only = True

    def __init__(self, postcorrect: PostCorrector):
        self._pc = postcorrect

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        return StageResult(self._pc.correct(text))


class LLMCleanup:
    name = "llm"
    final_only = True

    def __init__(self, llm: LLMProcessor):
        self._llm = llm

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        return StageResult(self._llm.process(text))


class Prefix:
    """Marks dictated text (e.g. "[voice]"). Applied on every pass so the
    marker appears from the first streaming chunk."""
    name = "prefix"
    final_only = False

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, text: str, ctx: StageContext) -> StageResult:
        return StageResult(f"{self.prefix} {text}")

    def strip(self, text: str) -> str:
        if text.startswith(self.prefix):
            return text[len(self.prefix):].lstrip()
        return text
