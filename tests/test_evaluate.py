"""Tests for the decoder-config eval harness.

The harness exists because every decoder knob was set from a plausible story and
one example. Its own detectors must not be in the same position — hence real
assertions on the hallucination/repetition counters, using the exact shapes seen
in production.
"""
from __future__ import annotations

import wave
from unittest.mock import patch

import numpy as np

from voiceio.evaluate import (
    count_hallucinations,
    count_repetitions,
    default_matrix,
    evaluate,
    sample_clips,
)


class TestHallucinationDetection:
    def test_catches_whisper_filler(self):
        assert count_hallucinations([" Thank you."]) == 1
        assert count_hallucinations(["Thanks for watching!"]) == 1
        assert count_hallucinations([" Thank you.", " Thank you."]) == 2

    def test_ignores_real_speech(self):
        assert count_hallucinations(["Thank you for the detailed review"]) == 0
        assert count_hallucinations(["so I want you to own this end to end"]) == 0

    def test_matches_whole_segment_not_substring(self):
        """'thank you' inside a sentence is speech, not a hallucination."""
        assert count_hallucinations(["I say thank you to the team"]) == 0

    def test_empty(self):
        assert count_hallucinations([]) == 0
        assert count_hallucinations(["", "   "]) == 0


class TestRepetitionDetection:
    def test_catches_the_real_loop(self):
        """Verbatim from a 422s dictation that looped four times."""
        text = ("make a hypothesis why intelligence is not working out "
                + "because it should not work out in the complex world " * 4
                + "so we have seen simpler simulations")
        assert count_repetitions(text) >= 1

    def test_ignores_normal_speech(self):
        text = ("So on each generation we have an algorithm to determine the "
                "general shape of the brain and once that is determined we add "
                "one or two nodes or more hidden nodes and more connections")
        assert count_repetitions(text) == 0

    def test_ignores_short_text(self):
        assert count_repetitions("testing testing one two three") == 0

    def test_needs_a_real_run_not_two(self):
        """Saying something twice is emphasis; three-plus is a decode loop."""
        twice = "the quick brown fox the quick brown fox and then it stopped"
        assert count_repetitions(twice) == 0


class TestDefaultMatrix:
    def test_includes_shipped_baseline(self):
        names = [c.name for c in default_matrix()]
        assert "shipped" in names

    def test_shipped_matches_the_worker(self):
        """If worker.py's decode params drift, this baseline is a lie."""
        shipped = next(c for c in default_matrix() if c.name == "shipped")
        src = (__import__("pathlib").Path(__file__).resolve().parent.parent
               / "voiceio" / "worker.py").read_text()
        assert "condition_on_previous_text=False" in src
        assert "vad_filter=True" in src
        assert shipped.condition_on_previous_text is False
        assert shipped.vad_filter is True

    def test_never_pairs_cond_with_hotwords(self):
        """They share faster-whisper's sot_prev slot: 223 + 223 + sot > 448,
        which raises 'maximum decoding length must be > 0'. Not a config."""
        for c in default_matrix():
            assert not (c.condition_on_previous_text and c.hotwords), c.name


def _write_wav(path, secs=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(int(16000 * secs), dtype=np.int16)).tobytes())


class TestSampleClips:
    def test_respects_audio_budget(self, tmp_path, monkeypatch):
        rec = tmp_path / "recordings"
        rec.mkdir()
        for i in range(10):
            _write_wav(rec / f"c{i}.wav", secs=2.0)
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)
        got = sample_clips(5.0)
        assert 0 < len(got) <= 3  # 2s each, budget 5s

    def test_no_recordings_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", tmp_path / "nope")
        assert sample_clips(100) == []

    def test_includes_long_clips(self, tmp_path, monkeypatch):
        """The bias this replaced: newest-first sampling was ALL short clips
        (8 clips, 22s avg), so condition_on_previous_text's repetition loops —
        which only happen on long dictation — could not appear, and the harness
        happily recommended cond=True."""
        rec = tmp_path / "recordings"
        rec.mkdir()
        for i in range(30):  # many short, newest
            _write_wav(rec / f"short{i:02d}.wav", secs=10.0)
        _write_wav(rec / "long.wav", secs=120.0)
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)

        got = sample_clips(300)
        names = [p.name for p in got]
        assert "long.wav" in names
        assert any(n.startswith("short") for n in names)

    def test_all_short_still_fills_budget(self, tmp_path, monkeypatch):
        """No long recordings must not mean a half-empty sample."""
        rec = tmp_path / "recordings"
        rec.mkdir()
        for i in range(20):
            _write_wav(rec / f"s{i:02d}.wav", secs=10.0)
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)
        got = sample_clips(100)
        assert len(got) >= 9  # ~10 x 10s, not capped at the 50% short share


class TestEvaluate:
    def test_no_audio_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", tmp_path / "nope")
        from voiceio.config import Config
        assert evaluate(Config(), default_matrix(), teacher=lambda p: "x") == []

    def test_teacher_silence_returns_empty(self, tmp_path, monkeypatch):
        """A teacher that hears nothing gives nothing to score against."""
        rec = tmp_path / "recordings"
        rec.mkdir()
        _write_wav(rec / "a.wav")
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)
        from voiceio.config import Config
        assert evaluate(Config(), default_matrix(), teacher=lambda p: "") == []

    def test_scores_and_ranks_by_wer(self, tmp_path, monkeypatch):
        rec = tmp_path / "recordings"
        rec.mkdir()
        _write_wav(rec / "a.wav", secs=1.0)
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)

        class _Seg:
            def __init__(self, t):
                self.text = t

        class _Model:
            def transcribe(self, audio, **kw):
                # The no-vad config "hears" the reference exactly; shipped drops
                # a word. Ranking must follow WER, not declaration order.
                if kw.get("vad_filter"):
                    return iter([_Seg("hello there")]), None
                return iter([_Seg("hello there world")]), None

        from voiceio.config import Config
        with patch("voiceio.worker.load_model", return_value=_Model()), \
             patch("voiceio.vocabulary.load_terms", return_value=[]):
            scores = evaluate(Config(), default_matrix(),
                              teacher=lambda p: "hello there world")

        assert scores
        assert scores[0].wer <= scores[-1].wer          # sorted best-first
        assert scores[0].config.vad_filter is False     # the exact-match config
        assert all(s.clips == 1 for s in scores)

    def test_a_crashing_config_is_a_result_not_an_error(self, tmp_path, monkeypatch):
        rec = tmp_path / "recordings"
        rec.mkdir()
        _write_wav(rec / "a.wav")
        monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", rec)

        class _Model:
            def transcribe(self, audio, **kw):
                raise ValueError("The maximum decoding length must be > 0")

        from voiceio.config import Config
        with patch("voiceio.worker.load_model", return_value=_Model()), \
             patch("voiceio.vocabulary.load_terms", return_value=[]):
            scores = evaluate(Config(), default_matrix(), teacher=lambda p: "hi")

        assert all(s.crashes == 1 for s in scores)
        assert all(s.clips == 0 for s in scores)
