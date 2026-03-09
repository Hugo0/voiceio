"""Tests for streaming transcription with word-level append."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from voiceio.typers.base import TyperBackend

import numpy as np
import pytest

from voiceio.streaming import (
    StreamingSession,
    _common_prefix_len,
    _word_match_len,
)
from voiceio.typers.base import StreamingTyper


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --- _common_prefix_len ---

class TestCommonPrefixLen:
    def test_identical(self):
        assert _common_prefix_len("hello", "hello") == 5

    def test_partial(self):
        assert _common_prefix_len("hello world", "hello there") == 6

    def test_empty(self):
        assert _common_prefix_len("", "") == 0
        assert _common_prefix_len("", "hello") == 0

    def test_no_match(self):
        assert _common_prefix_len("abc", "xyz") == 0

    def test_prefix_of(self):
        assert _common_prefix_len("hello", "hello world") == 5


# --- _word_match_len ---

class TestWordMatchLen:
    def test_identical(self):
        assert _word_match_len(["hello", "world"], ["hello", "world"]) == 2

    def test_punctuation_ignored(self):
        assert _word_match_len(["Hello,"], ["Hello"]) == 1
        assert _word_match_len(["hello."], ["hello,"]) == 1
        assert _word_match_len(["Testing"], ["Testing,"]) == 1

    def test_case_ignored(self):
        assert _word_match_len(["Hello"], ["hello"]) == 1

    def test_partial_match(self):
        assert _word_match_len(
            ["hello", "world", "foo"],
            ["hello", "world", "bar"],
        ) == 2

    def test_no_match(self):
        assert _word_match_len(["hello"], ["goodbye"]) == 0

    def test_empty(self):
        assert _word_match_len([], ["hello"]) == 0
        assert _word_match_len(["hello"], []) == 0


# --- _apply_correction ---

class TestApplyCorrection:
    def _make_session(self):
        return StreamingSession(
            transcriber=MagicMock(),
            typer=MagicMock(spec=TyperBackend),
            recorder=MagicMock(),
        )

    def test_initial_type(self):
        """First transcription: type everything."""
        s = self._make_session()
        s._apply_correction("Hello world")
        s._typer.type_text.assert_called_once_with("Hello world")
        s._typer.delete_chars.assert_not_called()
        assert s._typed_text == "Hello world"

    def test_word_append(self):
        """New text extends typed words: append only new part."""
        s = self._make_session()
        s._typed_text = "Hello world foo"  # >2 words to skip short-text path
        s._apply_correction("Hello, world, foo, how are you")
        s._typer.type_text.assert_called_once_with(" how are you")
        s._typer.delete_chars.assert_not_called()
        assert s._typed_text == "Hello world foo how are you"

    def test_punct_change_still_appends(self):
        """Whisper changes commas - word match ignores this, appends new."""
        s = self._make_session()
        s._typed_text = "Testing, testing, testing"  # >2 words
        s._apply_correction("Testing testing testing hello")
        s._typer.type_text.assert_called_once_with(" hello")
        s._typer.delete_chars.assert_not_called()

    def test_word_mismatch_skips(self):
        """Words don't match: skip entirely (no append, no delete)."""
        s = self._make_session()
        s._typed_text = "Hello world foo"
        s._apply_correction("Hello world bar baz")
        # "foo" vs "bar" - mismatch at word 3 (matched 2 < 3 typed)
        s._typer.type_text.assert_not_called()
        s._typer.delete_chars.assert_not_called()
        assert s._typed_text == "Hello world foo"

    def test_no_change(self):
        """Same text: nothing happens."""
        s = self._make_session()
        s._typed_text = "Hello"
        s._apply_correction("Hello")
        s._typer.type_text.assert_not_called()
        s._typer.delete_chars.assert_not_called()

    def test_shorter_text_skips(self):
        """Shorter new text: skip (no deletions during streaming)."""
        s = self._make_session()
        s._typed_text = "Hello world how are you"
        s._apply_correction("Hello world")
        # All words match but no new words → skip
        s._typer.type_text.assert_not_called()
        s._typer.delete_chars.assert_not_called()

    def test_final_allows_full_correction(self):
        """Final pass: char-level diff with deletions."""
        s = self._make_session()
        s._typed_text = "Hello, world, how"
        s._apply_correction("Hello world how are you", final=True)
        # prefix="Hello" (5), delete 12, type " world how are you"
        s._typer.delete_chars.assert_called_once_with(12)
        s._typer.type_text.assert_called_once_with(" world how are you")
        assert s._typed_text == "Hello world how are you"

    def test_final_no_change_needed(self):
        """Final pass with matching text: nothing happens."""
        s = self._make_session()
        s._typed_text = "Hello world"
        s._apply_correction("Hello world", final=True)
        s._typer.type_text.assert_not_called()
        s._typer.delete_chars.assert_not_called()

    def test_no_deletions_after_warmup(self):
        """After initial warmup (>2 words), streaming never deletes."""
        s = self._make_session()
        # Simulate warmup
        s._typed_text = "Testing testing testing"
        s._typer.reset_mock()
        sequence = [
            "Testing testing testing hello hello hello",
            "Testing, testing, testing, hello, hello, hello. Testing.",
            "Testing testing hello hello testing testing",
            "Testing, testing, testing! Hello Hello.",
        ]
        for text in sequence:
            s._apply_correction(text)
        s._typer.delete_chars.assert_not_called()


# --- Realistic Whisper sequences ---

class TestWhisperSequence:
    def _simulate(self, transcriptions: list[str]) -> tuple[list, str]:
        s = StreamingSession(
            transcriber=MagicMock(),
            typer=MagicMock(spec=TyperBackend),
            recorder=MagicMock(),
        )
        for text in transcriptions[:-1]:
            s._apply_correction(text)
        if transcriptions:
            s._apply_correction(transcriptions[-1], final=True)
        return s._typer.method_calls, s._typed_text

    def test_stable_growth(self):
        """Text grows steadily."""
        calls, final = self._simulate([
            "Hello",
            "Hello world",
            "Hello world how are you",
        ])
        assert final == "Hello world how are you"

    def test_punct_flipflop_keeps_growing(self):
        """Punctuation flip-flops don't stop text from growing."""
        calls, final = self._simulate([
            "Testing, testing.",
            "Testing testing testing hello.",
            "Testing, testing, testing, hello, how are you.",
            "Testing testing testing hello how are you today.",
        ])
        # Final should match last transcription
        assert final == "Testing testing testing hello how are you today."

    def test_final_corrects_drift(self):
        """Final pass fixes accumulated formatting drift."""
        calls, final = self._simulate([
            "Testing testing",
            "Testing, testing, hello",    # append "hello"
            "Testing testing hello world", # append "world"
            "Testing, testing, hello, world, goodbye.",  # final: corrects
        ])
        assert final == "Testing, testing, hello, world, goodbye."


# --- Fixture-based tests ---

def _load_fixture(name: str) -> dict | None:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_fixture_names() -> list[str]:
    if not FIXTURES_DIR.exists():
        return []
    return [p.stem for p in FIXTURES_DIR.glob("*.json")]


@pytest.mark.parametrize("fixture_name", _get_fixture_names() or ["__skip__"])
def test_fixture_no_large_streaming_deletions(fixture_name):
    """Fixture: streaming only deletes during short-text warmup (≤2 words)."""
    if fixture_name == "__skip__":
        pytest.skip("No fixtures. Run: python tests/record_fixture.py <name>")

    fixture = _load_fixture(fixture_name)
    transcriptions = [w["text"] for w in fixture["windows"] if w["text"]]
    if not transcriptions:
        pytest.skip("Silent recording")

    s = StreamingSession(
        transcriber=MagicMock(),
        typer=MagicMock(spec=TyperBackend),
        recorder=MagicMock(),
    )
    for text in transcriptions[:-1]:
        s._apply_correction(text)

    # Any streaming deletions should be small (short-text warmup only)
    for c in s._typer.delete_chars.call_args_list:
        assert c.args[0] <= 30, f"Large streaming deletion: {c.args[0]} chars"

    # Final pass
    s._apply_correction(transcriptions[-1], final=True)
    assert s._typed_text == transcriptions[-1]


@pytest.mark.parametrize("fixture_name", _get_fixture_names() or ["__skip__"])
def test_fixture_final_text_correct(fixture_name):
    """Fixture: final typed text matches Whisper's last output."""
    if fixture_name == "__skip__":
        pytest.skip("No fixtures. Run: python tests/record_fixture.py <name>")

    fixture = _load_fixture(fixture_name)
    transcriptions = [w["text"] for w in fixture["windows"] if w["text"]]
    if not transcriptions:
        pytest.skip("Silent recording")

    s = StreamingSession(
        transcriber=MagicMock(),
        typer=MagicMock(spec=TyperBackend),
        recorder=MagicMock(),
    )
    for text in transcriptions[:-1]:
        s._apply_correction(text)
    s._apply_correction(transcriptions[-1], final=True)

    assert s._typed_text == transcriptions[-1]


@pytest.mark.parametrize("fixture_name", _get_fixture_names() or ["__skip__"])
def test_fixture_text_grows(fixture_name):
    """Fixture: typed text grows over time (not stuck)."""
    if fixture_name == "__skip__":
        pytest.skip("No fixtures. Run: python tests/record_fixture.py <name>")

    fixture = _load_fixture(fixture_name)
    transcriptions = [w["text"] for w in fixture["windows"] if w["text"]]
    if len(transcriptions) < 3:
        pytest.skip("Too few transcriptions")

    s = StreamingSession(
        transcriber=MagicMock(),
        typer=MagicMock(spec=TyperBackend),
        recorder=MagicMock(),
    )
    lengths = []
    for text in transcriptions:
        s._apply_correction(text)
        lengths.append(len(s._typed_text))

    # Text should grow at some point (not all zeros after first)
    assert max(lengths) > lengths[0], "Text never grew beyond first transcription"


# --- Preedit path (StreamingTyper) ---

class _MockStreamingTyper:
    """Mock that satisfies StreamingTyper protocol."""
    name = "mock-ibus"

    def probe(self):
        pass

    def type_text(self, text):
        pass

    def delete_chars(self, n):
        pass

    def update_preedit(self, text):
        pass

    def commit_text(self, text):
        pass

    def clear_preedit(self):
        pass


class TestPreeditPath:
    def _make_session(self):
        typer = MagicMock(spec=_MockStreamingTyper)
        typer.name = "mock-ibus"
        # Make isinstance check work
        typer.update_preedit = MagicMock()
        typer.commit_text = MagicMock()
        typer.clear_preedit = MagicMock()
        typer.type_text = MagicMock()
        typer.delete_chars = MagicMock()
        return StreamingSession(
            transcriber=MagicMock(),
            typer=typer,
            recorder=MagicMock(),
        )

    def test_mock_is_streaming_typer(self):
        typer = _MockStreamingTyper()
        assert isinstance(typer, StreamingTyper)

    def test_initial_uses_preedit(self):
        s = self._make_session()
        s._apply_correction("Hello world")
        s._typer.update_preedit.assert_called_once_with("Hello world")
        s._typer.type_text.assert_not_called()
        assert s._typed_text == "Hello world"

    def test_update_replaces_preedit(self):
        s = self._make_session()
        s._apply_correction("Hello")
        s._apply_correction("Hello world")
        assert s._typer.update_preedit.call_count == 2
        s._typer.update_preedit.assert_called_with("Hello world")
        assert s._typed_text == "Hello world"

    def test_final_commits(self):
        s = self._make_session()
        s._apply_correction("Hello")
        s._apply_correction("Hello world", final=True)
        s._typer.commit_text.assert_called_once_with("Hello world")
        assert s._typed_text == "Hello world"

    def test_no_deletions_ever(self):
        s = self._make_session()
        for text in ["Hello", "Hello world", "Hello world how", "Different text entirely"]:
            s._apply_correction(text)
        s._typer.delete_chars.assert_not_called()

    def test_same_text_skipped(self):
        s = self._make_session()
        s._apply_correction("Hello")
        s._apply_correction("Hello")
        assert s._typer.update_preedit.call_count == 1

    def test_whisper_flipflop_trivial(self):
        """Preedit handles Whisper instability trivially - just replace text."""
        s = self._make_session()
        sequence = [
            "Testing, testing.",
            "Testing testing testing hello.",
            "Testing, testing, testing, hello, how are you.",
            "Testing testing testing hello how are you today.",
        ]
        for text in sequence[:-1]:
            s._apply_correction(text)
        s._apply_correction(sequence[-1], final=True)
        assert s._typed_text == sequence[-1]
        s._typer.delete_chars.assert_not_called()


# --- Worker loop integration ---

class TestWorkerLoop:
    def test_fires_on_event(self):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "Hello"
        typer = MagicMock(spec=TyperBackend)
        recorder = MagicMock()
        recorder.sample_rate = 16000
        recorder.get_audio_so_far.return_value = np.zeros(32000, dtype=np.float32)

        session = StreamingSession(transcriber, typer, recorder)
        session._worker_thread = threading.Thread(
            target=session._worker_loop, daemon=True,
        )
        session._worker_thread.start()

        session._pending.set()
        time.sleep(0.3)

        session._stop.set()
        session._pending.set()
        session._worker_thread.join(timeout=5)

        transcriber.transcribe.assert_called()
        typer.type_text.assert_called_with("Hello")

    def test_stop_does_final_transcription(self):
        transcriber = MagicMock()
        transcriber.transcribe.return_value = "Final text"
        typer = MagicMock(spec=TyperBackend)
        recorder = MagicMock()
        recorder.sample_rate = 16000
        recorder.get_audio_so_far.return_value = np.zeros(16000, dtype=np.float32)
        recorder.set_on_speech_pause = MagicMock()

        session = StreamingSession(transcriber, typer, recorder)
        session._worker_thread = threading.Thread(
            target=session._worker_loop, daemon=True,
        )
        session._worker_thread.start()

        result = session.stop()
        assert result == "Final text"

    def test_vad_pauses_collapse(self):
        call_count = 0
        transcriber = MagicMock()

        def slow_transcribe(audio):
            nonlocal call_count
            call_count += 1
            time.sleep(0.2)
            return f"text {call_count}"

        transcriber.transcribe.side_effect = slow_transcribe
        typer = MagicMock(spec=TyperBackend)
        recorder = MagicMock()
        recorder.sample_rate = 16000
        recorder.get_audio_so_far.return_value = np.zeros(32000, dtype=np.float32)
        recorder.set_on_speech_pause = MagicMock()

        session = StreamingSession(transcriber, typer, recorder)
        session._worker_thread = threading.Thread(
            target=session._worker_loop, daemon=True,
        )
        session._worker_thread.start()

        for _ in range(5):
            session._pending.set()
            time.sleep(0.02)

        time.sleep(1.0)
        session._stop.set()
        session._pending.set()
        session._worker_thread.join(timeout=5)

        assert transcriber.transcribe.call_count < 5
