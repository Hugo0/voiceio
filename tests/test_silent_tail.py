"""A stop tail (audio.input_latency_ms, hotkey.release_tail_ms) records
after the stop press. When nothing more was said it is silence, and given
silence Whisper writes its context prompt back or a stock phrase ("Thank you
for watching."). So the tail is decoded only up to its speech; everything
before the press decodes exactly as in 1.18.18."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from fakes import FakeDecoder
from test_sherpa import FIXED, FakeSherpa, _model_dir

from voiceio.audio_source import SPEECH_TAIL_PAD_SECS, SpeechSpans
from voiceio.config import AudioConfig
from voiceio.core.decoders import sherpa
from voiceio.recorder import AudioRecorder
from voiceio.streaming import StreamingSession
from voiceio.typers.base import TyperBackend

SR = 16000
BLOCK = 1600          # 0.1 s: one audio callback
WORD_BLOCKS = 5       # a "word" is 0.5 s of one constant level
STOCK = "Thank you for watching."
PAD = int(SPEECH_TAIL_PAD_SECS * SR)


def _words(*ids: int) -> np.ndarray:
    """Speech: word n is 0.5 s at level n/100 (n >= 10)."""
    return np.concatenate([np.full(WORD_BLOCKS * BLOCK, n / 100, np.float32) for n in ids])


def _quiet(secs: float) -> np.ndarray:
    """A real room: low noise, not digital zeros."""
    rng = np.random.default_rng(int(secs * 1000))
    return (rng.standard_normal(int(secs * SR)) * 1e-3).astype(np.float32)


def whisper_like(audio, final=False, context=None) -> str:
    """Reads the words back like Whisper, including its silence failures:
    no speech at all → the prompt's last sentence (or a stock phrase), and a
    second or more of trailing silence → a stock phrase after the words."""
    blocks = [float(np.mean(audio[i:i + BLOCK])) for i in range(0, len(audio), BLOCK)]
    words, last_loud = [], -1
    for i, v in enumerate(blocks):
        if abs(v) > 0.05:
            w = f"w{round(v * 100)}"
            if not words or words[-1] != w or i != last_loud + 1:
                words.append(w)
            last_loud = i
    if not words:
        return context.split(". ")[-1] if context else STOCK
    if len(blocks) - 1 - last_loud >= 10:
        words.append(STOCK)
    return " ".join(words)


class ThresholdVad:
    def reset(self) -> None:
        pass

    def is_speech(self, chunk) -> bool:
        return float(np.max(np.abs(chunk))) > 0.05


def _rig(freeze_secs=8.0, decoder=None):
    """A real AudioRecorder (VAD + pause detection, no device) feeding a
    session whose passes the test runs by hand."""
    rec = AudioRecorder(AudioConfig(sample_rate=SR, prebuffer_secs=0.0,
                                    silence_duration=0.6), vad=ThresholdVad())
    rec._stream = MagicMock()
    if decoder is None:
        decoder = FakeDecoder()
        decoder.transcribe.side_effect = whisper_like
    session = StreamingSession(decoder=decoder, typer=MagicMock(spec=TyperBackend),
                               recorder=rec, freeze_secs=freeze_secs)
    rec.start()
    rec.set_on_speech_pause(session._on_vad_pause)
    return rec, session, decoder


def _feed(rec, audio, session=None):
    """Deliver audio in 0.1 s callbacks; with a session, run a pass (what
    the worker does every second or at a pause) after every callback."""
    for i in range(0, len(audio), BLOCK):
        chunk = audio[i:i + BLOCK].reshape(-1, 1)
        rec._callback(chunk, len(chunk), None, None)
        if session is not None:
            session._transcribe_and_apply()


def _stop(rec, session):
    audio = rec.stop()
    return session.stop(audio, rec.speech_spans()), audio


def _decoded(dec) -> list[np.ndarray]:
    return [c.args[0] for c in dec.transcribe.call_args_list]


# Toggle: the tail is input_latency_ms; hold: release_tail_ms + latency.
TAILS = {"toggle": 2.0, "hold": 0.25 + 2.0}
TEN = tuple(range(20, 30))                     # 5 s of speech: freezes at its pause
TEN_TEXT = " ".join(f"w{n}" for n in TEN)


def test_the_fake_reproduces_whisper_on_silence():
    assert whisper_like(_quiet(2), context="Testing. Can you hear me") == "Can you hear me"
    assert whisper_like(np.concatenate([_words(20, 21), _quiet(2)])) == f"w20 w21 {STOCK}"


@pytest.mark.parametrize("mode", TAILS)
def test_silent_tail_after_speech_decodes_to_the_speech_only(mode):
    rec, s, dec = _rig()
    speech = _words(20, 21, 22, 23)            # 2 s: too short to freeze
    _feed(rec, speech)
    rec.mark_stop_press()
    _feed(rec, _quiet(TAILS[mode]))
    _, audio = _stop(rec, s)
    assert s.raw_final_text == "w20 w21 w22 w23"
    # The final decode: everything before the press, plus the pad.
    assert len(_decoded(dec)[-1]) == len(speech) + PAD
    assert len(audio) > len(speech) + SR


@pytest.mark.parametrize("mode", TAILS)
def test_silent_tail_after_a_freeze_is_not_decoded(mode):
    """The echo: the pause freeze lands in the tail, leaving the stop-time
    pass only silence, conditioned on the frozen text."""
    rec, s, dec = _rig()
    _feed(rec, _words(*TEN))
    rec.mark_stop_press()
    _feed(rec, _quiet(TAILS[mode]), session=s)   # pause → freeze, then interims
    assert s._frozen_raw == TEN_TEXT
    after = s.trace[[p["kind"] for p in s.trace].index("freeze") + 1:]
    assert after and all(p["text"] == "" and p["voiced_secs"] == 0 for p in after)
    assert s.interim_text == TEN_TEXT
    decodes = dec.transcribe.call_count
    _stop(rec, s)
    assert dec.transcribe.call_count == decodes       # the silent tail: no decode
    assert s.raw_final_text == TEN_TEXT


def test_interims_in_the_tail_show_no_hallucination():
    rec, s, dec = _rig()
    shown = []
    s._on_interim = shown.append
    _feed(rec, _words(20, 21), session=s)
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0), session=s)
    assert all(STOCK not in t for t in shown)
    assert all(STOCK not in p["text"] for p in s.trace)


@pytest.mark.parametrize("freeze_first", [False, True])
def test_speech_arriving_inside_the_tail_is_kept(freeze_first):
    """What the tail is for: the last word lands 1.5 s after the stop press."""
    rec, s, dec = _rig()
    ids = TEN if freeze_first else (20, 21, 22, 23)
    _feed(rec, _words(*ids), session=s)
    rec.mark_stop_press()
    _feed(rec, np.concatenate([_quiet(1.5), _words(40), _quiet(0.5)]), session=s)
    assert bool(s._frozen_samples) == freeze_first
    _stop(rec, s)
    assert s.raw_final_text == " ".join(f"w{n}" for n in ids) + " w40"


def test_trailing_pad_follows_the_last_speech_in_the_tail():
    rec, s, dec = _rig(freeze_secs=0)
    before = np.concatenate([_words(20), _quiet(0.2)])
    _feed(rec, before)
    rec.mark_stop_press()
    _feed(rec, np.concatenate([_words(21), _quiet(2.0)]))
    _stop(rec, s)
    decoded = _decoded(dec)[-1]
    end = len(before) + len(_words(21))
    assert len(decoded) == end + PAD
    assert np.array_equal(decoded[:end], np.concatenate([before, _words(21)]))


def test_audio_before_the_press_is_never_trimmed():
    """Silence the user left before pressing stop decodes as it always did."""
    rec, s, dec = _rig(freeze_secs=0)
    before = np.concatenate([_words(20, 21), _quiet(1.5)])
    _feed(rec, before)
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0))
    _stop(rec, s)
    assert np.array_equal(_decoded(dec)[-1], before)


def test_a_recording_the_vad_heard_no_speech_in_is_decoded_whole():
    rec, s, dec = _rig()
    _feed(rec, _quiet(1.0))
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0))
    _, audio = _stop(rec, s)
    assert np.array_equal(_decoded(dec)[-1], audio)


def test_withdrawn_stop_press_trims_nothing():
    rec, s, dec = _rig(freeze_secs=0)
    _feed(rec, _words(20))
    rec.mark_stop_press()
    rec.mark_stop_press(False)                 # a press during the tail resumed
    _feed(rec, _quiet(2.0))
    _, audio = _stop(rec, s)
    assert np.array_equal(_decoded(dec)[-1], audio)


def test_voiced_end_rules():
    spans = SpeechSpans(SR, unchecked=SR)        # 1 s pre-buffer
    spans.mark(SR, 2 * SR, False)
    spans.mark(2 * SR, 3 * SR, True)
    spans.mark(3 * SR, 6 * SR, False)
    assert spans.voiced_end(0, 6 * SR) == 6 * SR        # no press: untouched
    spans.press = 4 * SR
    assert spans.voiced_end(0, 6 * SR) == 4 * SR        # silent tail: cut at the press
    assert spans.voiced_end(0, 4 * SR + PAD) == 4 * SR + PAD   # tail <= pad: whole
    assert spans.voiced_end(5 * SR, 6 * SR) == 5 * SR   # silent span in the tail: nothing
    assert spans.voiced_end(0, 7 * SR) == 7 * SR        # 6..7 s not judged yet: kept
    spans.press = 3 * SR - PAD // 2                     # speech ran just past the press
    assert spans.voiced_end(0, 6 * SR) == 3 * SR + PAD
    spans.press = 3 * SR + PAD // 2                     # ...or ended just before it
    assert spans.voiced_end(0, 6 * SR) == 3 * SR + PAD


# ── unchanged from 1.18.18 without a long tail ───────────────────────────

def _decodes_with_and_without_spans(script):
    """Every decode input of `script`, run through the new code and through
    the 1.18.18 path (no speech spans anywhere)."""
    runs = []
    for spans in (True, False):
        rec, s, dec = _rig()
        if not spans:
            rec.speech_spans = lambda: None
        script(rec, s)
        audio = rec.stop()
        s.stop(audio, rec.speech_spans())
        runs.append((_decoded(dec), s.raw_final_text))
    return runs


def _assert_identical(runs):
    (new, new_text), (old, old_text) = runs
    assert new_text == old_text
    assert len(new) == len(old) > 1
    assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(new, old))


def test_toggle_without_latency_decodes_byte_identical():
    """input_latency_ms = 0: the stop is immediate, no press is marked."""
    def script(rec, s):
        _feed(rec, _words(*TEN), session=s)
        _feed(rec, _quiet(1.0), session=s)       # a pause freeze...
        _feed(rec, np.concatenate([_words(40, 41), _quiet(0.5)]), session=s)
    _assert_identical(_decodes_with_and_without_spans(script))


def test_hold_with_the_default_release_tail_decodes_byte_identical():
    """hotkey.release_tail_ms = 250, no latency: shorter than the pad."""
    def script(rec, s):
        _feed(rec, _words(*TEN), session=s)
        _feed(rec, _quiet(1.0), session=s)
        _feed(rec, _words(40, 41), session=s)
        _feed(rec, _quiet(0.5), session=s)       # paused before letting go
        rec.mark_stop_press()
        _feed(rec, _quiet(0.25), session=s)
    _assert_identical(_decodes_with_and_without_spans(script))


# ── every decode path gets the same trimming ─────────────────────────────

def test_parallel_tail_decode_is_trimmed_too():
    """A freeze still decoding at stop: the tail after its cut decodes on the
    draft lane, with the same trimming."""
    rec, s, dec = _rig()
    dec.draft = FakeDecoder()
    dec.draft.transcribe.side_effect = whisper_like
    speech = _words(20, 21)
    _feed(rec, speech)
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0))
    s._freeze_pending_end = 0
    audio = rec.stop()
    s._final_speech = rec.speech_spans()
    pre = s._decode_past_pending_freeze(audio)
    assert pre[1].text == "w20 w21"
    assert len(dec.draft.transcribe.call_args.args[0]) == len(speech) + PAD


def _parakeet_rig(monkeypatch, tmp_path):
    fake = FakeSherpa(text="hello")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake.module)
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    rec, s, _ = _rig(decoder=sherpa.SherpaTransducer(_model_dir(tmp_path)))
    return fake, rec, s


def test_parakeet_decodes_only_the_speech(monkeypatch, tmp_path):
    fake, rec, s = _parakeet_rig(monkeypatch, tmp_path)
    speech = _words(20, 21, 22)
    _feed(rec, speech)
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0))
    fake.streams.clear()
    _stop(rec, s)
    assert [st["samples"] for st in fake.streams] == [len(speech) + PAD]
    assert s.raw_final_text == "hello"


def test_parakeet_never_sees_a_silent_tail_after_a_freeze(monkeypatch, tmp_path):
    fake, rec, s = _parakeet_rig(monkeypatch, tmp_path)
    _feed(rec, _words(*TEN))
    rec.mark_stop_press()
    _feed(rec, _quiet(2.0), session=s)
    assert s._frozen_raw == "hello"
    fake.streams.clear()
    _stop(rec, s)
    assert fake.streams == []
    assert s.raw_final_text == "hello"


# ── the daemon marks the press and hands the spans to both final paths ───

def test_a_stop_with_a_tail_marks_the_press_and_a_resume_withdraws_it():
    from test_input_latency import _app, _start_then_stop

    from voiceio.app import _State
    vio = _app("toggle", latency_ms=5000)
    _start_then_stop(vio)
    assert vio._state == _State.RECORDING
    vio.recorder.mark_stop_press.assert_called_once_with()
    vio._resume()
    vio.recorder.mark_stop_press.assert_called_with(False)


def test_an_immediate_stop_marks_no_press():
    from test_input_latency import _app, _start_then_stop
    vio = _app("toggle", latency_ms=0)
    _start_then_stop(vio)
    vio.recorder.mark_stop_press.assert_not_called()


def test_stop_passes_the_speech_spans_to_the_finalizer():
    from test_cancel import _make_app

    from voiceio.app import _State
    vio = _make_app()
    spans = SpeechSpans(SR)
    vio.recorder.speech_spans.return_value = spans
    vio.recorder.stop.return_value = np.zeros(SR, np.float32)
    vio._state, vio._session, vio._streaming = _State.RECORDING, MagicMock(), True
    calls = []
    vio._output_thread = lambda target, *args: calls.append(args)
    vio._do_stop()
    assert calls[0][-1] is spans


def test_batch_decode_is_trimmed():
    from test_cancel import _make_app
    vio = _make_app()
    vio._retention_extra = lambda audio: {}
    vio.engine.transcribe.return_value = MagicMock(text="", abort=False)
    spans = SpeechSpans(SR)
    spans.mark(0, SR, True)
    spans.mark(SR, 3 * SR, False)
    spans.press = SR
    vio._process(np.zeros(3 * SR, np.float32), vio._generation, speech=spans)
    assert len(vio.engine.transcribe.call_args.args[0]) == SR + PAD
