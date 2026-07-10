"""Streaming transcription with word-level append and final correction."""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from voiceio.transcriber import transcribe_timeout
from voiceio.typers.base import StreamingTyper

if TYPE_CHECKING:
    from voiceio.commands import CommandProcessor
    from voiceio.corrections import CorrectionDict
    from voiceio.llm import LLMProcessor
    from voiceio.postcorrect import PostCorrector
    from voiceio.recorder import AudioRecorder
    from voiceio.transcriber import Transcriber
    from voiceio.typers.base import TyperBackend

log = logging.getLogger(__name__)
DELETE_SETTLE_SECS = 0.05  # delay between delete and type for ydotool reliability

# Incremental finalization: a freeze cut is only safe at a speech pause, so
# require the tail's trailing window to be quiet before cutting there.
_FREEZE_SILENCE_WINDOW_SECS = 0.3
# Quiet is judged RELATIVE to the tail's own level: mic gains vary wildly
# and the decode path normalizes audio anyway, so an absolute threshold
# would read a quiet mic's speech as silence (mid-word cuts) and a hot
# noise floor as speech (freeze never fires).
# Calibrated on real recordings: between-words noise floor sits at
# ~0.27-0.38 of overall tail RMS, speech windows at >=0.9. 0.6 separates
# both with margin; 0.15 never fired on a mic with a normal noise floor.
_FREEZE_SILENCE_RATIO = 0.6
_FREEZE_SILENCE_FLOOR = 1e-4  # digital silence is always quiet
# Whisper conditions on ~224 prompt tokens; more frozen context is wasted.
_FREEZE_CONTEXT_CHARS = 400


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _tail_ends_in_silence(tail: np.ndarray, sample_rate: int) -> bool:
    """True when the last _FREEZE_SILENCE_WINDOW_SECS of audio are quiet
    relative to the tail's overall level."""
    window = tail[-int(_FREEZE_SILENCE_WINDOW_SECS * sample_rate):]
    if len(window) == 0:
        return False
    threshold = max(_FREEZE_SILENCE_RATIO * _rms(tail), _FREEZE_SILENCE_FLOOR)
    return _rms(window) < threshold


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _join_text(a: str, b: str) -> str:
    """Join two transcript fragments with a single space."""
    a, b = a.strip(), (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    return a + " " + b


def _clean_word(w: str) -> str:
    """Strip punctuation for fuzzy word matching."""
    return re.sub(r'[^\w]', '', w).lower()


def _word_match_len(old_words: list[str], new_words: list[str]) -> int:
    """Count matching leading words, ignoring punctuation/case."""
    count = 0
    for o, n in zip(old_words, new_words):
        if _clean_word(o) == _clean_word(n):
            count += 1
        else:
            break
    return count


_TYPER_FAIL_THRESHOLD = 3  # consecutive failures before signalling re-probe


class StreamingSession:
    """Manages one streaming transcription cycle.

    During streaming: append-only with word-level fuzzy matching.
    Whisper changes punctuation/capitalization between passes, so word-level
    matching ignores these, so text keeps growing even when Whisper
    flip-flops on commas vs periods.

    On stop: one final char-level diff correction to fix accumulated drift.

    The session holds a reference to the recorder only during active
    recording. On stop(), the caller passes an audio snapshot and the
    session releases the recorder reference.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        typer: TyperBackend,
        recorder: AudioRecorder,
        generation: int = 0,
        cleanup: bool = False,
        number_conversion: bool = False,
        language: str = "en",
        commands: CommandProcessor | None = None,
        corrections: CorrectionDict | None = None,
        postcorrect: PostCorrector | None = None,
        llm: LLMProcessor | None = None,
        voice_input_prefix: str = "",
        on_typer_broken: Callable[[], None] | None = None,
        is_current: Callable[[], bool] | None = None,
        on_interim: Callable[[str], None] | None = None,
        freeze_secs: float = 25.0,
    ):
        self._transcriber = transcriber
        self._typer = typer
        # Output-ownership gate: the app sets this so a superseded session (its
        # generation was overtaken by a newer recording) stops emitting ANY
        # typer output on the final path, which would otherwise corrupt the new
        # session's text on non-IBus typers. Defaults to "always current".
        self._is_current = is_current or (lambda: True)
        self._recorder: AudioRecorder | None = recorder
        self._sample_rate = recorder.sample_rate
        self._generation = generation
        self._cleanup = cleanup
        self._number_conversion = number_conversion
        self._language = language
        self._commands = commands
        self._corrections = corrections
        self._postcorrect = postcorrect
        self._llm = llm
        self._voice_input_prefix = voice_input_prefix
        self._on_typer_broken = on_typer_broken
        self._on_interim = on_interim
        self._typer_fail_count = 0
        self._typer_broken_signalled = False
        self._typed_text = ""
        self._pending = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._final_audio: np.ndarray | None = None  # set on stop
        # Incremental finalization: audio before _frozen_samples has already
        # been beam-decoded into _frozen_raw and is never decoded again.
        # Interim and final passes only decode the tail after this offset.
        self._freeze_secs = freeze_secs
        self._frozen_samples = 0
        self._frozen_raw = ""
        self._frozen_segments: list[dict] = []
        # Decode backpressure: when the CPU is contended, decodes slow down
        # and 1s-tick interim passes would queue back-to-back, each covering
        # a longer tail — a starvation spiral. Require at least as much NEW
        # audio as the last decode took before starting another interim.
        self._last_decode_end = 0     # samples covered by the last decode
        self._last_decode_secs = 0.0  # how long that decode took
        # Raw (pre-pipeline) text + confidence of the final pass, for history
        self.raw_final_text: str | None = None
        self.final_latency: dict = {}
        self.final_segments: list[dict] = []
        # Per-pass decode trace (interim/freeze/final) for debugging,
        # profiling, and future training. Persisted via retention.save_trace.
        self.trace: list[dict] = []
        self._t0 = time.monotonic()

    @property
    def interim_text(self) -> str:
        """Best-so-far text (post-pipeline). Safe to read while finalizing."""
        return self._typed_text

    def set_is_current(self, is_current: Callable[[], bool]) -> None:
        """Install the output-ownership gate (see __init__).

        Called by the app at stop time, once it knows the generation this
        finalizer owns, so the final path can bail out of typer output the
        moment a newer recording supersedes this one.
        """
        self._is_current = is_current

    def set_typer(self, typer: TyperBackend) -> None:
        """Swap the output backend mid-session (used when the user switches
        the input source away from voiceio and we fall back to clipboard).

        Resets the typed-text tracking: the old backend's on-screen state
        (e.g. a stranded IBus preedit) no longer belongs to us.
        """
        self._typer = typer
        self._typed_text = ""

    def start(self) -> None:
        """Begin streaming. Recorder must already be started by caller."""
        self._recorder.set_on_speech_pause(self._on_vad_pause)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True,
        )
        self._worker_thread.start()
        log.debug("Streaming session started (gen=%d)", self._generation)

    def stop(self, audio: np.ndarray | None = None) -> str:
        """Stop streaming, run final transcription, return full text.

        Args:
            audio: Final audio snapshot from recorder.stop(). The session
                   uses this for the final transcription instead of reading
                   from the recorder (which may have been restarted).
        """
        self._final_audio = audio
        self._stop_event.set()
        self._pending.set()  # wake worker for final pass
        if self._worker_thread is not None:
            # Scale the join to the final-pass audio length so a long dictation
            # isn't abandoned before its (longer) decode finishes.
            dur = len(audio) / self._sample_rate if audio is not None else 0.0
            self._worker_thread.join(timeout=transcribe_timeout(dur) + 2)
            if self._worker_thread.is_alive():
                log.warning("Streaming worker did not exit in time (gen=%d)", self._generation)
        # Release recorder reference — session must not touch it after stop
        if self._recorder is not None:
            self._recorder.set_on_speech_pause(None)
            self._recorder = None
        log.debug("Streaming session stopped, typed: '%s'", self._typed_text)
        return self._typed_text

    def _on_vad_pause(self) -> None:
        """Called from audio thread on speech pause. Signals worker."""
        self._pending.set()

    def _worker_loop(self) -> None:
        """Worker thread: wake on Event, transcribe, apply diff."""
        while not self._stop_event.is_set():
            self._pending.clear()
            self._pending.wait(timeout=1.0)
            if self._stop_event.is_set():
                break
            try:
                self._transcribe_and_apply()
            except subprocess.CalledProcessError:
                self._typer_fail_count += 1
                if (self._typer_fail_count >= _TYPER_FAIL_THRESHOLD
                        and not self._typer_broken_signalled):
                    self._typer_broken_signalled = True
                    log.warning(
                        "Typer '%s' failed %d times in streaming, requesting re-probe",
                        self._typer.name, self._typer_fail_count,
                    )
                    if self._on_typer_broken:
                        self._on_typer_broken()
                elif self._typer_fail_count < _TYPER_FAIL_THRESHOLD:
                    log.exception("Streaming typer error (%d/%d)",
                                  self._typer_fail_count, _TYPER_FAIL_THRESHOLD)
            except Exception:
                log.exception("Streaming transcribe/apply error (non-fatal)")

        # Final transcription on stop using the audio snapshot
        try:
            self._transcribe_and_apply(min_seconds=0.5, final=True)
        except Exception:
            log.exception("Final transcribe/apply error")
        self._final_audio = None  # release memory

    def _frozen_context(self) -> str | None:
        """Frozen-text suffix used to condition the tail decode."""
        if not self._frozen_raw:
            return None
        return self._frozen_raw[-_FREEZE_CONTEXT_CHARS:]

    def _transcribe_and_apply(
        self, min_seconds: float = 1.0, final: bool = False,
    ) -> None:
        """Get audio, transcribe the un-frozen tail, apply correction.

        Incremental finalization: audio before _frozen_samples was already
        beam-decoded and frozen, so every pass — including the final one —
        only decodes the tail. Long dictations finalize in O(tail), not
        O(recording).
        """
        if final and self._final_audio is not None:
            # Use the snapshot passed to stop() — recorder may be gone
            audio = self._final_audio
        elif self._recorder is not None:
            audio = self._recorder.get_audio_so_far()
        else:
            return  # recorder released, nothing to do

        if audio is None:
            return
        if len(audio) < self._sample_rate * min_seconds:
            return

        tail = audio[self._frozen_samples:]
        if not final and len(tail) < self._sample_rate * min_seconds:
            return  # not enough new audio since the last freeze
        grown_secs = (len(audio) - self._last_decode_end) / self._sample_rate
        if not final and grown_secs < max(min_seconds, self._last_decode_secs):
            return  # backpressure: don't decode faster than we can keep up

        # Freeze when the tail has grown long AND ends at a speech pause
        # (never cut mid-word). The freeze pass gets beam search because its
        # text is never decoded again.
        freeze = (
            not final
            and self._freeze_secs > 0
            and len(tail) >= self._freeze_secs * self._sample_rate
            and _tail_ends_in_silence(tail, self._sample_rate)
        )

        tail_text = ""
        tail_segments: list[dict] = []
        t0 = time.monotonic()
        # The final pass decodes ANY remaining audio (a clipped last word
        # must not be dropped); interim/freeze passes skip sub-0.3s tails.
        if len(tail) > (0 if final else self._sample_rate * 0.3):
            try:
                tail_text = self._transcriber.transcribe(
                    tail, final=final or freeze, context=self._frozen_context(),
                )
            except Exception:
                # Decode FAILED (timeout/crash) — this is not silence. Never
                # advance the freeze boundary or char-diff against a partial
                # transcript; on final, commit what is already on screen.
                log.exception("Streaming transcription failed")
                if final:
                    self.raw_final_text = ""
                    self.final_latency = {
                        "audio_secs": round(len(audio) / self._sample_rate, 2),
                        "frozen_secs": round(self._frozen_samples / self._sample_rate, 2),
                        "tail_secs": round(len(tail) / self._sample_rate, 2),
                        "transcribe": round(time.monotonic() - t0, 3),
                        "decode_failed": True,
                    }
                    self._commit_interim_on_final()
                return
            tail_segments = list(getattr(self._transcriber, "last_segments", []))
        t_transcribe = time.monotonic() - t0
        self._last_decode_end = len(audio)
        self._last_decode_secs = t_transcribe

        self.trace.append({
            "t": round(time.monotonic() - self._t0, 2),
            "kind": "final" if final else ("freeze" if freeze else "interim"),
            "frozen_secs": round(self._frozen_samples / self._sample_rate, 2),
            "tail_secs": round(len(tail) / self._sample_rate, 2),
            "secs": round(t_transcribe, 3),
            "text": tail_text,
        })

        if freeze:
            self._frozen_raw = _join_text(self._frozen_raw, tail_text)
            self._frozen_samples = len(audio)
            self._frozen_segments.extend(tail_segments)
            log.debug(
                "Froze %.1fs of audio (frozen text now %d chars)",
                len(tail) / self._sample_rate, len(self._frozen_raw),
            )
        raw = self._frozen_raw if freeze else _join_text(self._frozen_raw, tail_text)

        # A slow interim decode can complete AFTER the user pressed stop
        # (observed: 29s under CPU contention). Applying it would overwrite
        # the preedit with stale partial text moments before the final
        # commit rewrites it again — discard it. Freeze bookkeeping above
        # is kept: it is beam-quality and shrinks the final tail.
        if not final and self._stop_event.is_set():
            return
        # An interim pass that decoded nothing must not touch the display:
        # rewriting to frozen-only text would visibly delete the un-frozen
        # words the previous pass just showed (transient decoder flake).
        if not final and not tail_text:
            return

        if final:
            self.raw_final_text = raw
            self.final_segments = self._frozen_segments + tail_segments
            self.final_latency = {
                "audio_secs": round(len(audio) / self._sample_rate, 2),
                "frozen_secs": round(self._frozen_samples / self._sample_rate, 2),
                "tail_secs": round(len(tail) / self._sample_rate, 2),
                "transcribe": round(t_transcribe, 3),
            }
            if not raw:
                # The final pass produced nothing. If we already have interim
                # text (e.g. the final transcription timed out on a long
                # dictation), commit that instead of silently dropping it —
                # the audio WAV is retained separately, so no extra save here.
                self._commit_interim_on_final()
                return

        if raw:
            from voiceio.postprocess import apply_pipeline
            t1 = time.monotonic()
            text, abort = apply_pipeline(
                raw,
                do_cleanup=self._cleanup,
                number_conversion=self._number_conversion,
                language=self._language,
                commands=self._commands,
                corrections=self._corrections,
                postcorrect=self._postcorrect,
                llm=self._llm,
                voice_input_prefix=self._voice_input_prefix,
                final=final,
            )
            if final:
                self.final_latency["pipeline"] = round(time.monotonic() - t1, 3)
                pc_secs = getattr(self._postcorrect, "last_secs", None)
                if pc_secs is not None:
                    self.final_latency["postcorrect"] = round(pc_secs, 3)
            if abort:
                # A voice command wiped the utterance — discard frozen text
                # too, or the next pass would resurrect it.
                self._frozen_raw = ""
                self._frozen_segments = []
                self._frozen_samples = len(audio)
                # A superseded session must not touch the typer on the final
                # path (would corrupt the newer session's output).
                if final and not self._is_current():
                    self._typed_text = ""
                    return
                if isinstance(self._typer, StreamingTyper):
                    self._typer.clear_preedit()
                elif self._typed_text:
                    self._typer.delete_chars(len(self._typed_text))
                self._typed_text = ""
                return

            if text:
                prev = self._typed_text
                self._apply_correction(text, final=final)
                if not final and self._on_interim and self._typed_text != prev:
                    try:
                        self._on_interim(self._typed_text)
                    except Exception:
                        log.debug("on_interim callback failed", exc_info=True)

    def _commit_interim_on_final(self) -> None:
        """Commit whatever interim text we have when the final pass yields none.

        For the IBus preedit path the interim text lives only in the preedit
        (uncommitted) — commit it so a final-pass timeout doesn't drop it. For
        keystroke typers the interim text is already in the target app, so
        there is nothing to do.
        """
        if not self._typed_text or not self._is_current():
            return
        if isinstance(self._typer, StreamingTyper):
            log.warning(
                "Final transcription empty; committing interim preedit text (%d chars)",
                len(self._typed_text),
            )
            self._typer.commit_text(self._typed_text)

    def _apply_correction(self, new_text: str, final: bool = False) -> None:
        """Apply correction to typed text.

        With StreamingTyper (IBus): use preedit during streaming, commit on final.
        Without: append-only via word-level matching, char-level diff on final.
        """
        old = self._typed_text

        # A superseded session must emit no typer output on the final path.
        if final and not self._is_current():
            log.debug("Final output suppressed: session superseded")
            self._typed_text = new_text
            return

        # Preedit path: trivial, just replace the preview text
        if isinstance(self._typer, StreamingTyper):
            if final:
                self._typer.commit_text(new_text)
                self._typed_text = new_text
                log.debug("Preedit commit: '%s'", new_text[:60])
            elif new_text != old:
                self._typer.update_preedit(new_text)
                self._typed_text = new_text
                log.debug("Preedit update: '%s'", new_text[:60])
            return

        if new_text == old:
            return

        if final:
            self._apply_final_correction(new_text)
            return

        if not old:
            # First transcription, just type it
            self._typer.type_text(new_text)
            self._typed_text = new_text
            log.debug("Initial: '%s'", new_text)
            return

        # Short text: allow full replacement (Whisper still warming up)
        if len(old.split()) <= 2:
            self._typer.delete_chars(len(old))
            time.sleep(DELETE_SETTLE_SECS)
            self._typer.type_text(new_text)
            self._typed_text = new_text
            log.debug("Replaced short text: '%s'", new_text)
            return

        # Word-level append-only
        old_words = old.split()
        new_words = new_text.split()
        matched = _word_match_len(old_words, new_words)

        if matched >= len(old_words) and len(new_words) > matched:
            new_tail = " ".join(new_words[matched:])
            to_type = " " + new_tail
            self._typer.type_text(to_type)
            self._typed_text = old + to_type
            log.debug("Appended: '%s'", to_type.strip())
        else:
            log.debug(
                "Skipping (matched %d/%d words)",
                matched, len(old_words),
            )

    def _apply_final_correction(self, new_text: str) -> None:
        """Final pass: char-level diff to fix accumulated drift."""
        old = self._typed_text
        prefix_len = _common_prefix_len(old, new_text)
        to_delete = len(old) - prefix_len
        to_type = new_text[prefix_len:]

        if to_delete > 0:
            log.debug("Final correction: delete %d, type '%s'", to_delete, to_type)
            self._typer.delete_chars(to_delete)
            time.sleep(DELETE_SETTLE_SECS)
        if to_type:
            self._typer.type_text(to_type)

        self._typed_text = new_text
