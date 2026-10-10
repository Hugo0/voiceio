"""Streaming transcription with word-level append and final correction."""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from voiceio.core.decoders.base import Bias
from voiceio.transcriber import describe_transcript, transcribe_timeout
from voiceio.typers.base import StreamingTyper

if TYPE_CHECKING:
    from voiceio.audio_source import AudioSource
    from voiceio.core.decoders.base import Decoder
    from voiceio.core.pipeline import Pipeline
    from voiceio.typers.base import TyperBackend

log = logging.getLogger(__name__)
DELETE_SETTLE_SECS = 0.05  # delay between delete and type for ydotool reliability

# Incremental finalization: a freeze cut is only safe at a speech pause.
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
# How far past freeze_secs to look for a pause. Real speech has a usable
# pause every ~5s (measured: longest pause-free stretch 5.6s in a 3min
# "continuous" note), so 8s of search span almost always finds one. When it
# doesn't, the tail is cut at its quietest window anyway: a cut at a soft
# spot beats re-decoding an ever-growing tail at stop (observed: 17 long
# dictations never froze, one decoding 121s of audio after stop).
_FREEZE_SEARCH_SECS = 8.0
# Whisper conditions on ~224 prompt tokens; more frozen context is wasted.
_FREEZE_CONTEXT_CHARS = 400


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _find_freeze_cut(
    tail: np.ndarray, sample_rate: int, min_secs: float, *, force: bool = False,
) -> int | None:
    """Sample index of the best freeze cut in `tail`, or None.

    Searches tail[min_secs : min_secs+_FREEZE_SEARCH_SECS] for the quietest
    300ms window below the relative silence threshold and cuts at that
    window's end. Searching the interior (rather than requiring the pass to
    land on a trailing pause) is what makes freezing actually fire on real
    speech: a pass's trailing 300ms is almost never a pause, but the
    interior of a 15-35s span almost always contains one.
    """
    win = int(_FREEZE_SILENCE_WINDOW_SECS * sample_rate)
    start = int(min_secs * sample_rate)
    end = min(len(tail), int((min_secs + _FREEZE_SEARCH_SECS) * sample_rate))
    if end - start < win:
        return None
    threshold = max(_FREEZE_SILENCE_RATIO * _rms(tail[:end]), _FREEZE_SILENCE_FLOOR)
    best_cut, best_rms = None, float("inf") if force else threshold
    for i in range(start, end - win + 1, win // 2):
        window_rms = _rms(tail[i:i + win])
        if window_rms < best_rms:
            best_rms = window_rms
            best_cut = i + win
    return best_cut


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

    `recorder` is any `voiceio.audio_source.AudioSource` — the desktop's
    microphone `AudioRecorder`, or a `PushAudioSource` fed from a socket. The
    session pulls (`get_audio_so_far()`) rather than being pushed to, so a
    producer that stalls delays interim text without stalling the session.
    The parameter keeps its historical name for callers passing positionally.
    """

    def __init__(
        self,
        transcriber: Decoder,
        typer: TyperBackend,
        recorder: AudioSource,
        generation: int = 0,
        pipeline: Pipeline | None = None,
        on_typer_broken: Callable[[], None] | None = None,
        is_current: Callable[[], bool] | None = None,
        on_interim: Callable[[str], None] | None = None,
        freeze_secs: float = 15.0,
        bias: Bias | None = None,
    ):
        self._transcriber = transcriber
        # Captured once per session: vocabulary/prompt changes made while this
        # recording runs apply to the next one, never halfway through this.
        self._bias = bias or Bias()
        self._typer = typer
        # Output-ownership gate: the app sets this so a superseded session (its
        # generation was overtaken by a newer recording) stops emitting ANY
        # typer output on the final path, which would otherwise corrupt the new
        # session's text on non-IBus typers. Defaults to "always current".
        self._is_current = is_current or (lambda: True)
        self._recorder: AudioSource | None = recorder
        self._sample_rate = recorder.sample_rate
        self._generation = generation
        if pipeline is None:
            from voiceio.core.pipeline import Pipeline
            pipeline = Pipeline([])
        self._pipeline = pipeline
        self._on_typer_broken = on_typer_broken
        self._on_interim = on_interim
        self._typer_fail_count = 0
        self._typer_broken_signalled = False
        self._typed_text = ""
        self._pending = threading.Event()
        self._stop_event = threading.Event()
        # The final pass runs in stop()'s thread, concurrently with whatever
        # interim the worker has in flight. A freeze in flight is progress the
        # final pass builds on, so it waits for that (freeze lock); an interim
        # is not, so it is abandoned — never waited for. Display updates are
        # serialized (apply lock) so an abandoned interim can't type over the
        # final text.
        self._freeze_lock = threading.Lock()
        # Sample offset where an in-flight freeze will end, else None: lets
        # stop() decode the rest in parallel instead of waiting for it.
        self._freeze_pending_end: int | None = None
        self._apply_lock = threading.Lock()
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
        self._pending.set()  # wake the worker so it exits
        dur = len(audio) / self._sample_rate if audio is not None else 0.0
        pre = self._decode_past_pending_freeze(audio)
        # Wait for an in-flight freeze (bounded like a decode), never for an
        # in-flight interim: its result is discarded anyway.
        if self._freeze_lock.acquire(timeout=transcribe_timeout(dur) + 2):
            try:
                self._transcribe_and_apply(min_seconds=0.5, final=True, pre=pre)
            except Exception:
                log.exception("Final transcribe/apply error")
            finally:
                self._freeze_lock.release()
        else:
            log.warning("Freeze pass did not finish in time (gen=%d)", self._generation)
            self._commit_interim_on_final()
        self._final_audio = None  # release memory
        # Release recorder reference — session must not touch it after stop
        if self._recorder is not None:
            self._recorder.set_on_speech_pause(None)
            self._recorder = None
        log.debug("Streaming session stopped, typed: %s", describe_transcript(self._typed_text))
        return self._typed_text

    def _decode_past_pending_freeze(self, audio: np.ndarray | None):
        """If a freeze is decoding on the final lane, decode the audio after
        its cut on the draft lane meanwhile, so stop waits for the longer of
        the two instead of their sum. Returns (start_sample, result) or None."""
        end = self._freeze_pending_end
        draft = getattr(self._transcriber, "draft", None)
        if end is None or draft is None or audio is None or len(audio) <= end:
            return None
        try:
            return end, draft.decode(audio[end:], final=True, bias=self._tail_bias())
        except Exception:  # noqa: BLE001 — the normal final path still runs
            log.debug("Parallel tail decode failed; decoding after the freeze", exc_info=True)
            return None

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
            except (subprocess.CalledProcessError, OSError):
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

    def _tail_bias(self) -> Bias:
        """Session bias plus the frozen-text suffix, which conditions the tail
        decode so sentences stay coherent across a freeze cut."""
        context = self._frozen_raw[-_FREEZE_CONTEXT_CHARS:] if self._frozen_raw else None
        return self._bias.with_context(context)

    def _freeze_chunk(self, audio: np.ndarray, chunk: np.ndarray) -> None:
        """Beam-decode `chunk` (the tail up to a pause cut) and freeze it.

        On decode failure the boundary is left untouched — the audio will be
        decoded again by a later pass (never treat lost audio as silence).
        """
        t0 = time.monotonic()
        try:
            result = self._transcriber.decode(chunk, final=True, bias=self._tail_bias())
        except Exception:
            log.exception("Freeze decode failed (boundary unchanged)")
            return
        chunk_text = result.text
        t_decode = time.monotonic() - t0
        self._frozen_raw = _join_text(self._frozen_raw, chunk_text)
        self._frozen_samples += len(chunk)
        self._frozen_segments.extend(result.segments)
        self._last_decode_end = len(audio)
        self._last_decode_secs = t_decode
        self.trace.append({
            "t": round(time.monotonic() - self._t0, 2),
            "kind": "freeze",
            "frozen_secs": round(self._frozen_samples / self._sample_rate, 2),
            "tail_secs": round(len(chunk) / self._sample_rate, 2),
            "secs": round(t_decode, 3),
            "text": chunk_text,
        })
        log.debug(
            "Froze %.1fs of audio in %.1fs (frozen text now %d chars)",
            len(chunk) / self._sample_rate, t_decode, len(self._frozen_raw),
        )

    def _transcribe_and_apply(
        self, min_seconds: float = 1.0, final: bool = False, pre=None,
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

        # Freeze: once the tail is long enough, cut it at an interior speech
        # pause and beam-decode ONLY that chunk (its text is never decoded
        # again). The freeze pass does not touch the display — the interim
        # text already covers this audio; the next interim refreshes it.
        if (not final and self._freeze_secs > 0
                and len(tail) >= self._freeze_secs * self._sample_rate):
            # Past the whole search span with no pause found: force a cut.
            force = len(tail) >= (self._freeze_secs + _FREEZE_SEARCH_SECS) * self._sample_rate
            cut = _find_freeze_cut(tail, self._sample_rate, self._freeze_secs, force=force)
            if cut is not None:
                with self._freeze_lock:
                    # Stop may have won the race for the lock and already
                    # finalized from the current boundary.
                    if not self._stop_event.is_set():
                        self._freeze_pending_end = self._frozen_samples + cut
                        try:
                            self._freeze_chunk(audio, tail[:cut])
                        finally:
                            self._freeze_pending_end = None
                return

        tail_text = ""
        tail_segments: list[dict] = []
        t0 = time.monotonic()
        # The final pass decodes ANY remaining audio (a clipped last word
        # must not be dropped); interim passes skip sub-0.3s tails.
        if len(tail) > (0 if final else self._sample_rate * 0.3):
            try:
                if pre is not None and pre[0] == self._frozen_samples:
                    result = pre[1]  # decoded in parallel with the last freeze
                else:
                    result = self._transcriber.decode(tail, final=final, bias=self._tail_bias())
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
            tail_text = result.text
            tail_segments = list(result.segments)
        t_transcribe = time.monotonic() - t0
        self._last_decode_end = len(audio)
        self._last_decode_secs = t_transcribe

        self.trace.append({
            "t": round(time.monotonic() - self._t0, 2),
            "kind": "final" if final else "interim",
            "frozen_secs": round(self._frozen_samples / self._sample_rate, 2),
            "tail_secs": round(len(tail) / self._sample_rate, 2),
            "secs": round(t_transcribe, 3),
            "text": tail_text,
        })

        raw = _join_text(self._frozen_raw, tail_text)

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

        with self._apply_lock:
            # Re-check under the lock: the final pass may have committed while
            # this interim was decoding.
            if not final and self._stop_event.is_set():
                return
            self._process_and_display(raw, audio, final)

    def _process_and_display(self, raw: str, audio: np.ndarray, final: bool) -> None:
        if raw:
            t1 = time.monotonic()
            processed = self._pipeline(raw, final=final)
            text, abort = processed.text, processed.abort
            if final:
                self.final_latency["pipeline"] = round(time.monotonic() - t1, 3)
                if "postcorrect" in processed.timings:
                    self.final_latency["postcorrect"] = processed.timings["postcorrect"]
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
                log.debug("Preedit commit: %s", describe_transcript(new_text))
            elif new_text != old:
                self._typer.update_preedit(new_text)
                self._typed_text = new_text
                log.debug("Preedit update: %s", describe_transcript(new_text))
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
            log.debug("Initial: %s", describe_transcript(new_text))
            return

        # Short text: allow full replacement (Whisper still warming up)
        if len(old.split()) <= 2:
            self._typer.delete_chars(len(old))
            time.sleep(DELETE_SETTLE_SECS)
            self._typer.type_text(new_text)
            self._typed_text = new_text
            log.debug("Replaced short text: %s", describe_transcript(new_text))
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
            log.debug("Appended: %s", describe_transcript(to_type.strip()))
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
            log.debug("Final correction: delete %d, type %s", to_delete, describe_transcript(to_type))
            self._typer.delete_chars(to_delete)
            time.sleep(DELETE_SETTLE_SECS)
        if to_type:
            self._typer.type_text(to_type)

        self._typed_text = new_text
