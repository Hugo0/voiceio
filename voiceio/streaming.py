"""Streaming transcription with word-level append and final correction."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from voiceio.transcriber import TRANSCRIBE_TIMEOUT
from voiceio.typers.base import StreamingTyper

if TYPE_CHECKING:
    import numpy as np
    from voiceio.recorder import AudioRecorder
    from voiceio.transcriber import Transcriber
    from voiceio.typers.base import TyperBackend

log = logging.getLogger(__name__)
DELETE_SETTLE_SECS = 0.05  # delay between delete and type for ydotool reliability


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


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
    ):
        self._transcriber = transcriber
        self._typer = typer
        self._recorder: AudioRecorder | None = recorder
        self._generation = generation
        self._typed_text = ""
        self._pending = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._final_audio: np.ndarray | None = None  # set on stop

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
            self._worker_thread.join(timeout=TRANSCRIBE_TIMEOUT + 2)
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
            self._transcribe_and_apply()

        # Final transcription on stop using the audio snapshot
        self._transcribe_and_apply(min_seconds=0.5, final=True)
        self._final_audio = None  # release memory

    def _transcribe_and_apply(
        self, min_seconds: float = 1.0, final: bool = False,
    ) -> None:
        """Get audio, transcribe, apply correction."""
        if final and self._final_audio is not None:
            # Use the snapshot passed to stop() — recorder may be gone
            audio = self._final_audio
        elif self._recorder is not None:
            audio = self._recorder.get_audio_so_far()
        else:
            return  # recorder released, nothing to do

        if audio is None:
            return
        if len(audio) < 16000 * min_seconds:
            return

        try:
            text = self._transcriber.transcribe(audio)
        except Exception:
            log.exception("Streaming transcription failed")
            return

        if text:
            self._apply_correction(text, final=final)

        if final and self._recorder is not None:
            self._recorder.mark_transcribed(len(audio))

    def _apply_correction(self, new_text: str, final: bool = False) -> None:
        """Apply correction to typed text.

        With StreamingTyper (IBus): use preedit during streaming, commit on final.
        Without: append-only via word-level matching, char-level diff on final.
        """
        old = self._typed_text

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
