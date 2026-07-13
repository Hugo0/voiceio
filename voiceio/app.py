"""Main VoiceIO engine: explicit state machine with generation-based cancellation."""
from __future__ import annotations

import enum
import logging
import os
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from voiceio import config, platform as plat, tray
from voiceio.commands import CommandProcessor
from voiceio.corrections import CorrectionDict
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.hotkeys.socket_backend import SocketHotkey
from voiceio.recorder import AudioRecorder
from voiceio.streaming import StreamingSession
from voiceio.transcriber import Transcriber
from voiceio.typers import chain as typer_chain
from voiceio.vad import load_vad
from voiceio.vocabulary import VocabularyLoader

if TYPE_CHECKING:
    from voiceio.typers.base import TyperBackend
log = logging.getLogger("voiceio")
_DEBOUNCE_SECS = 0.8

# Env vars needed for clipboard/tray/typing on graphical sessions.
# XDG_CURRENT_DESKTOP is required for is_gnome() detection, which gates
# IBus input source configuration.
_GRAPHICAL_ENV_VARS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
)


def _redetect_platform():
    """Clear the cached Platform and re-run detection.

    plat.detect() is @lru_cache'd, so the first call freezes the result.
    When voiceio starts before the desktop session has exported its env
    vars, that first call returns display='unknown' and we'd be stuck
    with that forever. Always clear the cache before re-detecting.
    """
    plat.detect.cache_clear()
    return plat.detect()


_graphical_env_complete = False


def _import_graphical_env() -> None:
    """Pull graphical session env vars from the systemd user manager.

    When started as a systemd user service, the process may launch before
    the desktop session imports DISPLAY/WAYLAND_DISPLAY. This queries
    ``systemctl --user show-environment`` to pick them up after the fact.

    Once all expected vars are present, sets ``_graphical_env_complete``
    and becomes a free no-op so callers in the health loop don't spawn
    a subprocess every 10s.
    """
    global _graphical_env_complete
    if _graphical_env_complete:
        return
    missing = [v for v in _GRAPHICAL_ENV_VARS if v not in os.environ]
    if not missing:
        _graphical_env_complete = True
        return
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show-environment"],
            text=True, timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return
    for line in out.splitlines():
        key, _, val = line.partition("=")
        if key in missing and val:
            os.environ[key] = val
            log.info("Imported %s=%s from systemd user env", key, val)
    if not [v for v in _GRAPHICAL_ENV_VARS if v not in os.environ]:
        _graphical_env_complete = True


_HEALTH_CHECK_INTERVAL = 10  # seconds between health checks

# Whisper's decoder has a hard 448-token sequence budget shared by hotwords,
# initial_prompt AND the transcription output. Hotwords + prompt must stay
# well under half of it or output gets truncated mid-utterance.
# ~600 chars ≈ 150 tokens.
_HOTWORDS_MAX_CHARS = 600


class _State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class VoiceIO:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg

        # Import graphical env vars early so typer/platform detection
        # sees DISPLAY, WAYLAND_DISPLAY, XDG_SESSION_TYPE, XDG_CURRENT_DESKTOP
        # even when started as a systemd user service before the desktop
        # session has exported them.
        _import_graphical_env()

        # Fresh detection (clears lru_cache in case anything warmed it)
        self.platform = _redetect_platform()
        log.info(
            "Platform: display=%s desktop=%s",
            self.platform.display_server, self.platform.desktop,
        )

        # Select backends
        self._hotkey = hotkey_chain.select(self.platform, cfg.hotkey.backend)
        self._typer = typer_chain.select(self.platform, cfg.output.method)
        self._auto_fallback = cfg.health.auto_fallback

        # Socket backend runs alongside native hotkey for extra robustness
        self._socket: SocketHotkey | None = None
        if self._hotkey.name != "socket":
            self._socket = SocketHotkey()
        elif self.platform.desktop not in ("GNOME", "KDE"):
            # Socket backend has no physical hotkey listener — user must bind
            # `voiceio toggle` to a key in their WM config.
            log.warning(
                "Hotkey backend is 'socket' on desktop '%s'. "
                "Physical hotkey capture is not available. "
                "Bind 'voiceio toggle' to a key in your window manager config "
                "(e.g. i3: bindsym Ctrl+Alt+v exec voiceio toggle). "
                "Or: sudo usermod -aG input $USER && log out/in to enable evdev.",
                self.platform.desktop,
            )
            print(
                f"\n  NOTE: No physical hotkey listener available on '{self.platform.desktop}'.\n"
                f"  Bind 'voiceio toggle' to a key in your WM config, e.g.:\n"
                f"    i3/sway:  bindsym Ctrl+Alt+v exec voiceio toggle\n"
                f"    hyprland: bind = CTRL ALT, V, exec, voiceio toggle\n"
                f"  Or fix evdev: sudo usermod -aG input $USER (then log out/in)\n",
            )

        print(f"Loading model '{cfg.model.name}'...", end="", flush=True)
        t0 = time.monotonic()
        self.transcriber = Transcriber(cfg.model)
        print(f" ready ({time.monotonic() - t0:.1f}s)")

        # VAD, vocabulary, commands
        vad = load_vad(cfg.audio)
        self.recorder = AudioRecorder(cfg.audio, vad=vad)

        self._corrections = CorrectionDict()
        # mtime-cached loader so `voiceio correct` vocabulary edits are picked
        # up per-recording without a daemon restart.
        self._vocab_loader = VocabularyLoader(cfg.model)
        # Vocabulary biases the decoder via hotwords; initial_prompt carries
        # recent-transcript context (rebuilt per recording via PromptBuilder).
        self._hotwords = ""
        self._refresh_hotwords()
        from voiceio.prompt import PromptBuilder
        self._prompt_builder = PromptBuilder()

        self._command_processor = CommandProcessor(enabled=cfg.commands.enabled, editing=cfg.commands.editing)
        self._cleanup = cfg.output.punctuation_cleanup
        self._number_conversion = cfg.output.number_conversion
        self._voice_input_prefix = cfg.output.voice_input_prefix
        self._streaming = cfg.output.streaming

        # LLM post-processing (optional, requires Ollama)
        # Processor is kept even when unavailable — is_available() has a retry cooldown
        self._llm = None
        if cfg.llm.enabled:
            from voiceio.llm import LLMProcessor
            self._llm = LLMProcessor(cfg.llm)
            if self._llm.is_available():
                log.info("LLM: %s via %s", cfg.llm.model or "auto", cfg.llm.base_url)
            else:
                log.warning("LLM enabled but Ollama not available (will retry)")

        # Constrained LLM post-correction (optional, cloud API, final pass only)
        self._postcorrect = None
        if cfg.postcorrect.enabled:
            from voiceio.postcorrect import PostCorrector
            self._postcorrect = PostCorrector(cfg)
            if self._postcorrect.is_available():
                log.info("PostCorrect: %s", cfg.postcorrect.model or cfg.autocorrect.model)
            else:
                log.warning("PostCorrect enabled but no API key resolved (disabled)")

        # Explicit state machine
        self._state = _State.IDLE
        self._generation = 0          # incremented on every stop; leaked threads check this
        self._session: StreamingSession | None = None
        self._record_start: float = 0
        self._last_clip_warn: float = 0
        self._context_title: str | None = None
        # Prune retained recordings over the size cap (startup housekeeping)
        from voiceio import retention
        threading.Thread(
            target=retention.prune, args=(cfg.data,), daemon=True,
        ).start()

        # Hotkey deduplication
        self._hotkey_lock = threading.Lock()
        self._last_hotkey: float = 0

        # TTS (text-to-speech)
        self._tts_engine = None
        self._tts_player = None
        self._tts_hotkey = None
        if cfg.tts.enabled:
            from voiceio.tts import select as tts_select
            from voiceio.tts.player import TTSPlayer
            self._tts_engine = tts_select(cfg.tts)
            if self._tts_engine:
                self._tts_player = TTSPlayer()
                # Create a second hotkey backend for TTS
                self._tts_hotkey = hotkey_chain.select(self.platform, cfg.hotkey.backend)
                log.info("TTS: engine=%s, hotkey=%s", self._tts_engine.name, cfg.tts.hotkey)
            else:
                log.warning("TTS enabled but no engine available")

        # IBus engine management
        self._prev_ibus_engine: str | None = None
        self._engine_proc: subprocess.Popen | None = None
        # Engine lifecycle: consecutive missed pings (one miss on a busy
        # system is not a zombie) and a guard against concurrent spawns
        # (health loop vs record-start).
        self._engine_ping_fails = 0
        self._engine_spawn_lock = threading.Lock()
        self._shutdown = threading.Event()
        # IBus input-source ownership. We claim the GNOME input source ONLY
        # while a recording is live (RECORDING/FINALIZING) and restore whatever
        # the user actually had. Never re-forced while IDLE so we don't fight
        # users who run a real IME (CJK).
        self._prev_input_source_index: int | None = None
        self._voiceio_source_index: int | None = None
        # Set when the user switched input source away from voiceio mid-record:
        # we stop fighting and fall back to clipboard output for the session.
        self._ibus_session_fallback = False

        # Audio stream recovery backoff
        self._stream_fail_count = 0
        self._next_stream_retry: float = 0

    def request_shutdown(self) -> None:
        """Request graceful shutdown from an external signal handler."""
        self._shutdown.set()

    # ── Hotkey entry points ─────────────────────────────────────────────

    def on_hotkey(self) -> None:
        """Toggle recording. Called by hotkey backends (evdev, socket)."""
        with self._hotkey_lock:
            now = time.monotonic()
            if now - self._last_hotkey < _DEBOUNCE_SECS:
                return
            self._last_hotkey = now
            self._toggle()

    def _on_auto_stop(self) -> None:
        """Called from audio thread when sustained silence triggers auto-stop."""
        threading.Thread(target=self._request_stop, daemon=True).start()

    def _request_stop(self) -> None:
        """Stop recording if active. Unlike on_hotkey, never starts."""
        with self._hotkey_lock:
            now = time.monotonic()
            if now - self._last_hotkey < _DEBOUNCE_SECS:
                return
            self._last_hotkey = now
            if self._state == _State.RECORDING:
                self._do_stop()

    # ── State machine ───────────────────────────────────────────────────

    def _toggle(self) -> None:
        """Central state transition: IDLE/FINALIZING/ERROR → RECORDING, RECORDING → stop."""
        if self._state == _State.RECORDING:
            self._do_stop()
        elif self._state == _State.ERROR:
            # Attempt recovery: try to restart worker, then start recording
            log.info("Attempting recovery from error state")
            try:
                self.transcriber._ensure_worker()
                self._state = _State.IDLE
                tray.set_error(False)
                self._do_start()
            except Exception:
                log.exception("Recovery failed")
        elif self._state in (_State.IDLE, _State.FINALIZING):
            # Allow starting a new recording even while old one finalizes.
            # The generation counter ensures the old finalizer exits cleanly.
            self._do_start()

    def _refresh_hotwords(self) -> None:
        """Rebuild Whisper hotwords from vocabulary + correction targets.

        The vocabulary read is mtime-cached (only a `stat` when unchanged), and
        `set_hotwords` is called only when the merged string actually changed,
        so this stays cheap enough to run on every recording start.
        """
        vocab = self._vocab_loader.get()
        # Only merge RARE correction targets (proper nouns, technical terms):
        # common-word targets ("review", "company") add nothing to decoder bias
        # and hundreds of them blow Whisper's 448-token prompt+output budget,
        # which truncates transcriptions mid-utterance.
        from voiceio.wordfreq import is_common
        vocab_terms = [
            t for t in self._corrections.vocabulary_terms()
            if not is_common(t, self.cfg.model.language)
        ]
        if vocab_terms:
            extra = ", ".join(vocab_terms)
            vocab = f"{vocab}, {extra}" if vocab else extra
        vocab = vocab[:_HOTWORDS_MAX_CHARS]
        if vocab != self._hotwords:
            self._hotwords = vocab
            if vocab:
                self.transcriber.set_hotwords(vocab)

    def _do_start(self) -> None:
        """Transition to RECORDING."""
        # Pre-flight: ensure audio stream is healthy before recording
        ok, reason = self.recorder.stream_health()
        if not ok:
            log.warning("Audio stream unhealthy before recording: %s — reopening", reason)
            try:
                self.recorder.reopen_stream()
            except Exception:
                log.exception("Cannot reopen audio stream, aborting recording")
                return

        if self.recorder.is_zombie():
            # Full pre-buffer of digital zeros with a "healthy" stream is the
            # post-suspend zombie signature (callbacks fire, capture node
            # gone). Reopening fixes it; a failed reopen means no audio is
            # possible, so abort rather than record silence.
            log.warning("Zombie audio stream (all-zero pre-buffer) — reopening")
            try:
                self.recorder.reopen_stream()
                time.sleep(0.4)  # let the new stream fill some pre-buffer
            except Exception:
                log.exception("Audio stream reopen failed, aborting recording")
                from voiceio import feedback
                feedback.notify(
                    "VoiceIO: microphone unavailable",
                    "Could not reopen the audio stream — check your input device.",
                )
                return
            if not self.recorder.has_signal():
                log.warning("Mic still silent after reopen — muted or wrong device?")
                from voiceio import feedback
                feedback.notify(
                    "VoiceIO: microphone appears silent",
                    "Check that your mic is unmuted and the right input device is selected.",
                )
        elif not self.recorder.has_signal():
            log.warning("Mic appears silent or muted (pre-buffer is all zeros)")

        self._state = _State.RECORDING
        self._ibus_session_fallback = False
        if self._typer.name == "ibus" and self._engine_proc is None:
            # Engine was reaped while dormant (normal — see _check_health).
            # Respawn in the background; the engine's pending buffer holds
            # preedits until the instance is ready, and with state=RECORDING
            # the spawn leaves the voiceio source active instead of yanking
            # it back to dormant mid-recording.
            threading.Thread(
                target=self._respawn_engine_for_recording, daemon=True,
            ).start()
        self._activate_ibus()
        self._corrections.load()  # hot-reload corrections on each recording
        self._refresh_hotwords()  # hot-reload vocabulary (mtime-cached)
        self.transcriber.set_initial_prompt(self._prompt_builder.build())
        if self.cfg.data.capture_context:
            # Snapshot the dictation target now — focus may change by finalize
            self._context_title = None
            threading.Thread(target=self._capture_context, daemon=True).start()
        self._record_start = time.monotonic()
        self.recorder.start()
        self.recorder.set_on_auto_stop(self._on_auto_stop)
        self._play_record_cue(start=True)
        tray.set_recording(True)
        if self._streaming:
            self._session = StreamingSession(
                self.transcriber, self._typer, self.recorder,
                generation=self._generation,
                cleanup=self._cleanup,
                number_conversion=self._number_conversion,
                language=self.cfg.model.language,
                commands=self._command_processor,
                corrections=self._corrections,
                postcorrect=self._postcorrect,
                llm=self._llm,
                voice_input_prefix=self._voice_input_prefix,
                on_typer_broken=self._on_typer_broken,
                on_interim=self._on_interim_text,
                freeze_secs=self.cfg.output.streaming_freeze_secs,
            )
            self._session.start()
        log.info("Recording... press [%s] again to stop", self.cfg.hotkey.key)

    def _do_stop(self) -> None:
        """Transition from RECORDING to FINALIZING (or IDLE for batch)."""
        elapsed = time.monotonic() - self._record_start
        self._generation += 1
        gen = self._generation

        # Stop audio capture immediately — chime and recorder stop are synchronous
        self._play_record_cue(start=False)
        self.recorder.set_on_auto_stop(None)
        audio = self.recorder.stop()
        tray.set_recording(False)
        self._warn_if_clipping()

        if self._postcorrect is not None:
            # Freshest context for the finalize pass: this recording's window
            # title (captured at start) and the latest transcripts/vocabulary.
            self._postcorrect.set_context(
                vocabulary=self._hotwords,
                recent=self._prompt_builder.recent(3),
                title=self._context_title,
            )

        if self._streaming and self._session is not None:
            self._state = _State.FINALIZING
            tray.set_processing(True)
            session = self._session
            self._session = None
            # Output-ownership gate: the moment a newer recording supersedes
            # this generation, the session must stop emitting typer output.
            session.set_is_current(lambda g=gen: self._generation == g)
            # Finalize in background. Pass audio snapshot — session no longer
            # touches the recorder. Generation check cancels if superseded.
            threading.Thread(
                target=self._finalize_streaming,
                args=(session, audio, elapsed, gen),
                daemon=True,
            ).start()
        elif not self._streaming:
            self._state = _State.IDLE
            log.info("Stopped recording (%.1fs)", elapsed)
            if audio is not None:
                # _process owns deactivation (its finally block).
                threading.Thread(
                    target=self._process,
                    args=(audio, gen),
                    daemon=True,
                ).start()
            else:
                # No processing thread will run — release the source here.
                self._deactivate_ibus(gen)
        else:
            # Streaming mode but session already gone (race) — nothing to do
            self._state = _State.IDLE
            log.debug("Stop: streaming session already finalized")
            # No finalizer will run to release the input source, so do it here.
            self._deactivate_ibus(gen)

    def _capture_context(self) -> None:
        from voiceio import retention
        self._context_title = retention.active_window_title()

    def _retention_extra(self, audio: np.ndarray | None) -> dict:
        """Save the utterance audio (before transcription, so it survives
        worker timeouts) and collect context for the history entry."""
        from voiceio import retention
        audio_name = None
        if audio is not None:
            audio_name = retention.save_audio(audio, time.time(), self.cfg.data)
        return {
            "audio": audio_name,
            "context": self._context_title if self.cfg.data.capture_context else None,
            "model": self.cfg.model.name,
        }

    def _strip_voice_prefix(self, text: str) -> str:
        """Remove the configured voice-input prefix before persisting.

        The prefix is presentation for the target app, not dictated content;
        storing it would pollute vocabulary mining and correction analysis.
        """
        if self._voice_input_prefix and text.startswith(self._voice_input_prefix):
            return text[len(self._voice_input_prefix):].lstrip()
        return text

    _CLIP_RATIO_WARN = 0.005   # >0.5% of samples in flat-top runs
    _CLIP_WARN_INTERVAL = 300  # seconds between warnings

    def _warn_if_clipping(self) -> None:
        """Warn (rate-limited) when the mic input is saturating the ADC.

        Clipping cannot be repaired in software — the gain must come down
        before the ADC — so tell the user how to lower it.
        """
        meter = self.recorder.get_meter()
        if meter["clip_ratio"] < self._CLIP_RATIO_WARN:
            return
        now = time.monotonic()
        if now - self._last_clip_warn < self._CLIP_WARN_INTERVAL:
            return
        self._last_clip_warn = now
        log.warning(
            "Microphone is clipping (%.1f%% of samples saturated, peak %.2f) — "
            "transcription quality suffers. Lower the input gain, e.g.: "
            "wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 10%%-",
            meter["clip_ratio"] * 100, meter["peak"],
        )
        from voiceio import feedback
        feedback.notify(
            "VoiceIO: microphone too loud",
            "Input is clipping — lower your mic gain in system sound settings.",
        )

    # ── Background work ─────────────────────────────────────────────────

    def _finalize_streaming(
        self, session: StreamingSession, audio: np.ndarray | None,
        elapsed: float, gen: int,
    ) -> None:
        """Run final transcription and commit in background thread."""
        t_final = time.monotonic()
        # Snappy clipboard: make the best-so-far text pasteable the moment
        # the user stops, instead of after the (possibly long) final decode.
        if (self.cfg.output.copy_to_clipboard == "live"
                and self._typer.name != "clipboard" and session.interim_text):
            from voiceio import clipboard_read
            clipboard_read.copy_text(session.interim_text)
        extra = self._retention_extra(audio)
        final_text = session.stop(audio)
        if self._generation != gen:
            log.debug("Finalize cancelled (gen %d, current %d)", gen, self._generation)
            return
        if final_text:
            self._copy_result_async(final_text)
            self._play_feedback(final_text)
            stored = self._strip_voice_prefix(final_text)
            self._prompt_builder.add_transcript(stored)
            latency = dict(session.final_latency)
            # stop-to-commit: what the user actually waits for
            latency["finalize_total"] = round(time.monotonic() - t_final, 3)
            extra["latency"] = latency
            from voiceio import history
            history.append(
                stored,
                raw=session.raw_final_text,
                segments=session.final_segments,
                duration=elapsed,
                extra=extra,
            )
            from voiceio import retention
            retention.save_trace(self.cfg.data, {
                "ts": time.time(),
                "audio": extra.get("audio"),
                "duration": round(elapsed, 2),
                "latency": latency,
                # Snapshot: a timed-out worker join means the thread may
                # still be appending while we serialize.
                "passes": list(session.trace),
            })
        log.info("Streaming done (%.1fs): '%s'", elapsed, final_text)
        # Release the IBus input source now that the final commit is done.
        # Generation-checked inside: if a newer recording started, it already
        # re-claimed the source and we must not restore it out from under it.
        self._deactivate_ibus(gen)
        # Transition to IDLE under lock to avoid racing with _toggle
        with self._hotkey_lock:
            if self._generation == gen and self._state == _State.FINALIZING:
                self._state = _State.IDLE
                tray.set_processing(False)

    def _process(self, audio: np.ndarray, gen: int) -> None:
        """Batch transcription (non-streaming mode)."""
        try:
            if self._generation != gen:
                return
            extra = self._retention_extra(audio)
            t0 = time.monotonic()
            raw = self.transcriber.transcribe(audio, final=True)
            latency = {
                "audio_secs": round(len(audio) / self.recorder.sample_rate, 2),
                "transcribe": round(time.monotonic() - t0, 3),
            }
            segments = self.transcriber.last_segments
            text = raw
            if self._generation != gen:
                return
            if text:
                from voiceio.postprocess import apply_pipeline
                t1 = time.monotonic()
                text, abort = apply_pipeline(
                    text,
                    do_cleanup=self._cleanup,
                    number_conversion=self._number_conversion,
                    language=self.cfg.model.language,
                    commands=self._command_processor,
                    corrections=self._corrections,
                    postcorrect=self._postcorrect,
                    llm=self._llm,
                    voice_input_prefix=self._voice_input_prefix,
                    final=True,
                )
                latency["pipeline"] = round(time.monotonic() - t1, 3)
                pc_secs = getattr(self._postcorrect, "last_secs", None)
                if pc_secs is not None:
                    latency["postcorrect"] = round(pc_secs, 3)
                extra["latency"] = latency
                if abort:
                    return
                # Superseded by a newer recording — do not touch the typer.
                if text and self._generation == gen:
                    self._type_with_fallback(text)
                    self._copy_result_async(text)
                    self._play_feedback(text)
                    stored = self._strip_voice_prefix(text)
                    self._prompt_builder.add_transcript(stored)
                    from voiceio import history
                    history.append(
                        stored,
                        raw=raw,
                        segments=segments,
                        duration=len(audio) / self.recorder.sample_rate,
                        extra=extra,
                    )
                    log.info("Typed: '%s'", text)
        except Exception:
            log.exception("Processing failed")
        finally:
            # Release the IBus input source after the batch commit (or failure).
            self._deactivate_ibus(gen)

    # ── Text-to-speech ─────────────────────────────────────────────

    def on_tts_hotkey(self) -> None:
        """Toggle TTS: if playing, cancel. Otherwise read clipboard and speak."""
        if self._tts_player is None or self._tts_engine is None:
            return
        if self._tts_player.is_playing():
            self._tts_player.cancel()
            tray.set_processing(False)
            log.info("TTS: cancelled")
            return
        from voiceio import clipboard_read
        text = clipboard_read.read_text()
        if not text:
            log.info("TTS: no text selected")
            return
        if self._state == _State.RECORDING:
            log.warning("TTS: speaking while recording — mic will pick up audio")
        tray.set_processing(True)
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str) -> None:
        """Synthesize and play text (runs in background thread)."""
        try:
            audio, sample_rate = self._tts_engine.synthesize(
                text, self.cfg.tts.voice, self.cfg.tts.speed,
            )
            self._tts_player.play(audio, sample_rate)
        except Exception:
            log.exception("TTS synthesis/playback failed")
        finally:
            tray.set_processing(False)

    # ── IBus management ─────────────────────────────────────────────────

    def _activate_ibus(self) -> None:
        """Claim the GNOME input source for the voiceio engine (record start).

        Records the source the user had so we can restore exactly that on
        deactivation, rather than hardcoding index 0.
        """
        if self._typer.name != "ibus":
            return
        threading.Thread(
            target=self._switch_gnome_input_source,
            args=("voiceio",), daemon=True,
        ).start()

    def _deactivate_ibus(self, gen: int | None = None) -> None:
        """Restore the user's input source after a recording finishes.

        Generation-checked: if a newer recording already re-claimed the source
        (self._generation moved past ``gen``), do NOT restore — that would yank
        the source out from under the live recording. Called after the final
        commit, never while IDLE.
        """
        if self._typer.name != "ibus":
            return
        if gen is not None and self._generation != gen:
            log.debug("Skip IBus deactivate: gen %d superseded by %d", gen, self._generation)
            return
        threading.Thread(
            target=self._restore_input_source, daemon=True,
        ).start()

    # ── Feedback ────────────────────────────────────────────────────────

    def _play_record_cue(self, start: bool) -> None:
        if not self.cfg.feedback.sound_enabled:
            return
        if start:
            from voiceio.feedback import play_record_start
            play_record_start()
        else:
            from voiceio.feedback import play_record_stop
            play_record_stop()

    def _on_interim_text(self, text: str) -> None:
        """Streaming update: mirror the best-so-far text to the clipboard.

        Runs on the streaming worker thread (a copy is a few ms). Skipped for
        the clipboard typer, which pastes FROM the clipboard — overwriting it
        here could race a pending Ctrl+V and paste the full text twice.
        """
        if self.cfg.output.copy_to_clipboard != "live":
            return
        if self._typer.name == "clipboard":
            return
        from voiceio import clipboard_read
        clipboard_read.copy_text(text)

    def _copy_result_async(self, text: str) -> None:
        """Mirror the final text to the clipboard, off the commit hot path."""
        if self.cfg.output.copy_to_clipboard not in ("final", "live"):
            return

        def _copy() -> None:
            # Give a just-sent paste keystroke time to consume the clipboard
            # before we overwrite it with the full text.
            if self._typer.name == "clipboard":
                time.sleep(0.3)
            from voiceio import clipboard_read
            clipboard_read.copy_text(text)

        threading.Thread(target=_copy, daemon=True).start()

    def _play_feedback(self, text: str) -> None:
        if self.cfg.feedback.sound_enabled:
            from voiceio.feedback import play_commit_sound
            play_commit_sound()
        if self.cfg.feedback.notify_clipboard:
            from voiceio.feedback import notify_clipboard
            notify_clipboard(text)

    def _type_with_fallback(self, text: str) -> None:
        try:
            self._typer.type_text(text)
        except Exception as e:
            if not self._auto_fallback:
                raise
            log.warning("Typer '%s' failed: %s, trying fallback", self._typer.name, e)
            probe = self._typer.probe()
            if not probe.ok:
                log.warning("Typer '%s' no longer works: %s", self._typer.name, probe.reason)
            try:
                self._typer = typer_chain.select(self.platform)
                log.info("Switched to typer: %s", self._typer.name)
                self._typer.type_text(text)
            except RuntimeError:
                log.error("No working typer backend available")

    def _on_typer_broken(self) -> None:
        """Called by streaming session when typer fails repeatedly.

        Defers the re-probe+upgrade to a background thread that waits
        for the state to reach IDLE. We cannot hot-swap mid-recording:
        the streaming session tracks ``_typed_text`` in terms of the old
        typer's behavior (char-level for clipboard vs preedit for ibus),
        and mid-stream swapping would leave stale or duplicated chars.
        The current recording is already broken — accept that, fix it
        before the next one.
        """
        threading.Thread(
            target=self._deferred_typer_upgrade,
            daemon=True, name="typer-upgrade",
        ).start()

    def _deferred_typer_upgrade(self) -> None:
        """Wait for IDLE state, then re-detect platform and upgrade typer.

        Lock discipline: all subprocess probing (redetect, chain resolve,
        select, engine start) runs OUTSIDE ``_hotkey_lock``. Only the field
        assignments go through the lock-guarded swap helpers, which apply the
        change only while IDLE. We must never hold the lock across a typer or
        engine subprocess call.
        """
        # Wait up to 30s for recording/finalizing to complete
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._state == _State.IDLE or self._shutdown.is_set():
                break
            time.sleep(0.5)
        if self._shutdown.is_set():
            return
        if self._state != _State.IDLE:
            log.debug("Typer upgrade abandoned: state=%s after 30s", self._state)
            return
        _import_graphical_env()
        if not self._swap_platform(_redetect_platform()):
            return  # left IDLE before we could apply
        log.info(
            "Typer broken: re-detected platform: display=%s desktop=%s",
            self.platform.display_server, self.platform.desktop,
        )
        self._try_upgrade_typer(reason="streaming-failure")

    def _swap_platform(self, new_platform) -> bool:
        """Assign ``self.platform`` under the lock, only while IDLE.

        Health/upgrade threads must not mutate shared state that the hotkey and
        finalizer paths read, except while IDLE and holding the lock. Returns
        True if applied.
        """
        with self._hotkey_lock:
            if self._state != _State.IDLE:
                return False
            self.platform = new_platform
            return True

    def _swap_typer(self, new_typer: TyperBackend, reason: str = "") -> bool:
        """Assign ``self._typer`` under the lock, only while IDLE.

        Contains NO subprocess/engine operations — callers perform those
        (probing, engine start) outside the lock. Returns True if applied.
        """
        with self._hotkey_lock:
            if self._state != _State.IDLE:
                log.debug("Typer swap (%s) skipped: state=%s", reason, self._state)
                return False
            old = self._typer.name
            self._typer = new_typer
        log.info("Typer upgraded: %s -> %s (%s)", old, new_typer.name, reason)
        return True

    def _try_upgrade_typer(self, reason: str = "") -> bool:
        """Try to switch to a better typer backend.

        Called on resume, after env import, or after repeated typer failures.
        If the current typer is a fallback (e.g. clipboard) and a preferred
        backend (e.g. ibus) is now available, switch to it.

        Returns True if typer was upgraded.
        """
        chain = typer_chain._get_chain(self.platform)
        current_idx = chain.index(self._typer.name) if self._typer.name in chain else len(chain)
        log.debug(
            "Upgrade attempt (%s): current=%s (idx %d) chain=%s",
            reason, self._typer.name, current_idx, chain,
        )
        if current_idx == 0:
            log.debug("Upgrade (%s) skipped: already on best backend in chain", reason)
            return False  # already on the best backend

        # Reset clipboard tool cache so re-probe sees updated env
        from voiceio.typers.clipboard import ClipboardTyper
        if isinstance(self._typer, ClipboardTyper):
            self._typer.reset_tools()

        # Log all probe results so we can diagnose why upgrade failed
        try:
            results = typer_chain.resolve(self.platform, self.cfg.output.method)
        except Exception:
            log.exception("Upgrade (%s): typer resolve failed", reason)
            return False
        for name, _backend, probe in results:
            status = "OK" if probe.ok else f"FAIL: {probe.reason}"
            log.info("Upgrade (%s): probe %s -> %s", reason, name, status)

        try:
            new_typer = typer_chain.select(self.platform, self.cfg.output.method)
        except RuntimeError as e:
            log.warning("Upgrade (%s) select failed: %s", reason, e)
            return False

        new_idx = chain.index(new_typer.name) if new_typer.name in chain else len(chain)
        if new_idx < current_idx:
            # Apply the swap under the lock (IDLE-gated); if it didn't apply
            # (state left IDLE), don't start the engine.
            if not self._swap_typer(new_typer, reason):
                return False
            if new_typer.name == "ibus" and self._engine_proc is None:
                try:
                    self._ensure_ibus_engine()  # subprocess ops, outside the lock
                except Exception:
                    log.exception("Upgrade (%s): failed to start IBus engine", reason)
            return True

        log.info(
            "Upgrade (%s): no better backend available (stayed on %s)",
            reason, new_typer.name,
        )
        # Even if same backend, re-resolve tools (e.g. clipboard switching
        # from xclip to wl-copy after env vars change)
        if isinstance(self._typer, ClipboardTyper):
            self._typer._resolve_tools()
            log.debug("Clipboard typer tools re-resolved (%s)", reason)

        return False

    # ── IBus engine lifecycle ───────────────────────────────────────────

    def _respawn_engine_for_recording(self) -> None:
        """Background lazy respawn at record start. Never raises."""
        try:
            self._ensure_ibus_engine()
        except Exception:
            log.exception("IBus engine respawn at record start failed")

    def _ensure_ibus_engine(self) -> None:
        """Start the VoiceIO IBus engine and activate it.

        Single-flight: a concurrent spawn (health loop vs record start)
        is skipped rather than queued — two dances would fight over the
        input source.
        """
        if not self._engine_spawn_lock.acquire(blocking=False):
            log.debug("IBus engine spawn already in progress, skipping")
            return
        try:
            self._spawn_ibus_engine()
        finally:
            self._engine_spawn_lock.release()

    def _spawn_ibus_engine(self) -> None:
        from voiceio.ibus import READY_PATH, SOCKET_PATH
        from voiceio.typers.ibus import LAUNCHER_PATH, _ibus_env

        ibus_env = _ibus_env()

        # Save current engine for restore on exit
        try:
            result = subprocess.run(
                ["ibus", "engine"], capture_output=True, text=True,
                timeout=3, env=ibus_env,
            )
            if result.returncode == 0:
                prev = result.stdout.strip()
                if prev != "voiceio":
                    self._prev_ibus_engine = prev
                    log.debug("Previous IBus engine: %s", prev)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Kill any stale engine process from a previous session
        self._kill_stale_engine(SOCKET_PATH)
        READY_PATH.unlink(missing_ok=True)

        # Spawn the engine process directly
        log.info("Starting VoiceIO IBus engine...")
        try:
            self._engine_proc = subprocess.Popen(
                [str(LAUNCHER_PATH)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                env=ibus_env,
            )
        except OSError as e:
            log.warning("Could not start IBus engine: %s", e)
            return

        # Phase 1: wait for socket
        for i in range(40):
            if SOCKET_PATH.exists():
                log.info("VoiceIO IBus engine socket ready (%.1fs)", i * 0.1)
                break
            time.sleep(0.1)
        else:
            if self._engine_proc.poll() is not None:
                stderr = self._engine_proc.stderr.read().decode(errors="replace") if self._engine_proc.stderr else ""
                log.error("IBus engine crashed (rc=%d): %s", self._engine_proc.returncode, stderr.strip()[-500:])
            else:
                log.warning("IBus engine started but socket not found, commands may fail")
            return

        # Remember the source the user is on now so we can return the engine
        # to dormancy without hardcoding index 0 (which may not be the user's
        # keyboard). Phase 2 briefly activates voiceio to spawn the instance.
        dormant_idx = self._get_current_input_source_index()

        # Phase 2: activate via `ibus engine voiceio`
        log.info("Activating VoiceIO IBus engine...")
        activate_proc = subprocess.Popen(
            ["ibus", "engine", "voiceio"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=ibus_env,
        )

        # Phase 3: wait for engine instance
        for i in range(200):
            if READY_PATH.exists():
                log.info("VoiceIO IBus engine instance ready (%.1fs)", i * 0.1)
                break
            time.sleep(0.1)
        else:
            log.warning("IBus engine instance not created, preedit may not work")

        try:
            activate_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            activate_proc.kill()

        # Return to dormancy on the user's real source (not hardcoded 0) —
        # unless a recording claimed the source while we were spawning:
        # yanking it away now would kill the live preedit for that whole
        # recording (the "underlined text never appears" symptom).
        if self._state == _State.IDLE:
            self._set_gnome_input_source_index(dormant_idx if dormant_idx is not None else 0)
            self._release_ibus_engine()
            log.info("VoiceIO IBus engine ready (dormant until recording)")
        else:
            log.info("VoiceIO IBus engine ready (recording live — staying active)")

    def _ping_ibus_engine(self) -> bool:
        """Check if the IBus engine's socket listener is alive.

        Returns True if the engine responds to a ping within 1 second.
        A False result means the engine process is alive but its socket
        listener / GLib loop is dead (zombie engine).
        """
        import socket as _socket
        from voiceio.ibus import SOCKET_PATH
        try:
            sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM)
            sock.settimeout(2.0)
            sock.bind("")
            sock.sendto(b"ping", str(SOCKET_PATH))
            data, _ = sock.recvfrom(64)
            sock.close()
            return data == b"pong"
        except (OSError, _socket.timeout):
            return False

    def _detect_input_source_hijack(self) -> None:
        """If the user switched input source away from voiceio mid-recording,
        stop fighting: log it and fall back to the clipboard typer for the
        rest of this session instead of re-forcing our engine.
        """
        if self._ibus_session_fallback or self._voiceio_source_index is None:
            return
        cur = self._get_current_input_source_index()
        if cur is None or cur == self._voiceio_source_index:
            return
        log.warning(
            "Input source switched away from voiceio mid-recording (now index %d); "
            "falling back to clipboard for this session (not fighting the user)",
            cur,
        )
        self._ibus_session_fallback = True
        try:
            from voiceio.typers.clipboard import ClipboardTyper
            clip = ClipboardTyper(self.platform)
        except Exception:
            log.debug("Could not build clipboard fallback typer", exc_info=True)
            return
        session = self._session
        if session is not None:
            session.set_typer(clip)

    def _wait_for_ibus(self, chain: list[str]) -> TyperBackend | None:
        """Wait for IBus daemon to become available and switch to IBus typer.

        At session startup, voiceio may start before ibus-daemon is ready.
        This polls briefly so we can use IBus instead of a broken fallback.
        """
        from voiceio.typers.ibus import _ibus_daemon_running

        log.info("IBus preferred but not available yet, waiting for ibus-daemon...")
        for i in range(30):  # up to ~15 seconds
            time.sleep(0.5)
            if _ibus_daemon_running():
                log.info("IBus daemon ready after %.1fs, re-probing typers", (i + 1) * 0.5)
                try:
                    typer = typer_chain.select(self.platform, self.cfg.output.method)
                    if typer.name == "ibus":
                        self._ensure_ibus_engine()
                        return typer
                except RuntimeError:
                    pass
                break
        log.warning("IBus daemon did not start in time, using fallback typer: %s", self._typer.name)
        return None

    @staticmethod
    def _kill_stale_engine(socket_path) -> None:
        socket_path.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "voiceio.ibus.engine"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid:
                        log.debug("Killing stale engine process %s", pid)
                        subprocess.run(["kill", pid], capture_output=True, timeout=3)
                time.sleep(0.3)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _get_current_input_source_index(self) -> int | None:
        """Read the active GNOME input source index, or None if unavailable."""
        if not self.platform.is_gnome:
            return None
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.input-sources", "current"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    def _restore_input_source(self) -> None:
        """Restore the input source the user had before we claimed it."""
        if not self.platform.is_gnome:
            return
        idx = self._prev_input_source_index
        self._set_gnome_input_source_index(idx if idx is not None else 0)
        self._prev_input_source_index = None
        self._voiceio_source_index = None
        self._release_ibus_engine()

    def _ibus_engine_fallback(self) -> str:
        """The user's non-voiceio ibus engine name, for releasing the engine."""
        if self._prev_ibus_engine and self._prev_ibus_engine != "voiceio":
            return self._prev_ibus_engine
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
                capture_output=True, text=True, timeout=3,
            )
            import ast
            for kind, name in ast.literal_eval(result.stdout.strip()):
                if kind == "xkb":
                    return f"xkb:{name}::eng"
        except Exception:
            pass
        return "xkb:us::eng"

    def _release_ibus_engine(self) -> None:
        """Point ibus's GLOBAL engine back at the user's layout when dormant.

        Restoring the GNOME input source is not enough: the spawn dance sets
        the global engine directly (`ibus engine voiceio`), and a gsettings
        write that doesn't change the value never makes GNOME push an engine
        change. The stale global engine then routes EVERY input context
        through voiceio while "dormant" — constant focus churn that discards
        any visible preedit, surviving daemon restarts.
        """
        from voiceio.typers.ibus import _ibus_env
        try:
            # Unconditional: querying first races the (async) activation —
            # a stale reading made the release skip itself and left voiceio
            # active. Going dormant means the user's engine should be the
            # global one; setting it is idempotent.
            fallback = self._ibus_engine_fallback()
            subprocess.run(
                ["ibus", "engine", fallback],
                capture_output=True, timeout=3, env=_ibus_env(),
            )
            log.debug("Released ibus global engine to %s", fallback)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _switch_gnome_input_source(self, engine_name: str) -> None:
        if not self.platform.is_gnome:
            return
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return
            sources = result.stdout.strip()
            if f"('ibus', '{engine_name}')" not in sources:
                return
            import ast
            try:
                source_list = ast.literal_eval(sources)
            except (ValueError, SyntaxError):
                return
            for i, (kind, name) in enumerate(source_list):
                if kind == "ibus" and name == engine_name:
                    # Record the source the user actually had so we can restore
                    # exactly it, and remember voiceio's index to detect a
                    # mid-recording switch-away.
                    prev = self._get_current_input_source_index()
                    if prev is not None and prev != i:
                        self._prev_input_source_index = prev
                    self._voiceio_source_index = i
                    subprocess.run(
                        ["gsettings", "set", "org.gnome.desktop.input-sources",
                         "current", str(i)],
                        capture_output=True, timeout=3,
                    )
                    # The gsettings write alone is a no-op on newer GNOME
                    # (the 'current' key is legacy) — activation only ever
                    # worked while the global engine was accidentally stuck
                    # on voiceio. Claim it explicitly, mirroring
                    # _release_ibus_engine() on deactivation.
                    from voiceio.typers.ibus import _ibus_env
                    subprocess.run(
                        ["ibus", "engine", engine_name],
                        capture_output=True, timeout=3, env=_ibus_env(),
                    )
                    log.info("Switched GNOME input source to index %d (%s)", i, engine_name)
                    time.sleep(0.5)
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _stop_ibus_engine(self) -> None:
        self._set_gnome_input_source_index(0)
        if self._engine_proc is not None:
            self._engine_proc.terminate()
            try:
                self._engine_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._engine_proc.kill()
            self._engine_proc = None
            log.debug("Stopped IBus engine process")
        from voiceio.ibus import SOCKET_PATH
        SOCKET_PATH.unlink(missing_ok=True)
        if self._prev_ibus_engine:
            from voiceio.typers.ibus import _ibus_env
            try:
                subprocess.run(
                    ["ibus", "engine", self._prev_ibus_engine],
                    capture_output=True, timeout=3, env=_ibus_env(),
                )
                log.debug("Restored IBus engine: %s", self._prev_ibus_engine)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            self._prev_ibus_engine = None

    def _set_gnome_input_source_index(self, index: int) -> None:
        if not self.platform.is_gnome:
            return
        try:
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.input-sources",
                 "current", str(index)],
                capture_output=True, timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ── Health watchdog ─────────────────────────────────────────────

    # If the gap between health checks exceeds this, we probably resumed
    # from suspend/hibernate and should re-probe everything.
    _RESUME_THRESHOLD = 30  # seconds

    def _health_loop(self) -> None:
        """Periodic health check: transcriber worker, IBus engine, audio stream."""
        last_check = time.monotonic()
        while not self._shutdown.is_set():
            self._shutdown.wait(_HEALTH_CHECK_INTERVAL)
            if self._shutdown.is_set():
                break

            now = time.monotonic()
            gap = now - last_check
            last_check = now

            try:
                if gap > self._RESUME_THRESHOLD:
                    log.info("System resume detected (%.0fs gap), re-probing all backends", gap)
                    self._on_resume()
                self._check_health()
            except Exception:
                log.debug("Health check error", exc_info=True)

    def _on_resume(self) -> None:
        """Re-probe all backends after system suspend/hibernate.

        Sleep/hibernate breaks connections to system services (IBus, audio,
        tray D-Bus, ydotoold) across all platforms. Instead of catching each
        failure individually, do a single sweep to restore everything.
        """
        _import_graphical_env()

        # Re-detect platform now that env vars may have been refreshed
        # (lock-guarded, IDLE-only — a live recording keeps its platform).
        self._swap_platform(_redetect_platform())

        # Audio: reopen stream (device may have changed or died)
        try:
            self.recorder.reopen_stream()
            log.info("Resume: audio stream reopened")
        except Exception:
            log.warning("Resume: audio stream reopen failed", exc_info=True)

        # Typer: re-probe if on a fallback backend (e.g. clipboard instead
        # of ibus) — the preferred backend may be available now.
        self._try_upgrade_typer(reason="resume")

        # IBus: an engine that died during sleep is respawned lazily at the
        # next record start (same as health-loop reaping — respawning here
        # would grab the input source right as the user unlocks).
        if self._typer.name == "ibus" and self._engine_proc is not None:
            if self._engine_proc.poll() is not None:
                log.debug("Resume: IBus engine died during sleep — will respawn at next recording")
                self._engine_proc = None

        # Tray: restart if subprocess died
        if self.cfg.tray.enabled and not tray.is_alive():
            log.info("Resume: restarting tray")
            tray.restart(self.on_hotkey)

        # Transcriber: ensure worker is alive
        if not self.transcriber.is_worker_alive():
            try:
                self.transcriber._ensure_worker()
                log.info("Resume: transcriber worker restarted")
            except RuntimeError:
                log.error("Resume: transcriber worker failed")

    def _check_health(self) -> None:
        """Run one health check cycle."""
        # Check transcriber worker
        if not self.transcriber.is_worker_alive():
            if self._state == _State.RECORDING:
                log.warning("Transcriber worker died during recording")
            try:
                self.transcriber._ensure_worker()
                log.info("Transcriber worker recovered")
            except RuntimeError:
                log.error("Transcriber worker permanently failed")
                with self._hotkey_lock:
                    if self._state != _State.RECORDING:
                        self._state = _State.ERROR
                        tray.set_error(True)

        # Check audio stream (covers ALSA underrun, PulseAudio/PipeWire
        # restart, device disconnect, stale callback heartbeat)
        ok, reason = self.recorder.stream_health()
        if not ok:
            now = time.monotonic()
            if now < self._next_stream_retry:
                return  # backoff: skip this cycle
            log.warning("Audio stream unhealthy: %s — reopening (attempt %d)",
                        reason, self._stream_fail_count + 1)
            try:
                self.recorder.reopen_stream()
                self._stream_fail_count = 0
                self._next_stream_retry = 0
                tray.set_error(False)
                log.info("Audio stream recovered")
            except Exception:
                self._stream_fail_count += 1
                # Backoff: 10s, 20s, 40s, 80s, max 5min
                delay = min(10 * (2 ** (self._stream_fail_count - 1)), 300)
                self._next_stream_retry = now + delay
                tray.set_error(True)
                log.error("Audio stream recovery failed (retry in %ds)", delay)
        elif self._stream_fail_count > 0:
            # Stream recovered externally (e.g. device plugged back in)
            self._stream_fail_count = 0
            self._next_stream_retry = 0
            tray.set_error(False)

        # Check tray subprocess (restart if died / lost D-Bus registration)
        if self.cfg.tray.enabled and not tray.is_alive():
            log.warning("Tray subprocess died, restarting")
            tray.restart(self.on_hotkey)

        # Typer upkeep: only touch typer/platform while IDLE, and only through
        # the lock-guarded swap helpers (never hot-swap under a live recording).
        if self._state == _State.IDLE:
            self._health_typer_upkeep()

        # Check IBus engine liveness. A dormant engine dying is NORMAL:
        # ibus-daemon reaps deactivated engine processes (SIGTERM ~10s after
        # the input source switches away). Eagerly respawning here created a
        # spawn→activate→dormant→reap loop that grabbed the user's input
        # source every cycle — the focus churn wiped any visible preedit
        # (flickering underline). Respawn lazily at the next record start.
        if self._typer.name == "ibus" and self._engine_proc is not None:
            if self._engine_proc.poll() is not None:
                rc = self._engine_proc.returncode
                self._engine_proc = None
                self._engine_ping_fails = 0
                if self._state == _State.IDLE:
                    log.debug(
                        "IBus engine exited while dormant (rc=%s) — normal "
                        "reaping, will respawn at next recording", rc,
                    )
                else:
                    log.warning("IBus engine died mid-recording (rc=%s), restarting", rc)
                    try:
                        self._ensure_ibus_engine()
                        log.info("IBus engine recovered")
                    except Exception:
                        log.exception("IBus engine recovery failed")
            elif self._state == _State.IDLE:
                # Engine process alive — check if its socket listener is
                # actually responding. A zombie engine (process alive but
                # GLib loop stuck / socket thread dead) silently drops all
                # preedit and commit messages. One missed ping on a loaded
                # system is not a zombie — require two consecutive misses.
                if not self._ping_ibus_engine():
                    self._engine_ping_fails += 1
                    if self._engine_ping_fails < 2:
                        log.debug(
                            "IBus engine missed a ping (%d/2) — busy system?",
                            self._engine_ping_fails,
                        )
                        return
                    log.warning("IBus engine not responding to ping, restarting")
                    self._engine_ping_fails = 0
                    self._engine_proc.kill()
                    try:
                        self._engine_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
                    self._engine_proc = None
                    try:
                        self._ensure_ibus_engine()
                        log.info("IBus engine recovered (was zombie)")
                    except Exception:
                        log.exception("IBus engine recovery failed")
                else:
                    self._engine_ping_fails = 0
            else:
                # RECORDING / FINALIZING: don't re-force the source; if the
                # user switched away from voiceio, fall back to clipboard.
                self._detect_input_source_hijack()

    def _health_typer_upkeep(self) -> None:
        """IDLE-only: refresh platform + upgrade/repair the typer.

        Runs the expensive probing outside ``_hotkey_lock``; the platform and
        typer field mutations go through the lock-guarded swap helpers.
        """
        _import_graphical_env()
        old_desktop = self.platform.desktop
        new_platform = _redetect_platform()
        if new_platform.desktop != old_desktop:
            log.info(
                "Platform refreshed: display=%s desktop=%s (was desktop=%s)",
                new_platform.display_server, new_platform.desktop, old_desktop,
            )
        self._swap_platform(new_platform)

        chain = typer_chain._get_chain(self.platform)
        current_idx = chain.index(self._typer.name) if self._typer.name in chain else len(chain)
        if current_idx > 0:
            self._try_upgrade_typer(reason="health-check")

        # Re-probe current backend; re-select if it broke.
        probe = self._typer.probe()
        if not probe.ok:
            log.warning("Typer '%s' probe failed: %s — re-selecting", self._typer.name, probe.reason)
            self._try_upgrade_typer(reason="probe-failed")

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        from voiceio.config import PID_PATH, LOG_DIR

        # Refresh env and platform in case the desktop session exported
        # vars between __init__ and run() (e.g. boot race with user@.service).
        _import_graphical_env()
        old_platform = self.platform
        self.platform = _redetect_platform()
        if (self.platform.display_server != old_platform.display_server
                or self.platform.desktop != old_platform.desktop):
            log.info(
                "Platform changed since init: display=%s desktop=%s (was %s/%s), re-selecting typer",
                self.platform.display_server, self.platform.desktop,
                old_platform.display_server, old_platform.desktop,
            )
            self._try_upgrade_typer(reason="run-init")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._pid_fd = open(PID_PATH, "w")
        try:
            from voiceio.pidlock import lock_pid_file
            lock_pid_file(self._pid_fd)
        except (BlockingIOError, OSError):
            self._pid_fd.close()
            log.error("Another voiceio instance is already running")
            print("voiceio is already running. Stop it first: voiceio service stop")
            return
        self._pid_fd.write(str(os.getpid()))
        self._pid_fd.flush()

        if self._typer.name == "ibus":
            # Daemon-start is one of the only places allowed to install the
            # component / restart IBus (probe() is read-only now, fix #6).
            try:
                if hasattr(self._typer, "ensure_installed"):
                    self._typer.ensure_installed()
            except Exception:
                log.exception("IBus ensure_installed failed (continuing)")
            self._ensure_ibus_engine()
        else:
            # IBus daemon may not be ready at startup (race with graphical
            # session). If IBus is in the preferred chain but wasn't selected,
            # wait briefly and re-probe. Re-fetch the chain after env refresh
            # above so we don't use a stale "clipboard-only" chain from when
            # platform=unknown.
            chain = typer_chain._get_chain(self.platform)
            if "ibus" in chain and self._typer.name != "ibus":
                self._typer = self._wait_for_ibus(chain) or self._typer

        self.recorder.open_stream()

        # Pre-open sound output stream so first cue plays instantly
        if self.cfg.feedback.sound_enabled:
            from voiceio.feedback import warm_up
            warm_up()

        if self.cfg.tray.enabled:
            tray.start(self.request_shutdown, self.on_hotkey)

        self._hotkey.start(self.cfg.hotkey.key, self.on_hotkey)
        if self._socket is not None:
            self._socket.start(self.cfg.hotkey.key, self.on_hotkey)

        if self._tts_hotkey:
            self._tts_hotkey.start(self.cfg.tts.hotkey, self.on_tts_hotkey)

        # Start health watchdog
        threading.Thread(
            target=self._health_loop, daemon=True, name="health-watchdog",
        ).start()

        from voiceio import __version__
        log.info(
            "voiceio v%s ready. Press [%s] to toggle recording (hotkey=%s, typer=%s)",
            __version__, self.cfg.hotkey.key, self._hotkey.name, self._typer.name,
        )
        print(
            f"voiceio v{__version__} ready. Press [{self.cfg.hotkey.key}] to record "
            f"(model={self.cfg.model.name}, typer={self._typer.name})",
        )

        # Ctrl-C and `systemctl stop voiceio` must run the SAME cleanup path
        # (restore input source, engine shutdown) — the finally block below.
        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
        try:
            self._shutdown.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self._hotkey.stop()
            if self._socket is not None:
                self._socket.stop()
            if self._tts_hotkey:
                self._tts_hotkey.stop()
            if self._tts_player:
                self._tts_player.cancel()
            if self._tts_engine:
                self._tts_engine.shutdown()
            self.recorder.close_stream()
            self.transcriber.shutdown()
            tray.stop()
            self._stop_ibus_engine()
            self._pid_fd.close()
            PID_PATH.unlink(missing_ok=True)
        log.info("voiceio stopped")
