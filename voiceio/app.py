"""Main VoiceIO engine: explicit state machine with generation-based cancellation."""
from __future__ import annotations

import enum
import logging
import os
import signal
import threading
import time

import numpy as np

from voiceio import config, platform as plat, tray
from voiceio.commands import CommandProcessor
from voiceio.core.decoders.base import Bias
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.hotkeys.socket_backend import SocketHotkey
from voiceio.recorder import AudioRecorder
from voiceio.streaming import StreamingSession
from voiceio.core.decoders import build_decoder
from voiceio.privacy import describe_transcript
from voiceio.typers import chain as typer_chain, typer_options
from voiceio.typers.manager import TyperManager
from voiceio.vad import load_vad
from voiceio.watchdog import Watchdog

log = logging.getLogger("voiceio")
_DEBOUNCE_SECS = 0.8


class _State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class VoiceIO:
    @property
    def _state(self) -> _State:
        return self.__state

    @_state.setter
    def _state(self, value: _State) -> None:
        self.__state = value
        # Only the instance holding the PID lock speaks for the daemon: a
        # second launch must not overwrite the running one's state.
        if not getattr(self, "_publishes_state", False):
            return
        try:   # best-effort: only `service restart` reads it
            tmp = config.STATE_PATH.with_suffix(".tmp")
            tmp.write_text(value.value)
            tmp.replace(config.STATE_PATH)
        except OSError:
            pass

    def __init__(self, cfg: config.Config):
        self.cfg = cfg

        # Import graphical env vars early so typer/platform detection
        # sees DISPLAY, WAYLAND_DISPLAY, XDG_SESSION_TYPE, XDG_CURRENT_DESKTOP
        # even when started as a systemd user service before the desktop
        # session has exported them.
        plat.import_graphical_env()

        # Fresh detection (clears lru_cache in case anything warmed it)
        platform = plat.redetect()
        log.info(
            "Platform: display=%s desktop=%s",
            platform.display_server, platform.desktop,
        )

        # Select backends
        self._hotkey = hotkey_chain.select(platform, cfg.hotkey.backend)
        typer = typer_chain.select(platform, cfg.output.method, **typer_options(cfg))

        # Socket backend runs alongside native hotkey for extra robustness
        self._socket: SocketHotkey | None = None
        if self._hotkey.name != "socket":
            self._socket = SocketHotkey()
        elif platform.desktop not in ("gnome", "kde"):
            # Socket backend has no physical hotkey listener — user must bind
            # `voiceio toggle` to a key in their WM config.
            from voiceio.hotkeys.evdev import access_hint as evdev_access_hint
            log.warning(
                "Hotkey backend is 'socket' on desktop '%s'. "
                "Physical hotkey capture is not available. "
                "Bind 'voiceio toggle' to a key in your window manager config "
                "(e.g. i3: bindsym Ctrl+Alt+v exec voiceio toggle). "
                "Or enable evdev hotkeys: %s",
                platform.desktop, evdev_access_hint(),
            )
            print(
                f"\n  NOTE: No physical hotkey listener available on '{platform.desktop}'.\n"
                f"  Bind 'voiceio toggle' to a key in your WM config, e.g.:\n"
                f"    i3/sway:  bindsym Ctrl+Alt+v exec voiceio toggle\n"
                f"    hyprland: bind = CTRL ALT, V, exec, voiceio toggle\n"
                f"  Or enable evdev hotkeys: {evdev_access_hint()}\n",
            )

        print(f"Loading model '{cfg.model.name}'...", end="", flush=True)
        t0 = time.monotonic()
        self.decoder = build_decoder(cfg.model)
        print(f" ready ({time.monotonic() - t0:.1f}s)")

        # VAD, vocabulary, commands
        vad = load_vad(cfg.audio)
        self.recorder = AudioRecorder(cfg.audio, vad=vad)

        self._recording_bias = Bias()

        self._command_processor = CommandProcessor(enabled=cfg.commands.enabled, editing=cfg.commands.editing)
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
                log.info("PostCorrect: %s", cfg.postcorrect.model or cfg.cloud.model)
            else:
                log.warning("PostCorrect enabled but no API key resolved (disabled)")

        # The shared engine: this decoder plus the user's vocabulary,
        # corrections, prompt history and text pipeline (same as the server).
        from voiceio.core.engine import Engine
        self.engine = Engine.from_config(
            cfg, decoder=self.decoder, commands=self._command_processor,
            postcorrect=self._postcorrect, llm=self._llm,
        )
        self._corrections = self.engine.corrections

        # Explicit state machine
        self._state = _State.IDLE
        # Output ownership: bumped on stop, so a finalizer whose recording was
        # followed by another stop knows its output is stale. A recording
        # started while the previous one finalizes does NOT supersede it —
        # that finalizer still commits its text.
        self._generation = 0
        # Input-source ownership: bumped on start. A finalizer releases the
        # IBus claim only if no newer recording has claimed it since.
        self._recording = 0
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
                self._tts_hotkey = hotkey_chain.select(platform, cfg.hotkey.backend)
                log.info("TTS: engine=%s, hotkey=%s", self._tts_engine.name, cfg.tts.hotkey)
            else:
                log.warning("TTS enabled but no engine available")

        self._shutdown = threading.Event()
        # IBus engine process + input-source claim (only while recording).
        from voiceio.ibus.session import IBusSession
        self._ibus = IBusSession(
            lambda: self.platform,
            is_idle=self._is_idle,
            generation=lambda: self._recording,
        )
        # Batch mode is IDLE again before its thread types the text; a typer
        # swap must also wait for output in flight (a mid-commit swap to IBus
        # would commit through a source never claimed for that recording).
        self._outputs = 0
        self._outputs_lock = threading.Lock()
        self._typers = TyperManager(
            platform, typer,
            ibus=self._ibus,
            method=cfg.output.method,
            auto_fallback=cfg.health.auto_fallback,
            lock=self._hotkey_lock,
            is_idle=lambda: self._is_idle() and not self._outputs,
            state=lambda: self._state,
            shutdown=self._shutdown,
            options=typer_options(cfg),
        )
        self._watchdog = Watchdog(
            cfg,
            decoder=self.decoder,
            recorder=self.recorder,
            typers=self._typers,
            ibus=self._ibus,
            session=lambda: self._session,
            is_idle=self._is_idle,
            is_recording=lambda: self._state == _State.RECORDING,
            on_decoder_failed=self._on_decoder_failed,
            on_hotkey=self.on_hotkey,
            shutdown=self._shutdown,
        )

    @property
    def platform(self) -> plat.Platform:
        return self._typers.platform

    def _is_idle(self) -> bool:
        return self._state == _State.IDLE

    def _output_thread(self, target, *args) -> None:
        """Run a finalizer that types text, counted as output in flight."""
        def run() -> None:
            with self._outputs_lock:
                self._outputs += 1
            try:
                target(*args)
            finally:
                with self._outputs_lock:
                    self._outputs -= 1
        threading.Thread(target=run, daemon=True).start()

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

    def _on_auto_stop(self, reason: str = "silence") -> None:
        """Called from the audio thread when the recorder auto-stops.

        reason="silence": spoke, then went quiet — a normal end of dictation.
        reason="no_speech": nothing was heard the whole time — a muted mic or a
        hotkey pressed by accident. Surface it so it isn't a silent failure
        (today's Discord-mute confusion: recording ran, nothing was captured,
        and there was no sign why).
        """
        if reason == "no_speech":
            from voiceio import feedback
            if not self.recorder.has_signal():
                feedback.notify(
                    "VoiceIO: microphone is muted",
                    "Stopped — no audio is reaching voiceio. Check your mic's "
                    "mute button and that the right input device is selected.",
                )
            else:
                feedback.notify(
                    "VoiceIO: stopped — no speech detected",
                    "Recording auto-stopped after silence.",
                )
        threading.Thread(target=self._request_stop, daemon=True).start()

    def _warn_if_nothing_captured(self, final_text: str, gen: int) -> None:
        """Warn when a finished recording captured no audio at all.

        This is the moment the user actually notices — they look for their text
        and nothing appeared. The pre-flight warning is a 3s toast they talk
        straight through, and the 20s no-speech auto-stop never fires on a
        shorter take stopped by hand, which is exactly how a muted mic slips
        by silently. Generation-guarded: a newer recording owns the recorder,
        so never read its state to judge this one.
        """
        if self._generation != gen:
            return
        if final_text.strip() or self.recorder.heard_signal:
            return
        log.warning("Recording captured no audio at all — mic muted?")
        from voiceio import feedback
        feedback.notify(
            "VoiceIO: microphone is muted — nothing recorded",
            "No audio reached voiceio for that entire recording. Check your "
            "mic's mute button and which input device is selected.",
            urgent=True,
        )

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
                self.decoder.recover()
                self._state = _State.IDLE
                tray.set_error(False)
                self._do_start()
            except Exception:
                log.exception("Recovery failed")
        elif self._state in (_State.IDLE, _State.FINALIZING):
            # Allow starting a new recording even while old one finalizes.
            # The generation counter ensures the old finalizer exits cleanly.
            self._do_start()

    def _on_decoder_failed(self) -> None:
        """Watchdog: the worker could not be recovered."""
        with self._hotkey_lock:
            if self._state != _State.RECORDING:
                self._state = _State.ERROR
                tray.set_error(True)

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
                    urgent=True,
                )
                return
            if not self.recorder.has_signal():
                log.warning("Mic still silent after reopen — muted or wrong device?")
                from voiceio import feedback
                feedback.notify(
                    "VoiceIO: microphone appears silent",
                    "Check that your mic is unmuted and the right input device is selected.",
                    urgent=True,
                )
        elif not self.recorder.has_signal():
            log.warning("Mic appears silent or muted (pre-buffer is all zeros)")

        self._state = _State.RECORDING
        self._recording += 1
        self._typers.begin(self._recording)
        # Vocabulary and corrections are mtime-cached, so edits made with
        # `voiceio vocab`/`corrections` apply from the next recording.
        bias = self.engine.bias()
        self._recording_bias = bias
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
            self._session = self.engine.session(
                self.recorder, self._typers.typer, bias=bias,
                generation=self._generation,
                on_typer_broken=self._typers.on_broken,
                on_interim=self._on_interim_text,
            )
            self._session.start()
        log.info("Recording... press [%s] again to stop", self.cfg.hotkey.key)

    def _do_stop(self) -> None:
        """Transition from RECORDING to FINALIZING (or IDLE for batch)."""
        elapsed = time.monotonic() - self._record_start
        self._generation += 1
        gen = self._generation
        rid = self._recording

        # Stop audio capture immediately — chime and recorder stop are synchronous
        self._play_record_cue(start=False)
        self.recorder.set_on_auto_stop(None)
        audio = self.recorder.stop()
        tray.set_recording(False)
        self._warn_if_clipping()

        self.engine.prepare_final(title=self._context_title)

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
            self._output_thread(self._finalize_streaming, session, audio, elapsed, gen, rid)
        elif not self._streaming:
            self._state = _State.IDLE
            log.info("Stopped recording (%.1fs)", elapsed)
            if audio is not None:
                # _process owns deactivation (its finally block).
                self._output_thread(self._process, audio, gen, rid, self._recording_bias)
            else:
                # No processing thread will run — release the source here.
                self._typers.end(rid)
        else:
            # Streaming mode but session already gone (race) — nothing to do
            self._state = _State.IDLE
            log.debug("Stop: streaming session already finalized")
            # No finalizer will run to release the input source, so do it here.
            self._typers.end(rid)

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
        elapsed: float, gen: int, rid: int,
    ) -> None:
        """Run final transcription and commit in background thread."""
        t_final = time.monotonic()
        # Snappy clipboard: make the best-so-far text pasteable the moment
        # the user stops, instead of after the (possibly long) final decode.
        if (self.cfg.output.copy_to_clipboard == "live"
                and not self._typers.pastes_from_clipboard and session.interim_text):
            from voiceio import clipboard_read
            clipboard_read.copy_text(session.interim_text)
        extra = self._retention_extra(audio)
        final_text = session.stop(audio)
        if self._generation != gen:
            # A later recording has already stopped: this output is stale, but
            # the user's words must not vanish — keep them in history.
            log.info("Finalize superseded (gen %d, current %d): %s saved to history only",
                     gen, self._generation, describe_transcript(final_text))
            if final_text:
                self.engine.commit(final_text, raw=session.raw_final_text,
                                   duration=elapsed, extra=extra)
            self._typers.end(rid)
            return
        if final_text:
            self._copy_result_async(final_text)
            self._play_feedback(final_text)
            latency = dict(session.final_latency)
            # stop-to-commit: what the user actually waits for
            latency["finalize_total"] = round(time.monotonic() - t_final, 3)
            extra["latency"] = latency
            self.engine.commit(final_text, raw=session.raw_final_text,
                               segments=session.final_segments, duration=elapsed, extra=extra)
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
        log.info("Streaming done (%.1fs): %s", elapsed, describe_transcript(final_text))
        self._warn_if_nothing_captured(final_text, gen)
        # Release the IBus input source now that the final commit is done.
        # A no-op if a newer recording has claimed it since.
        self._typers.end(rid)
        # Transition to IDLE under lock to avoid racing with _toggle
        with self._hotkey_lock:
            if self._generation == gen and self._state == _State.FINALIZING:
                self._state = _State.IDLE
                tray.set_processing(False)

    def _process(self, audio: np.ndarray, gen: int, rid: int = 0,
                 bias: Bias | None = None) -> None:
        """Batch transcription (non-streaming mode)."""
        try:
            if self._generation != gen:
                return
            extra = self._retention_extra(audio)
            result = self.engine.transcribe(audio, bias=bias)
            # Superseded by a newer recording, or a voice command discarded
            # it: do not touch the typer.
            if self._generation != gen or result.abort or not result.text:
                return
            extra["latency"] = result.latency
            self._typers.type_with_fallback(result.text)
            self._copy_result_async(result.text)
            self._play_feedback(result.text)
            self.engine.commit(result.text, raw=result.raw, segments=result.segments,
                               duration=len(audio) / self.recorder.sample_rate, extra=extra)
            log.info("Typed: %s", describe_transcript(result.text))
        except Exception:
            log.exception("Processing failed")
        finally:
            # Release the IBus input source after the batch commit (or failure).
            self._typers.end(rid)

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
        if self._typers.pastes_from_clipboard:
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
            if self._typers.pastes_from_clipboard:
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

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        from voiceio.config import PID_PATH, LOG_DIR

        self._typers.redetect_at_start()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._pid_fd = open(PID_PATH, "a")   # not "w": the running daemon's PID stays
        try:
            from voiceio.pidlock import lock_pid_file
            lock_pid_file(self._pid_fd)
        except (BlockingIOError, OSError):
            self._pid_fd.close()
            log.error("Another voiceio instance is already running")
            print("voiceio is already running. Stop it first: voiceio service stop")
            return
        self._pid_fd.truncate(0)
        self._pid_fd.write(str(os.getpid()))
        self._pid_fd.flush()
        self._publishes_state = True
        self._state = self._state

        self._typers.prepare()

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

        self._watchdog.start()

        from voiceio import __version__
        log.info(
            "voiceio v%s ready. Press [%s] to toggle recording (hotkey=%s, typer=%s)",
            __version__, self.cfg.hotkey.key, self._hotkey.name, self._typers.typer.name,
        )
        print(
            f"voiceio v{__version__} ready. Press [{self.cfg.hotkey.key}] to record "
            f"(model={self.cfg.model.name}, typer={self._typers.typer.name})",
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
            self.decoder.close()
            tray.stop()
            self._ibus.shutdown()
            self._pid_fd.close()
            PID_PATH.unlink(missing_ok=True)
            config.STATE_PATH.unlink(missing_ok=True)
        log.info("voiceio stopped")
