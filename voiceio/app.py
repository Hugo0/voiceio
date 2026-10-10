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
from voiceio.core.decoders.base import Bias
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.hotkeys.socket_backend import SocketHotkey
from voiceio.recorder import AudioRecorder
from voiceio.streaming import StreamingSession
from voiceio.core.decoders import build_decoder
from voiceio.transcriber import describe_transcript
from voiceio.typers import chain as typer_chain
from voiceio.vad import load_vad

if TYPE_CHECKING:
    from voiceio.typers.base import TyperBackend
log = logging.getLogger("voiceio")
_DEBOUNCE_SECS = 0.8

# Env vars needed for clipboard/tray/typing on graphical sessions.
# XDG_CURRENT_DESKTOP is required for is_gnome() detection, which gates
# IBus input source configuration.
_GRAPHICAL_ENV_VARS = (
    "DISPLAY",
    # Without the X auth cookie, X11 tools (xclip via the XWayland bridge)
    # fail silently inside the daemon and clipboard copies fall back to
    # wl-copy — whose focus steal discards any visible IBus preedit.
    "XAUTHORITY",
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
        elif self.platform.desktop not in ("gnome", "kde"):
            # Socket backend has no physical hotkey listener — user must bind
            # `voiceio toggle` to a key in their WM config.
            from voiceio.hotkeys.evdev import access_hint as evdev_access_hint
            log.warning(
                "Hotkey backend is 'socket' on desktop '%s'. "
                "Physical hotkey capture is not available. "
                "Bind 'voiceio toggle' to a key in your window manager config "
                "(e.g. i3: bindsym Ctrl+Alt+v exec voiceio toggle). "
                "Or enable evdev hotkeys: %s",
                self.platform.desktop, evdev_access_hint(),
            )
            print(
                f"\n  NOTE: No physical hotkey listener available on '{self.platform.desktop}'.\n"
                f"  Bind 'voiceio toggle' to a key in your WM config, e.g.:\n"
                f"    i3/sway:  bindsym Ctrl+Alt+v exec voiceio toggle\n"
                f"    hyprland: bind = CTRL ALT, V, exec, voiceio toggle\n"
                f"  Or enable evdev hotkeys: {evdev_access_hint()}\n",
            )

        print(f"Loading model '{cfg.model.name}'...", end="", flush=True)
        t0 = time.monotonic()
        self.transcriber = build_decoder(cfg.model)
        print(f" ready ({time.monotonic() - t0:.1f}s)")

        # VAD, vocabulary, commands
        vad = load_vad(cfg.audio)
        self.recorder = AudioRecorder(cfg.audio, vad=vad)

        self._recording_bias = Bias()

        self._command_processor = CommandProcessor(enabled=cfg.commands.enabled, editing=cfg.commands.editing)
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

        # The shared engine: this decoder plus the user's vocabulary,
        # corrections, prompt history and text pipeline (same as the server).
        from voiceio.core.engine import Engine
        self.engine = Engine.from_config(
            cfg, decoder=self.transcriber, commands=self._command_processor,
            postcorrect=self._postcorrect, llm=self._llm,
        )
        self._corrections = self.engine.corrections
        self._pipeline = self.engine.pipeline

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
                self._tts_hotkey = hotkey_chain.select(self.platform, cfg.hotkey.backend)
                log.info("TTS: engine=%s, hotkey=%s", self._tts_engine.name, cfg.tts.hotkey)
            else:
                log.warning("TTS enabled but no engine available")

        self._shutdown = threading.Event()
        # IBus engine process + input-source claim (only while recording).
        from voiceio.ibus.session import IBusSession
        self._ibus = IBusSession(
            lambda: self.platform,
            is_idle=lambda: self._state == _State.IDLE,
            generation=lambda: self._recording,
        )

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
                self.transcriber.recover()
                self._state = _State.IDLE
                tray.set_error(False)
                self._do_start()
            except Exception:
                log.exception("Recovery failed")
        elif self._state in (_State.IDLE, _State.FINALIZING):
            # Allow starting a new recording even while old one finalizes.
            # The generation counter ensures the old finalizer exits cleanly.
            self._do_start()

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
        if self._typer.name == "ibus":
            self._ibus.claim(self._recording)
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
                self.recorder, self._typer, bias=bias,
                generation=self._generation,
                on_typer_broken=self._on_typer_broken,
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

        if self._postcorrect is not None:
            # Freshest context for the finalize pass: this recording's window
            # title (captured at start) and the latest transcripts/vocabulary.
            #
            # The FULL vocabulary, not the hotwords string: the decoder can only
            # be biased with ~40-60 terms, but this is an LLM with no 448-token
            # budget. The long tail that can never fit in hotwords gets its only
            # chance to be corrected here.
            self._postcorrect.set_context(
                vocabulary=", ".join(self.engine.vocabulary_terms),
                recent=self.engine.recent(3),
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
                args=(session, audio, elapsed, gen, rid),
                daemon=True,
            ).start()
        elif not self._streaming:
            self._state = _State.IDLE
            log.info("Stopped recording (%.1fs)", elapsed)
            if audio is not None:
                # _process owns deactivation (its finally block).
                threading.Thread(
                    target=self._process,
                    args=(audio, gen, rid, self._recording_bias),
                    daemon=True,
                ).start()
            else:
                # No processing thread will run — release the source here.
                self._ibus.release(rid)
        else:
            # Streaming mode but session already gone (race) — nothing to do
            self._state = _State.IDLE
            log.debug("Stop: streaming session already finalized")
            # No finalizer will run to release the input source, so do it here.
            self._ibus.release(rid)

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
        elapsed: float, gen: int, rid: int,
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
            # A later recording has already stopped: this output is stale, but
            # the user's words must not vanish — keep them in history.
            log.info("Finalize superseded (gen %d, current %d): %s saved to history only",
                     gen, self._generation, describe_transcript(final_text))
            if final_text:
                from voiceio import history
                history.append(self._strip_voice_prefix(final_text),
                               raw=session.raw_final_text, duration=elapsed)
            self._ibus.release(rid)
            return
        if final_text:
            self._copy_result_async(final_text)
            self._play_feedback(final_text)
            stored = self._strip_voice_prefix(final_text)
            self.engine.record_usage(stored)
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
        log.info("Streaming done (%.1fs): %s", elapsed, describe_transcript(final_text))
        log.debug("Final text: %r", final_text)
        self._warn_if_nothing_captured(final_text, gen)
        # Release the IBus input source now that the final commit is done.
        # A no-op if a newer recording has claimed it since.
        self._ibus.release(rid)
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
            t0 = time.monotonic()
            result = self.transcriber.decode(audio, final=True, bias=bias or Bias())
            raw = result.text
            latency = {
                "audio_secs": round(len(audio) / self.recorder.sample_rate, 2),
                "transcribe": round(time.monotonic() - t0, 3),
            }
            segments = list(result.segments)
            text = raw
            if self._generation != gen:
                return
            if text:
                t1 = time.monotonic()
                processed = self._pipeline(text, final=True)
                text, abort = processed.text, processed.abort
                latency["pipeline"] = round(time.monotonic() - t1, 3)
                if "postcorrect" in processed.timings:
                    latency["postcorrect"] = processed.timings["postcorrect"]
                extra["latency"] = latency
                if abort:
                    return
                # Superseded by a newer recording — do not touch the typer.
                if text and self._generation == gen:
                    self._type_with_fallback(text)
                    self._copy_result_async(text)
                    self._play_feedback(text)
                    stored = self._strip_voice_prefix(text)
                    self.engine.record_usage(stored)
                    from voiceio import history
                    history.append(
                        stored,
                        raw=raw,
                        segments=segments,
                        duration=len(audio) / self.recorder.sample_rate,
                        extra=extra,
                    )
                    log.info("Typed: %s", describe_transcript(text))
                    log.debug("Typed text: %r", text)
        except Exception:
            log.exception("Processing failed")
        finally:
            # Release the IBus input source after the batch commit (or failure).
            self._ibus.release(rid)

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
            if new_typer.name == "ibus" and not self._ibus.has_engine:
                try:
                    # subprocess ops, outside the lock; IDLE → never activate
                    self._ibus.ensure_engine(activate=False)
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

    def _wait_for_ibus(self) -> TyperBackend | None:
        """At login voiceio may start before ibus-daemon: wait briefly and
        switch to the IBus typer instead of settling on a broken fallback."""
        if self._ibus.wait_for_daemon():
            from voiceio.typers.ibus import refresh_stale_launcher
            refresh_stale_launcher()
            try:
                typer = typer_chain.select(self.platform, self.cfg.output.method)
                if typer.name == "ibus":
                    self._ibus.ensure_engine(activate=False)
                    return typer
            except RuntimeError:
                pass
        log.warning("IBus daemon did not start in time, using fallback typer: %s", self._typer.name)
        return None

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

        if self._typer.name == "ibus":
            self._ibus.on_resume()

        # Tray: restart if subprocess died
        if self.cfg.tray.enabled and not tray.is_alive():
            log.info("Resume: restarting tray")
            tray.restart(self.on_hotkey)

        # Transcriber: ensure worker is alive
        if not self.transcriber.healthy():
            try:
                self.transcriber.recover()
                log.info("Resume: transcriber worker restarted")
            except RuntimeError:
                log.error("Resume: transcriber worker failed")

    def _check_health(self) -> None:
        """Run one health check cycle."""
        # Check transcriber worker
        if not self.transcriber.healthy():
            if self._state == _State.RECORDING:
                log.warning("Transcriber worker died during recording")
            try:
                self.transcriber.recover()
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

        if self._typer.name == "ibus":
            idle = self._state == _State.IDLE
            self._ibus.health_tick(idle)
            # Mid-recording: don't re-force the source; if the user switched
            # away from voiceio, fall back to clipboard for this session.
            if not idle and self._ibus.hijacked():
                self._fall_back_to_clipboard()

    def _fall_back_to_clipboard(self) -> None:
        try:
            from voiceio.typers.clipboard import ClipboardTyper
            clip = ClipboardTyper(self.platform)
        except Exception:
            log.debug("Could not build clipboard fallback typer", exc_info=True)
            return
        session = self._session
        if session is not None:
            session.set_typer(clip)

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
            # Register the engine but never claim the input source at boot —
            # the instance is created on first activation (record start).
            self._ibus.ensure_engine(activate=False)
        else:
            # IBus daemon may not be ready at startup (race with graphical
            # session). If IBus is in the preferred chain but wasn't selected,
            # wait briefly and re-probe. Re-fetch the chain after env refresh
            # above so we don't use a stale "clipboard-only" chain from when
            # platform=unknown.
            chain = typer_chain._get_chain(self.platform)
            if "ibus" in chain and self._typer.name != "ibus":
                self._typer = self._wait_for_ibus() or self._typer

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
            self.transcriber.close()
            tray.stop()
            self._ibus.shutdown()
            self._pid_fd.close()
            PID_PATH.unlink(missing_ok=True)
        log.info("voiceio stopped")
