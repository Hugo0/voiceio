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
from voiceio.vocabulary import load_vocabulary

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

        vocab = load_vocabulary(cfg.model)
        self._corrections = CorrectionDict()
        # Merge correction targets into vocabulary for Whisper conditioning
        vocab_terms = self._corrections.vocabulary_terms()
        if vocab_terms:
            extra = ", ".join(vocab_terms)
            vocab = f"{vocab}, {extra}" if vocab else extra
        if vocab:
            self.transcriber.set_initial_prompt(vocab)

        self._command_processor = CommandProcessor(enabled=cfg.commands.enabled, editing=cfg.commands.editing)
        self._cleanup = cfg.output.punctuation_cleanup
        self._number_conversion = cfg.output.number_conversion
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

        # Explicit state machine
        self._state = _State.IDLE
        self._generation = 0          # incremented on every stop; leaked threads check this
        self._session: StreamingSession | None = None
        self._record_start: float = 0

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
        self._shutdown = threading.Event()

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

        if not self.recorder.has_signal():
            log.warning("Mic appears silent or muted (pre-buffer is all zeros)")

        self._state = _State.RECORDING
        self._activate_ibus()
        self._corrections.load()  # hot-reload corrections on each recording
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
                llm=self._llm,
                on_typer_broken=self._on_typer_broken,
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

        if self._streaming and self._session is not None:
            self._state = _State.FINALIZING
            tray.set_processing(True)
            session = self._session
            self._session = None
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
                threading.Thread(
                    target=self._process,
                    args=(audio, gen),
                    daemon=True,
                ).start()
        else:
            # Streaming mode but session already gone (race) — nothing to do
            self._state = _State.IDLE
            log.debug("Stop: streaming session already finalized")

        self._deactivate_ibus()

    # ── Background work ─────────────────────────────────────────────────

    def _finalize_streaming(
        self, session: StreamingSession, audio: np.ndarray | None,
        elapsed: float, gen: int,
    ) -> None:
        """Run final transcription and commit in background thread."""
        final_text = session.stop(audio)
        if self._generation != gen:
            log.debug("Finalize cancelled (gen %d, current %d)", gen, self._generation)
            return
        if final_text:
            self._play_feedback(final_text)
            from voiceio import history
            history.append(final_text)
        log.info("Streaming done (%.1fs): '%s'", elapsed, final_text)
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
            text = self.transcriber.transcribe(audio)
            if self._generation != gen:
                return
            if text:
                from voiceio.postprocess import apply_pipeline
                text, abort = apply_pipeline(
                    text,
                    do_cleanup=self._cleanup,
                    number_conversion=self._number_conversion,
                    language=self.cfg.model.language,
                    commands=self._command_processor,
                    corrections=self._corrections,
                    llm=self._llm,
                    final=True,
                )
                if abort:
                    return
                if text:
                    self._type_with_fallback(text)
                    self._play_feedback(text)
                    from voiceio import history
                    history.append(text)
                    log.info("Typed: '%s'", text)
        except Exception:
            log.exception("Processing failed")

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
        """Switch GNOME input source to voiceio engine for text injection."""
        if self._typer.name != "ibus":
            return
        threading.Thread(
            target=self._switch_gnome_input_source,
            args=("voiceio",), daemon=True,
        ).start()

    def _deactivate_ibus(self) -> None:
        """Switch GNOME input source back to normal keyboard."""
        if self._typer.name != "ibus":
            return
        threading.Thread(
            target=self._set_gnome_input_source_index,
            args=(0,), daemon=True,
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
        """Wait for IDLE state, then re-detect platform and upgrade typer."""
        # Wait up to 30s for recording/finalizing to complete
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._state == _State.IDLE or self._shutdown.is_set():
                break
            time.sleep(0.5)
        if self._shutdown.is_set():
            return
        with self._hotkey_lock:
            if self._state != _State.IDLE:
                log.debug("Typer upgrade abandoned: state=%s after 30s", self._state)
                return
            _import_graphical_env()
            self.platform = _redetect_platform()
            log.info(
                "Typer broken: re-detected platform: display=%s desktop=%s",
                self.platform.display_server, self.platform.desktop,
            )
            self._try_upgrade_typer(reason="streaming-failure")

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
            old_name = self._typer.name
            self._typer = new_typer
            log.info("Typer upgraded: %s -> %s (%s)", old_name, new_typer.name, reason)
            if new_typer.name == "ibus" and self._engine_proc is None:
                try:
                    self._ensure_ibus_engine()
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

    def _ensure_ibus_engine(self) -> None:
        """Start the VoiceIO IBus engine and activate it."""
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

        self._set_gnome_input_source_index(0)
        log.info("VoiceIO IBus engine ready (dormant until recording)")

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
            sock.settimeout(1.0)
            sock.bind("")
            sock.sendto(b"ping", str(SOCKET_PATH))
            data, _ = sock.recvfrom(64)
            sock.close()
            return data == b"pong"
        except (OSError, _socket.timeout):
            return False

    def _reactivate_ibus_if_stale(self) -> None:
        """Re-activate IBus engine if it lost registration (e.g. after hibernate).

        After suspend/hibernate, the IBus daemon may forget about our engine.
        Check by querying the current active engine; if it's not 'voiceio',
        re-run ``ibus engine voiceio`` to re-register.
        """
        from voiceio.typers.ibus import _ibus_env
        try:
            result = subprocess.run(
                ["ibus", "engine"], capture_output=True, text=True,
                timeout=3, env=_ibus_env(),
            )
            current = result.stdout.strip() if result.returncode == 0 else ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if current == "voiceio":
            return  # still registered, nothing to do
        log.warning("IBus engine stale (current=%r), re-activating", current)
        try:
            subprocess.run(
                ["ibus", "engine", "voiceio"],
                capture_output=True, timeout=5, env=_ibus_env(),
            )
            log.info("IBus engine re-activated")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning("IBus re-activation failed: %s", e)

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
                    subprocess.run(
                        ["gsettings", "set", "org.gnome.desktop.input-sources",
                         "current", str(i)],
                        capture_output=True, timeout=3,
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
        self.platform = _redetect_platform()

        # Audio: reopen stream (device may have changed or died)
        try:
            self.recorder.reopen_stream()
            log.info("Resume: audio stream reopened")
        except Exception:
            log.warning("Resume: audio stream reopen failed", exc_info=True)

        # Typer: re-probe if on a fallback backend (e.g. clipboard instead
        # of ibus) — the preferred backend may be available now.
        self._try_upgrade_typer(reason="resume")

        # IBus: re-activate engine registration
        if self._typer.name == "ibus" and self._engine_proc is not None:
            if self._engine_proc.poll() is not None:
                # Engine process died during sleep
                self._engine_proc = None
                try:
                    self._ensure_ibus_engine()
                    log.info("Resume: IBus engine restarted")
                except Exception:
                    log.exception("Resume: IBus engine restart failed")
            else:
                self._reactivate_ibus_if_stale()

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

        # Check typer: if on a fallback, try to upgrade to preferred backend.
        # Refresh env + platform first in case we started before the desktop
        # session had exported DISPLAY/WAYLAND_DISPLAY/XDG_CURRENT_DESKTOP.
        _import_graphical_env()
        old_desktop = self.platform.desktop
        self.platform = _redetect_platform()
        if self.platform.desktop != old_desktop:
            log.info(
                "Platform refreshed: display=%s desktop=%s (was desktop=%s)",
                self.platform.display_server, self.platform.desktop, old_desktop,
            )
        chain = typer_chain._get_chain(self.platform)
        current_idx = chain.index(self._typer.name) if self._typer.name in chain else len(chain)
        if current_idx > 0:
            self._try_upgrade_typer(reason="health-check")

        # Check typer: re-probe if current backend is broken
        probe = self._typer.probe()
        if not probe.ok:
            log.warning("Typer '%s' probe failed: %s — re-selecting", self._typer.name, probe.reason)
            _import_graphical_env()
            self.platform = _redetect_platform()
            self._try_upgrade_typer(reason="probe-failed")

        # Check IBus engine (restart if died, zombie, or stale after resume)
        if self._typer.name == "ibus" and self._engine_proc is not None:
            if self._engine_proc.poll() is not None:
                log.warning("IBus engine process died (rc=%d), restarting",
                            self._engine_proc.returncode)
                self._engine_proc = None
                try:
                    self._ensure_ibus_engine()
                    log.info("IBus engine recovered")
                except Exception:
                    log.exception("IBus engine recovery failed")
            elif self._state == _State.IDLE:
                # Engine process alive — check if its socket listener is
                # actually responding. A zombie engine (process alive but
                # GLib loop stuck / socket thread dead) silently drops all
                # preedit and commit messages, causing the "transcription
                # works but no text appears" symptom.
                if not self._ping_ibus_engine():
                    log.warning("IBus engine not responding to ping, restarting")
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
                    # Engine alive and responding — check IBus registration
                    self._reactivate_ibus_if_stale()

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

        signal.signal(signal.SIGINT, lambda *_: self._shutdown.set())
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
