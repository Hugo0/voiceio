"""Main VoiceIO engine — state machine, backend wiring, self-healing."""
from __future__ import annotations

import fcntl
import logging
import os
import signal
import subprocess
import threading
import time

import numpy as np

from voiceio import config, platform as plat
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.hotkeys.socket_backend import SocketHotkey
from voiceio.recorder import AudioRecorder
from voiceio.streaming import StreamingSession
from voiceio.transcriber import Transcriber
from voiceio.typers import chain as typer_chain
from voiceio.typers.base import StreamingTyper
log = logging.getLogger("voiceio")



class VoiceIO:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.platform = plat.detect()

        # Select backends
        self._hotkey = hotkey_chain.select(self.platform, cfg.hotkey.backend)
        self._typer = typer_chain.select(self.platform, cfg.output.method)
        self._auto_fallback = cfg.health.auto_fallback

        # Always start socket backend alongside native hotkey
        self._socket: SocketHotkey | None = None
        if self._hotkey.name != "socket":
            self._socket = SocketHotkey()

        print(f"Loading model '{cfg.model.name}'...", end="", flush=True)
        t0 = time.monotonic()
        self.transcriber = Transcriber(cfg.model)
        print(f" ready ({time.monotonic() - t0:.1f}s)")
        self.recorder = AudioRecorder(cfg.audio)
        self._streaming = cfg.output.streaming
        self._session: StreamingSession | None = None
        self._processing = False
        self._record_start: float = 0
        self._prev_ibus_engine: str | None = None
        self._engine_proc: subprocess.Popen | None = None
        self._shutdown = threading.Event()

    def request_shutdown(self) -> None:
        """Request graceful shutdown from an external signal handler."""
        self._shutdown.set()

    def on_hotkey(self) -> None:
        if self.recorder.is_recording:
            elapsed = time.monotonic() - self._record_start
            if elapsed < self.cfg.output.cancel_window_secs:
                # Quick double-press = cancel recording without typing
                if self._streaming and self._session is not None:
                    self._session.stop()
                    self._session = None
                self.recorder.stop()
                if isinstance(self._typer, StreamingTyper):
                    self._typer.clear_preedit()
                self._deactivate_ibus()
                log.info("Recording cancelled (double-press)")
                return
            if elapsed < self.cfg.output.min_recording_secs:
                log.debug("Ignoring stop — only %.1fs into recording (min %.1fs)", elapsed, self.cfg.output.min_recording_secs)
                return

            self._play_record_cue(start=False)
            if self._streaming and self._session is not None:
                final_text = self._session.stop()
                self.recorder.stop()
                self._session = None
                if final_text:
                    self._play_feedback(final_text)
                log.info("Streaming done (%.1fs): '%s'", elapsed, final_text)
            else:
                audio = self.recorder.stop()
                log.info("Stopped recording (%.1fs)", elapsed)
                if audio is not None and not self._processing:
                    threading.Thread(target=self._process, args=(audio,), daemon=True).start()
            # Deactivate IBus engine — return keyboard to normal
            self._deactivate_ibus()
        elif not self._processing:
            # Activate IBus engine so preedit/commit can reach the focused app
            self._activate_ibus()
            self._record_start = time.monotonic()
            self.recorder.start()
            self._play_record_cue(start=True)
            if self._streaming:
                self._session = StreamingSession(
                    self.transcriber, self._typer, self.recorder,
                )
                self._session.start()
            log.info("Recording... press [%s] again to stop", self.cfg.hotkey.key)

    def _process(self, audio: np.ndarray) -> None:
        self._processing = True
        try:
            text = self.transcriber.transcribe(audio)
            if text:
                self._type_with_fallback(text)
                self._play_feedback(text)
                log.info("Typed: '%s'", text)
        except Exception:
            log.exception("Processing failed")
        finally:
            self._processing = False
            self._deactivate_ibus()

    def _activate_ibus(self) -> None:
        """Switch GNOME input source to voiceio engine for text injection.

        Done in a thread to avoid blocking the hotkey handler — the 0.5s
        GNOME activation delay is fine since transcription takes ~1s anyway.
        """
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
        self._set_gnome_input_source_index(0)
        log.debug("IBus engine deactivated — keyboard restored")

    def _play_record_cue(self, start: bool) -> None:
        """Play a subtle click on record start/stop."""
        if not self.cfg.feedback.sound_enabled:
            return
        if start:
            from voiceio.feedback import play_record_start
            play_record_start()
        else:
            from voiceio.feedback import play_record_stop
            play_record_stop()

    def _play_feedback(self, text: str) -> None:
        """Play sound and/or notification after committing text."""
        if self.cfg.feedback.sound_enabled:
            from voiceio.feedback import play_commit_sound
            play_commit_sound()
        if self.cfg.feedback.notify_clipboard:
            from voiceio.feedback import notify_clipboard
            notify_clipboard(text)

    def _type_with_fallback(self, text: str) -> None:
        """Type text, falling back to next backend on failure."""
        try:
            self._typer.type_text(text)
        except Exception as e:
            if not self._auto_fallback:
                raise
            log.warning("Typer '%s' failed: %s — trying fallback", self._typer.name, e)
            probe = self._typer.probe()
            if not probe.ok:
                log.warning("Typer '%s' no longer works: %s", self._typer.name, probe.reason)
            try:
                self._typer = typer_chain.select(self.platform)
                log.info("Switched to typer: %s", self._typer.name)
                self._typer.type_text(text)
            except RuntimeError:
                log.error("No working typer backend available")

    def _ensure_ibus_engine(self) -> None:
        """Start the VoiceIO IBus engine and activate it.

        We spawn the engine process directly (bypassing `ibus engine` which
        is unreliable), then switch the GNOME input source to voiceio.
        """
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

        # Phase 1: wait for socket (engine process started, accepting commands)
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
                log.warning("IBus engine started but socket not found — commands may fail")
            return

        # Phase 2: activate via `ibus engine voiceio` to create engine instance.
        # This triggers do_create_engine. We do NOT switch GNOME input source
        # here — that only happens during active recording to avoid blocking
        # keyboard input when voiceio is idle.
        log.info("Activating VoiceIO IBus engine...")
        activate_proc = subprocess.Popen(
            ["ibus", "engine", "voiceio"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=ibus_env,
        )

        # Phase 3: wait for engine instance (created by IBus via factory)
        for i in range(200):  # up to 20s
            if READY_PATH.exists():
                log.info("VoiceIO IBus engine instance ready (%.1fs)", i * 0.1)
                break
            time.sleep(0.1)
        else:
            log.warning("IBus engine instance not created — preedit may not work")

        # Clean up the activation process (don't leave it dangling)
        try:
            activate_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            activate_proc.kill()

        # Switch back to normal keyboard — engine is ready but stays dormant
        # until recording starts
        self._set_gnome_input_source_index(0)

        log.info("VoiceIO IBus engine ready (dormant until recording)")

    @staticmethod
    def _kill_stale_engine(socket_path) -> None:
        """Kill any orphaned voiceio-ibus-engine process and remove stale socket."""
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
                time.sleep(0.3)  # let it die
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _switch_gnome_input_source(self, engine_name: str) -> None:
        """Switch GNOME input source to the given IBus engine."""
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
            # Find index of ('ibus', 'voiceio') in the sources list
            # and set current to that index
            if f"('ibus', '{engine_name}')" not in sources:
                return
            # Parse to find index — sources format: [('xkb', 'us'), ('ibus', 'voiceio')]
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
                    # Give GNOME a moment to activate
                    time.sleep(0.5)
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _stop_ibus_engine(self) -> None:
        """Stop the IBus engine process and restore previous input method."""
        # Always restore normal keyboard first — most critical step
        self._set_gnome_input_source_index(0)

        # Terminate engine process we spawned
        if self._engine_proc is not None:
            self._engine_proc.terminate()
            try:
                self._engine_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._engine_proc.kill()
            self._engine_proc = None
            log.debug("Stopped IBus engine process")

        # Clean up socket
        from voiceio.ibus import SOCKET_PATH
        SOCKET_PATH.unlink(missing_ok=True)

        # Restore previous IBus engine
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
        """Set GNOME input source by index."""
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

    def run(self) -> None:
        from voiceio.config import PID_PATH, LOG_DIR

        # Single-instance guard via file lock (atomic, no TOCTOU race)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._pid_fd = open(PID_PATH, "w")
        try:
            fcntl.flock(self._pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._pid_fd.close()
            log.error("Another voiceio instance is already running")
            print("voiceio is already running. Stop it first: voiceio service stop")
            return
        self._pid_fd.write(str(os.getpid()))
        self._pid_fd.flush()

        # Start IBus engine if needed
        if self._typer.name == "ibus":
            self._ensure_ibus_engine()

        # Open always-on audio stream for pre-buffering
        self.recorder.open_stream()

        # Start hotkey backends
        self._hotkey.start(self.cfg.hotkey.key, self.on_hotkey)
        if self._socket is not None:
            self._socket.start(self.cfg.hotkey.key, self.on_hotkey)

        from voiceio import __version__
        log.info(
            "voiceio v%s ready — press [%s] to toggle recording (hotkey=%s, typer=%s)",
            __version__, self.cfg.hotkey.key, self._hotkey.name, self._typer.name,
        )
        print(
            f"voiceio v{__version__} ready — press [{self.cfg.hotkey.key}] to record "
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
            self.recorder.close_stream()
            self.transcriber.shutdown()
            self._stop_ibus_engine()
            self._pid_fd.close()
            PID_PATH.unlink(missing_ok=True)
        log.info("voiceio stopped")
