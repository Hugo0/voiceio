"""IBus engine process + desktop input-source lifecycle for the daemon.

Two jobs, both hazardous when done at the wrong time (see the engine
lifecycle notes in CLAUDE.md / git history):

* keep the VoiceIO engine PROCESS resident and registered while idle, but
  never activated — "dead while idle" lets ibus-daemon exec-spawn a
  socket-stealing duplicate, "active while idle" routes every input context
  through voiceio and wipes visible preedits;
* claim the desktop input source ONLY while a recording is live, and give
  back exactly what the user had.

How the source is switched depends on the desktop, behind ``InputSources``:
GNOME owns it via gsettings (plus an explicit ``ibus engine`` claim, since
the gsettings 'current' key is a no-op on newer GNOME); everything else
(KDE, sway, ...) just switches ibus's global engine.
"""
from __future__ import annotations

import ast
import logging
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from voiceio.platform import Platform

log = logging.getLogger("voiceio")

ENGINE_NAME = "voiceio"
_GSCHEMA = "org.gnome.desktop.input-sources"


def _ibus_env() -> dict[str, str]:
    from voiceio.typers.ibus import _ibus_env as env
    return env()


def query_global_engine() -> str | None:
    """ibus's current global engine name, or None if unavailable."""
    try:
        result = subprocess.run(
            ["ibus", "engine"], capture_output=True, text=True,
            timeout=3, env=_ibus_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def set_global_engine(name: str) -> None:
    try:
        subprocess.run(
            ["ibus", "engine", name],
            capture_output=True, timeout=3, env=_ibus_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def kill_stale_engines(socket_path) -> None:
    """Remove the command socket and SIGTERM any running engine process."""
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


# ── Input sources ───────────────────────────────────────────────────────

class InputSources(Protocol):
    """How the desktop switches input methods.

    ``activate_voiceio`` remembers what the user had; ``restore`` gives it
    back and points ibus's global engine at the user's layout again.
    """

    def current(self) -> int | str | None: ...
    def activate_voiceio(self) -> None: ...
    def restore(self) -> None: ...
    def return_to(self, token: int | str | None) -> None: ...
    def switched_away(self) -> bool: ...
    def shutdown(self) -> None: ...


class GnomeInputSources:
    """GNOME: gsettings ``org.gnome.desktop.input-sources`` + ``ibus engine``."""

    def __init__(self, fallback_engine: Callable[[], str]):
        self._fallback_engine = fallback_engine
        # The source the user had before we claimed (None: unknown, or
        # already on voiceio) and voiceio's own index (None: not claimed).
        self.prev_index: int | None = None
        self.voiceio_index: int | None = None

    def current(self) -> int | None:
        try:
            result = subprocess.run(
                ["gsettings", "get", _GSCHEMA, "current"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    def set_index(self, index: int) -> None:
        try:
            subprocess.run(
                ["gsettings", "set", _GSCHEMA, "current", str(index)],
                capture_output=True, timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def release_engine(self) -> None:
        """Point ibus's GLOBAL engine back at the user's layout.

        Restoring the GNOME source is not enough: activation sets the global
        engine directly, and a gsettings write that doesn't change the value
        never makes GNOME push an engine change. A stale global engine routes
        EVERY input context through voiceio while "dormant" — focus churn that
        discards any visible preedit, surviving daemon restarts.

        Unconditional: querying first races the (async) activation — a stale
        reading made the release skip itself and left voiceio active.
        """
        fallback = self._fallback_engine()
        set_global_engine(fallback)
        log.debug("Released ibus global engine to %s", fallback)

    def activate_voiceio(self) -> None:
        try:
            result = subprocess.run(
                ["gsettings", "get", _GSCHEMA, "sources"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return
            sources = result.stdout.strip()
            if f"('ibus', '{ENGINE_NAME}')" not in sources:
                return
            try:
                source_list = ast.literal_eval(sources)
            except (ValueError, SyntaxError):
                return
            for i, (kind, name) in enumerate(source_list):
                if kind == "ibus" and name == ENGINE_NAME:
                    # Record the source the user actually had so we can
                    # restore exactly it, and remember voiceio's index to
                    # detect a mid-recording switch-away.
                    prev = self.current()
                    if prev is not None and prev != i:
                        self.prev_index = prev
                    self.voiceio_index = i
                    self.set_index(i)
                    # The gsettings write alone is a no-op on newer GNOME
                    # (the 'current' key is legacy) — claim the global engine
                    # explicitly, mirroring release_engine().
                    set_global_engine(ENGINE_NAME)
                    log.info("Switched GNOME input source to index %d (%s)", i, ENGINE_NAME)
                    time.sleep(0.5)
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def restore(self) -> None:
        idx = self.prev_index
        self.set_index(idx if idx is not None else 0)
        self.prev_index = None
        self.voiceio_index = None
        self.release_engine()

    def return_to(self, token: int | str | None) -> None:
        self.set_index(token if isinstance(token, int) else 0)
        self.release_engine()

    def switched_away(self) -> bool:
        if self.voiceio_index is None:
            return False
        cur = self.current()
        return cur is not None and cur != self.voiceio_index

    def shutdown(self) -> None:
        # Only hand back a source we claimed; never force the user to index 0.
        claimed = self.voiceio_index is not None
        if self.prev_index is not None:
            self.set_index(self.prev_index)
        self.prev_index = None
        self.voiceio_index = None
        if claimed:
            self.release_engine()


class IBusEngineSources:
    """Non-GNOME desktops (KDE, sway, ...): switch ibus's global engine.

    There is no desktop-owned source list to drive, so claiming means
    ``ibus engine voiceio`` and restoring means switching back to the engine
    the user had at that moment.
    """

    def __init__(self, fallback_engine: Callable[[], str]):
        self._fallback_engine = fallback_engine
        self.prev_engine: str | None = None
        self.claimed = False

    def current(self) -> str | None:
        return query_global_engine()

    def activate_voiceio(self) -> None:
        prev = self.current()
        if prev and prev != ENGINE_NAME:
            self.prev_engine = prev
        set_global_engine(ENGINE_NAME)
        self.claimed = True
        log.info("Switched IBus engine to %s (was %s)", ENGINE_NAME, prev)
        time.sleep(0.5)

    def _back_to(self, engine: str | None) -> None:
        target = engine if engine and engine != ENGINE_NAME else self._fallback_engine()
        set_global_engine(target)
        log.debug("Restored ibus global engine to %s", target)

    def restore(self) -> None:
        self._back_to(self.prev_engine)
        self.prev_engine = None
        self.claimed = False

    def return_to(self, token: int | str | None) -> None:
        self._back_to(token if isinstance(token, str) else None)

    def switched_away(self) -> bool:
        if not self.claimed:
            return False
        cur = self.current()
        return cur is not None and cur != ENGINE_NAME

    def shutdown(self) -> None:
        if self.claimed:
            self.restore()


# ── Session ─────────────────────────────────────────────────────────────

class IBusSession:
    """Owns the engine process and the recording-time input-source claim.

    ``claim``/``release`` bracket one recording. Generations: the daemon bumps
    its generation on every start and stop; ``release(gen)`` from a finalizer
    whose generation is no longer current is a no-op — a newer recording has
    claimed the source and restoring it would kill that recording's preedit.
    """

    def __init__(
        self,
        platform: Callable[[], Platform],
        *,
        is_idle: Callable[[], bool],
        generation: Callable[[], int],
        sources: InputSources | None = None,
    ):
        self._platform = platform
        self._is_idle = is_idle
        self._generation = generation
        self._fixed_sources = sources
        self._gnome = GnomeInputSources(self.engine_fallback)
        self._plain = IBusEngineSources(self.engine_fallback)

        self._proc: subprocess.Popen | None = None
        # The user's engine before ours was spawned (for releasing).
        self._prev_engine: str | None = None
        # Consecutive missed pings: one miss on a busy system isn't a zombie.
        self._ping_fails = 0
        # Engine stderr goes to a file, never a pipe (see _spawn).
        self._stderr_file = None
        self._stderr_path = None
        # Single-flight spawning (health loop vs record start).
        self._spawn_lock = threading.Lock()

        # Claim bookkeeping. _lock guards the fields and is never held across
        # a subprocess call (claim() runs under the daemon's hotkey lock);
        # _source_lock serializes the actual activate/restore switching.
        self._lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._wanted: int | None = None       # generation of the live claim
        self._claimer: InputSources | None = None
        self._hijack_reported = False

    # ── sources ─────────────────────────────────────────────────────────

    def sources(self) -> InputSources:
        if self._fixed_sources is not None:
            return self._fixed_sources
        return self._gnome if self._platform().is_gnome else self._plain

    def engine_fallback(self) -> str:
        """The user's non-voiceio ibus engine name, for releasing ours."""
        if self._prev_engine and self._prev_engine != ENGINE_NAME:
            return self._prev_engine
        try:
            result = subprocess.run(
                ["gsettings", "get", _GSCHEMA, "sources"],
                capture_output=True, text=True, timeout=3,
            )
            for kind, name in ast.literal_eval(result.stdout.strip()):
                if kind == "xkb":
                    return f"xkb:{name}::eng"
        except Exception:
            pass
        return "xkb:us::eng"

    # ── recording claim ─────────────────────────────────────────────────

    @property
    def has_engine(self) -> bool:
        return self._proc is not None

    def claim(self, generation: int) -> None:
        """Record start: claim the input source for voiceio (background)."""
        sources = self.sources()
        with self._lock:
            self._wanted = generation
            self._claimer = sources
            self._hijack_reported = False
        if self._proc is None:
            # Reaped while dormant (normal — see health_tick). Respawn in the
            # background; the engine's pending buffer holds early preedits,
            # and since we're not idle the spawn leaves voiceio active.
            threading.Thread(target=self._respawn_for_recording, daemon=True).start()
        threading.Thread(
            target=self._activate, args=(sources, generation), daemon=True,
        ).start()

    def _activate(self, sources: InputSources, generation: int) -> None:
        with self._source_lock:
            with self._lock:
                if self._wanted != generation:
                    return  # released before we got here
            sources.activate_voiceio()

    def release(self, generation: int) -> None:
        """Recording finished: give the source back (background).

        A no-op if ``generation`` was superseded, or nothing is claimed.
        """
        with self._lock:
            current = self._generation()
            if generation != current:
                log.debug("Skip IBus release: gen %d superseded by %d", generation, current)
                return
            if self._wanted is None:
                return
            self._wanted = None
            sources = self._claimer
        threading.Thread(target=self._restore, args=(sources,), daemon=True).start()

    def _restore(self, sources: InputSources) -> None:
        with self._source_lock:
            with self._lock:
                if self._wanted is not None:
                    return  # a newer recording claimed meanwhile; it owns it
            sources.restore()

    def hijacked(self) -> bool:
        """True once per recording if the user switched away from voiceio.

        The caller should stop fighting and fall back to the clipboard.
        """
        with self._lock:
            if self._hijack_reported or self._wanted is None:
                return False
            sources = self._claimer
        if sources is None or not sources.switched_away():
            return False
        with self._lock:
            if self._hijack_reported:
                return False
            self._hijack_reported = True
        log.warning(
            "Input source switched away from voiceio mid-recording; "
            "falling back to clipboard for this session (not fighting the user)",
        )
        return True

    # ── engine process ──────────────────────────────────────────────────

    def _respawn_for_recording(self) -> None:
        try:
            self.ensure_engine()
        except Exception:
            log.exception("IBus engine respawn at record start failed")

    def stderr_tail(self, limit: int = 500) -> str:
        """Last bytes of the engine's stderr file, for crash reporting."""
        if self._stderr_path is None:
            return ""
        try:
            if self._stderr_file is not None:
                self._stderr_file.flush()
            return self._stderr_path.read_text(errors="replace").strip()[-limit:]
        except OSError:
            return ""

    def ensure_engine(self, activate: bool = True) -> None:
        """Start the engine, optionally pre-creating its instance.

        ``activate=False`` spawns and registers the process without touching
        the input source — the only safe mode while idle. ``activate=True``
        is reserved for a live recording.

        Single-flight: a concurrent spawn is skipped rather than queued —
        two spawn dances would fight over the input source.
        """
        if not self._spawn_lock.acquire(blocking=False):
            log.debug("IBus engine spawn already in progress, skipping")
            return
        try:
            self._spawn(activate)
        finally:
            self._spawn_lock.release()

    def _spawn(self, activate: bool) -> None:
        from voiceio.config import LOG_DIR
        from voiceio.ibus import READY_PATH, SOCKET_PATH
        from voiceio.typers.ibus import LAUNCHER_PATH

        ibus_env = _ibus_env()

        prev = query_global_engine()
        if prev and prev != ENGINE_NAME:
            self._prev_engine = prev
            log.debug("Previous IBus engine: %s", prev)

        kill_stale_engines(SOCKET_PATH)
        READY_PATH.unlink(missing_ok=True)

        # stderr goes to a FILE, never a pipe: the engine logs at DEBUG, and
        # an undrained pipe fills its 64KB buffer and blocks the engine
        # mid-write — alive but frozen, so it stops answering pings and the
        # watchdog reaps it.
        log.info("Starting VoiceIO IBus engine...")
        stderr_target = subprocess.DEVNULL
        try:
            self._stderr_path = LOG_DIR / "ibus-engine-stderr.log"
            self._stderr_file = open(self._stderr_path, "w")
            stderr_target = self._stderr_file
        except OSError:
            self._stderr_path = None
        try:
            self._proc = subprocess.Popen(
                [str(LAUNCHER_PATH)],
                stdout=subprocess.DEVNULL, stderr=stderr_target,
                env=ibus_env,
            )
        except OSError as e:
            log.warning("Could not start IBus engine: %s", e)
            return

        # Phase 1: wait for the command socket
        for i in range(40):
            if SOCKET_PATH.exists():
                log.info("VoiceIO IBus engine socket ready (%.1fs)", i * 0.1)
                break
            time.sleep(0.1)
        else:
            if self._proc.poll() is not None:
                log.error("IBus engine crashed (rc=%d): %s",
                          self._proc.returncode, self.stderr_tail())
            else:
                log.warning("IBus engine started but socket not found, commands may fail")
            return

        if not activate:
            # Registered-but-dormant: our factory owns the component name, so
            # ibus-daemon won't exec-spawn a duplicate, and an accidental
            # switch onto the voiceio source lands on a live pass-through
            # engine. The instance is created on first activation.
            log.info("VoiceIO IBus engine ready (registered, dormant)")
            return

        # Remember where the user is so we can return to dormancy there.
        sources = self.sources()
        dormant = sources.current()

        # Phase 2: activate
        log.info("Activating VoiceIO IBus engine...")
        activate_proc = subprocess.Popen(
            ["ibus", "engine", ENGINE_NAME],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=ibus_env,
        )

        # Phase 3: wait for the engine instance
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

        # Back to dormancy — unless a recording claimed the source while we
        # were spawning: yanking it now would kill that recording's preedit.
        if self._is_idle():
            sources.return_to(dormant)
            log.info("VoiceIO IBus engine ready (dormant until recording)")
        else:
            log.info("VoiceIO IBus engine ready (recording live — staying active)")

    def ping(self) -> bool:
        """False: process alive but its socket listener dead or starved."""
        from voiceio.ibus import ping_engine
        return ping_engine(timeout=2.0)

    def health_tick(self, idle: bool) -> None:
        """Watchdog: keep the engine resident; never touch the source.

        A dormant engine dying is NORMAL — ibus-daemon reaps deactivated
        engines (SIGTERM ~10s after the source switches away). Respawn it
        registered but never activated. Activation happens only at record
        start: idle activation was the focus churn that wiped preedits.
        """
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is not None:
            rc = proc.returncode
            self._proc = None
            self._ping_fails = 0
            if idle:
                log.info(
                    "IBus engine exited while dormant (rc=%s) — "
                    "respawning registered, without activation", rc,
                )
                try:
                    self.ensure_engine(activate=False)
                except Exception:
                    log.exception("IBus engine respawn failed")
            else:
                log.warning("IBus engine died mid-recording (rc=%s), restarting", rc)
                try:
                    self.ensure_engine()
                    log.info("IBus engine recovered")
                except Exception:
                    log.exception("IBus engine recovery failed")
            return
        if not idle:
            return
        # Zombie check: process alive but socket stolen / listener dead
        # silently drops every preedit. Require two consecutive misses.
        if self.ping():
            self._ping_fails = 0
            return
        self._ping_fails += 1
        if self._ping_fails < 2:
            log.debug("IBus engine missed a ping (%d/2) — busy system?", self._ping_fails)
            return
        log.warning("IBus engine not responding to ping, restarting")
        self._ping_fails = 0
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        self._proc = None
        try:
            self.ensure_engine(activate=False)
            log.info("IBus engine recovered (was zombie)")
        except Exception:
            log.exception("IBus engine recovery failed")

    def on_resume(self) -> None:
        """An engine that died during sleep is respawned lazily at the next
        record start — respawning now would grab the source as the user
        unlocks."""
        if self._proc is not None and self._proc.poll() is not None:
            log.debug("Resume: IBus engine died during sleep — will respawn at next recording")
            self._proc = None

    def wait_for_daemon(self, polls: int = 30, interval: float = 0.5) -> bool:
        """Poll for ibus-daemon (it may start after us at login)."""
        from voiceio.typers.ibus import _ibus_daemon_running
        log.info("IBus preferred but not available yet, waiting for ibus-daemon...")
        for i in range(polls):
            time.sleep(interval)
            if _ibus_daemon_running():
                log.info("IBus daemon ready after %.1fs, re-probing typers", (i + 1) * interval)
                return True
        return False

    def shutdown(self) -> None:
        """Stop the engine; hand back the source only if we claimed it."""
        with self._lock:
            sources = self._claimer
            self._wanted = None
        # Each sources object knows whether it still holds a claim (a release
        # may be in flight); one that never claimed, or already restored,
        # does nothing.
        with self._source_lock:
            if sources is not None:
                sources.shutdown()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            log.debug("Stopped IBus engine process")
        from voiceio.ibus import SOCKET_PATH
        SOCKET_PATH.unlink(missing_ok=True)
        # Don't leave ibus pointing at an engine that no longer exists — but
        # only if it really is ours: forcing a stale saved engine would
        # override a layout the user switched to since.
        if query_global_engine() == ENGINE_NAME:
            set_global_engine(self.engine_fallback())
            log.debug("Released ibus global engine at shutdown")
