"""The daemon's current typer: selection, runtime fallback and upgrades."""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from voiceio import platform as plat
from voiceio.typers import chain as typer_chain

if TYPE_CHECKING:
    from voiceio.ibus.session import IBusSession
    from voiceio.platform import Platform
    from voiceio.typers.base import TyperBackend

log = logging.getLogger("voiceio")


class TyperManager:
    """Owns ``typer`` and ``platform`` and the upgrade/fallback policy.

    The hotkey and finalizer paths read both, so after start-up they change
    only through the swap helpers: under ``lock`` (the daemon's hotkey lock)
    and only while ``is_idle()``. ``state`` is for log messages.
    """

    def __init__(
        self,
        platform: Platform,
        typer: TyperBackend,
        *,
        ibus: IBusSession,
        method: str,
        auto_fallback: bool,
        lock: threading.Lock,
        is_idle: Callable[[], bool],
        state: Callable[[], object],
        shutdown: threading.Event,
    ):
        self.platform = platform
        self.typer = typer
        self._ibus = ibus
        self._method = method
        self._auto_fallback = auto_fallback
        self._lock = lock
        self._is_idle = is_idle
        self._state = state
        self._shutdown = shutdown

    # ── Daemon start ────────────────────────────────────────────────────

    def redetect_at_start(self) -> None:
        """Refresh env and platform in case the desktop session exported
        vars between __init__ and run() (e.g. boot race with user@.service)."""
        plat.import_graphical_env()
        old_platform = self.platform
        self.platform = plat.redetect()
        if (self.platform.display_server != old_platform.display_server
                or self.platform.desktop != old_platform.desktop):
            log.info(
                "Platform changed since init: display=%s desktop=%s (was %s/%s), re-selecting typer",
                self.platform.display_server, self.platform.desktop,
                old_platform.display_server, old_platform.desktop,
            )
            self.try_upgrade(reason="run-init")

    def prepare(self) -> None:
        """Bring the selected backend up before hotkeys start."""
        if self.typer.name == "ibus":
            # Daemon-start is one of the only places allowed to install the
            # component / restart IBus (probe() is read-only now, fix #6).
            try:
                if hasattr(self.typer, "ensure_installed"):
                    self.typer.ensure_installed()
            except Exception:
                log.exception("IBus ensure_installed failed (continuing)")
            # Register the engine but never claim the input source at boot —
            # the instance is created on first activation (record start).
            self._ibus.ensure_engine(activate=False)
        else:
            # IBus daemon may not be ready at startup (race with graphical
            # session). If IBus is in the preferred chain but wasn't selected,
            # wait briefly and re-probe. Re-fetch the chain after env refresh
            # so we don't use a stale "clipboard-only" chain from when
            # platform=unknown.
            chain = typer_chain._get_chain(self.platform)
            if "ibus" in chain and self.typer.name != "ibus":
                self.typer = self.wait_for_ibus() or self.typer

    def wait_for_ibus(self) -> TyperBackend | None:
        """At login voiceio may start before ibus-daemon: wait briefly and
        switch to the IBus typer instead of settling on a broken fallback."""
        if self._ibus.wait_for_daemon():
            from voiceio.ibus.install import refresh_stale_launcher
            refresh_stale_launcher()
            try:
                typer = typer_chain.select(self.platform, self._method)
                if typer.name == "ibus":
                    self._ibus.ensure_engine(activate=False)
                    return typer
            except RuntimeError:
                pass
        log.warning("IBus daemon did not start in time, using fallback typer: %s", self.typer.name)
        return None

    # ── Runtime fallback ────────────────────────────────────────────────

    # ── per-recording ────────────────────────────────────────────────────
    def begin(self, rid: int) -> None:
        """A recording starts: an IBus typer needs the input source."""
        if self.typer.name == "ibus":
            self._ibus.claim(rid)

    def end(self, rid: int) -> None:
        """Its output is committed (or abandoned): give the source back. A
        no-op if a newer recording has claimed it, or nothing was claimed."""
        self._ibus.release(rid)

    @property
    def pastes_from_clipboard(self) -> bool:
        """The typer pastes from the clipboard, so nothing else may overwrite
        it while a paste may still be pending."""
        return self.typer.name == "clipboard"

    def type_with_fallback(self, text: str) -> None:
        try:
            self.typer.type_text(text)
        except Exception as e:
            if not self._auto_fallback:
                raise
            log.warning("Typer '%s' failed: %s, trying fallback", self.typer.name, e)
            probe = self.typer.probe()
            if not probe.ok:
                log.warning("Typer '%s' no longer works: %s", self.typer.name, probe.reason)
            try:
                new = typer_chain.select(self.platform)   # probes: outside the lock
            except RuntimeError:
                log.error("No working typer backend available")
                return
            with self._lock:
                self.typer = new
            log.info("Switched to typer: %s", new.name)
            new.type_text(text)

    def clipboard_fallback(self) -> TyperBackend | None:
        """A clipboard typer for the current platform, or None if unavailable."""
        try:
            from voiceio.typers.clipboard import ClipboardTyper
            return ClipboardTyper(self.platform)
        except Exception:
            log.debug("Could not build clipboard fallback typer", exc_info=True)
            return None

    def on_broken(self) -> None:
        """Called by streaming session when typer fails repeatedly.

        Defers the re-probe+upgrade to a background thread that waits
        for the state to reach IDLE. We cannot hot-swap mid-recording:
        the streaming session's TypedText tracks text in terms of the old
        typer's behavior (char-level for clipboard vs preedit for ibus),
        and mid-stream swapping would leave stale or duplicated chars.
        The current recording is already broken — accept that, fix it
        before the next one.
        """
        threading.Thread(
            target=self._deferred_upgrade,
            daemon=True, name="typer-upgrade",
        ).start()

    def _deferred_upgrade(self) -> None:
        """Wait for IDLE state, then re-detect platform and upgrade typer.

        Lock discipline: all subprocess probing (redetect, chain resolve,
        select, engine start) runs OUTSIDE the hotkey lock. Only the field
        assignments go through the lock-guarded swap helpers, which apply the
        change only while IDLE. We must never hold the lock across a typer or
        engine subprocess call.
        """
        # Wait up to 30s for recording/finalizing to complete
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._is_idle() or self._shutdown.is_set():
                break
            time.sleep(0.5)
        if self._shutdown.is_set():
            return
        if not self._is_idle():
            log.debug("Typer upgrade abandoned: state=%s after 30s", self._state())
            return
        plat.import_graphical_env()
        if not self.swap_platform(plat.redetect()):
            return  # left IDLE before we could apply
        log.info(
            "Typer broken: re-detected platform: display=%s desktop=%s",
            self.platform.display_server, self.platform.desktop,
        )
        self.try_upgrade(reason="streaming-failure")

    # ── IDLE-gated swaps ────────────────────────────────────────────────

    def swap_platform(self, new_platform: Platform) -> bool:
        """Assign ``platform`` under the lock, only while IDLE.

        Health/upgrade threads must not mutate shared state that the hotkey and
        finalizer paths read, except while IDLE and holding the lock. Returns
        True if applied.
        """
        with self._lock:
            if not self._is_idle():
                return False
            self.platform = new_platform
            return True

    def swap_typer(self, new_typer: TyperBackend, reason: str = "") -> bool:
        """Assign ``typer`` under the lock, only while IDLE.

        Contains NO subprocess/engine operations — callers perform those
        (probing, engine start) outside the lock. Returns True if applied.
        """
        with self._lock:
            if not self._is_idle():
                log.debug("Typer swap (%s) skipped: state=%s", reason, self._state())
                return False
            old = self.typer.name
            self.typer = new_typer
        log.info("Typer upgraded: %s -> %s (%s)", old, new_typer.name, reason)
        return True

    def refresh_platform(self) -> None:
        """Re-detect the platform now that env vars may have been refreshed
        (lock-guarded, IDLE-only — a live recording keeps its platform)."""
        plat.import_graphical_env()
        self.swap_platform(plat.redetect())

    # ── Upgrades ────────────────────────────────────────────────────────

    def try_upgrade(self, reason: str = "") -> bool:
        """Try to switch to a better typer backend.

        Called on resume, after env import, or after repeated typer failures.
        If the current typer is a fallback (e.g. clipboard) and a preferred
        backend (e.g. ibus) is now available, switch to it.

        Returns True if typer was upgraded.
        """
        chain = typer_chain._get_chain(self.platform)
        current_idx = chain.index(self.typer.name) if self.typer.name in chain else len(chain)
        log.debug(
            "Upgrade attempt (%s): current=%s (idx %d) chain=%s",
            reason, self.typer.name, current_idx, chain,
        )
        if current_idx == 0:
            log.debug("Upgrade (%s) skipped: already on best backend in chain", reason)
            return False  # already on the best backend

        # Reset clipboard tool cache so re-probe sees updated env
        from voiceio.typers.clipboard import ClipboardTyper
        if isinstance(self.typer, ClipboardTyper):
            self.typer.reset_tools()

        # Log all probe results so we can diagnose why upgrade failed
        try:
            results = typer_chain.resolve(self.platform, self._method)
        except Exception:
            log.exception("Upgrade (%s): typer resolve failed", reason)
            return False
        for name, _backend, probe in results:
            status = "OK" if probe.ok else f"FAIL: {probe.reason}"
            log.info("Upgrade (%s): probe %s -> %s", reason, name, status)

        try:
            new_typer = typer_chain.select(self.platform, self._method)
        except RuntimeError as e:
            log.warning("Upgrade (%s) select failed: %s", reason, e)
            return False

        new_idx = chain.index(new_typer.name) if new_typer.name in chain else len(chain)
        if new_idx < current_idx:
            # Apply the swap under the lock (IDLE-gated); if it didn't apply
            # (state left IDLE), don't start the engine.
            if not self.swap_typer(new_typer, reason):
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
        if isinstance(self.typer, ClipboardTyper):
            self.typer._resolve_tools()
            log.debug("Clipboard typer tools re-resolved (%s)", reason)

        return False

    def health_upkeep(self) -> None:
        """IDLE-only: refresh platform + upgrade/repair the typer.

        Runs the expensive probing outside the hotkey lock; the platform and
        typer field mutations go through the lock-guarded swap helpers.
        """
        plat.import_graphical_env()
        old_desktop = self.platform.desktop
        new_platform = plat.redetect()
        if new_platform.desktop != old_desktop:
            log.info(
                "Platform refreshed: display=%s desktop=%s (was desktop=%s)",
                new_platform.display_server, new_platform.desktop, old_desktop,
            )
        self.swap_platform(new_platform)

        chain = typer_chain._get_chain(self.platform)
        current_idx = chain.index(self.typer.name) if self.typer.name in chain else len(chain)
        if current_idx > 0:
            self.try_upgrade(reason="health-check")

        # Re-probe current backend; re-select if it broke.
        probe = self.typer.probe()
        if not probe.ok:
            log.warning("Typer '%s' probe failed: %s — re-selecting", self.typer.name, probe.reason)
            self.try_upgrade(reason="probe-failed")
