"""IBus text injection backend: atomic text insertion via input method."""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time

from voiceio.backends import ProbeResult
from voiceio.ibus import SOCKET_PATH, install

log = logging.getLogger(__name__)
PING_TIMEOUT = 2.0


class IBusTyper:
    """IBus preedit for streaming preview, IBus commit for final text.

    Also copies committed text to clipboard so terminal users (where IBus
    doesn't reach) can Ctrl+Shift+V to paste.  When the IBus engine reports
    that no client has focus (e.g. terminal emulators), we auto-paste via
    ydotool/wtype so the user doesn't have to paste manually.
    """

    name = "ibus"

    # A healthy IBus setup does not change under our feet, and the full probe
    # spawns ~3 subprocesses. The health watchdog calls probe() every 10s, so
    # cache a successful result and only re-run the full check every few
    # minutes. Failures are never cached — they must be re-checked immediately.
    _PROBE_CACHE_TTL = 300.0

    def __init__(self, platform=None, **kwargs):
        self._wl_copy = shutil.which("wl-copy")
        self._ydotool = shutil.which("ydotool")
        self._wtype = shutil.which("wtype")
        self._sock: socket.socket | None = None
        self._wl_copy_proc: subprocess.Popen | None = None
        self._probe_cache: ProbeResult | None = None
        self._probe_cache_time = 0.0

    def probe(self) -> ProbeResult:
        """Read-only health probe with a cached cheap tier.

        Returns a cached OK result for up to _PROBE_CACHE_TTL to avoid spawning
        subprocesses on every 10s health tick. Never writes files or restarts
        IBus — installation lives in ensure_installed().
        """
        now = time.monotonic()
        cached = self._probe_cache
        if (cached is not None and cached.ok
                and now - self._probe_cache_time < self._PROBE_CACHE_TTL):
            return cached
        result = self._probe_full()
        self._probe_cache = result
        self._probe_cache_time = now
        return result

    def _probe_full(self) -> ProbeResult:
        """Full read-only probe (no side effects)."""
        from voiceio.platform import pkg_install
        if not shutil.which("ibus"):
            return ProbeResult(
                ok=False,
                reason="ibus not installed",
                fix_hint=pkg_install("ibus"),
            )

        if not install.has_ibus_gi():
            return ProbeResult(
                ok=False,
                reason="IBus Python bindings not available for system Python",
                fix_hint=pkg_install("gir1.2-ibus-1.0", "python3-gi"),
            )

        if not install.daemon_running():
            from voiceio.platform import detect
            if detect().input_method == "fcitx":
                return ProbeResult(
                    ok=False,
                    reason="fcitx5 is the active input method (IBus not used)",
                )
            return ProbeResult(
                ok=False,
                reason="IBus daemon not running",
                fix_hint="ibus-daemon -drxR",
            )

        if not install.component_installed():
            return ProbeResult(
                ok=False,
                reason="VoiceIO IBus component not installed",
                fix_hint="Run 'voiceio setup' to install",
            )

        if not install.gnome_source_configured():
            return ProbeResult(
                ok=False,
                reason="VoiceIO not in GNOME input sources",
                fix_hint="Run 'voiceio setup' to configure",
            )

        return ProbeResult(ok=True)

    def ensure_installed(self) -> bool:
        """Install the component + GNOME input source if missing.

        This is the ONLY place that writes files / restarts IBus. Call it from
        setup and daemon-start paths, never from the health loop. Invalidates
        the probe cache so the next probe() re-checks.
        """
        installed = install.ensure()
        self._probe_cache = None
        return installed

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._send(f"commit:{text}")
        self._copy_to_clipboard(text)
        if not self._engine_has_focus():
            self._simulate_paste()

    def delete_chars(self, n: int) -> None:
        pass  # Not needed: preedit handles corrections atomically

    def update_preedit(self, text: str) -> None:
        self._send(f"preedit:{text}")

    def commit_text(self, text: str) -> None:
        """Clear preedit and commit via IBus. Also copies to clipboard."""
        if not text:
            self._send("clear")
            return
        log.debug("Committing via IBus (%d chars)", len(text))
        self._send(f"commit:{text}")
        self._copy_to_clipboard(text)
        if not self._engine_has_focus():
            self._simulate_paste()

    def clear_preedit(self) -> None:
        self._send("clear")

    def _send(self, msg: str) -> None:
        """Send a command to the IBus engine via Unix DGRAM socket."""
        try:
            if self._sock is None:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._sock.sendto(msg.encode("utf-8"), str(SOCKET_PATH))
        except OSError as e:
            log.warning("IBus engine not reachable: %s", e)
            # Reset socket on error so next call retries
            if self._sock is not None:
                self._sock.close()
                self._sock = None

    def _engine_has_focus(self) -> bool:
        """Ask the IBus engine whether it has input focus in the active window."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.bind("")  # autobind for receiving reply
            sock.sendto(b"focus?", str(SOCKET_PATH))
            data, _ = sock.recvfrom(64)
            sock.close()
            focused = data == b"focused"
            log.debug("IBus engine focus: %s", data.decode(errors="replace"))
            return focused
        except (OSError, socket.timeout):
            # Engine not responding — assume NOT focused so we fall back
            # to clipboard paste. Worst case: a redundant Ctrl+V that
            # pastes over IBus-committed text. Better than silently losing
            # the text entirely.
            log.warning("IBus engine not responding to focus query, assuming unfocused")
            return False

    @staticmethod
    def _ydotoold_running() -> bool:
        """Check if ydotoold daemon is running (required for reliable ydotool)."""
        try:
            result = subprocess.run(
                ["pgrep", "-x", "ydotoold"],
                capture_output=True, timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _release_stuck_modifiers(self) -> None:
        """Force-release modifiers a half-executed ydotool may have left down.

        `ydotool key 29:1 … 29:0` emits press and release as separate uinput
        events. If the process is killed between them — timeout, OOM, ydotoold
        dying mid-call — Ctrl stays held at the *uinput* level: below the
        compositor, in every application, and clearable only by the user
        physically tapping Ctrl. That is indistinguishable from a broken
        keyboard, so never leave it to chance.

        29=LeftCtrl, 42=LeftShift, 56=LeftAlt. Releasing an already-released
        key is a no-op, so this is always safe to call.
        """
        if not self._ydotool:
            return
        try:
            subprocess.run([self._ydotool, "key", "29:0", "42:0", "56:0"],
                           capture_output=True, timeout=2)
            log.debug("Released modifiers after ydotool failure")
        except (OSError, subprocess.TimeoutExpired):
            log.warning("Could not release modifiers after a failed ydotool "
                        "paste — if Ctrl seems stuck, tap it once to clear")

    def _simulate_paste(self) -> None:
        """Simulate Ctrl+V to paste clipboard into non-IBus apps (terminals)."""
        try:
            if self._ydotool and self._ydotoold_running():
                # ydotool key scancodes: 29=LCtrl, 47=V
                try:
                    result = subprocess.run(
                        [self._ydotool, "key", "29:1", "47:1", "47:0", "29:0"],
                        capture_output=True, timeout=3,
                    )
                except subprocess.TimeoutExpired:
                    # Killed mid-chord — Ctrl may be down with no release coming.
                    self._release_stuck_modifiers()
                    raise
                if result.returncode != 0:
                    log.warning("ydotool paste failed (rc=%d): %s",
                                result.returncode, result.stderr.decode(errors="replace").strip())
                    # A non-zero exit can still have emitted the press.
                    self._release_stuck_modifiers()
                else:
                    log.debug("Pasted via ydotool")
            elif self._wtype:
                result = subprocess.run(
                    [self._wtype, "-M", "ctrl", "-k", "v", "-m", "ctrl"],
                    capture_output=True, timeout=3,
                )
                if result.returncode != 0:
                    log.warning("wtype paste failed (rc=%d): %s",
                                result.returncode, result.stderr.decode(errors="replace").strip())
                else:
                    log.debug("Pasted via wtype")
            else:
                log.warning("No paste tool available (ydotoold not running, wtype not found)")
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning("Paste simulation failed: %s", e)

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard as backup for non-IBus apps (terminals).

        Routed through clipboard_read.copy_text (xclip / XWayland bridge
        first): wl-copy momentarily steals keyboard focus on GNOME, and the
        focus-out discards any visible preedit.
        """
        # Kill any wl-copy left over from the pre-xclip implementation.
        if self._wl_copy_proc is not None:
            try:
                self._wl_copy_proc.kill()
            except OSError:
                pass
            self._wl_copy_proc = None
        from voiceio import clipboard_read
        if not clipboard_read.copy_text(text):
            log.debug("Clipboard copy failed (no working tool)")
