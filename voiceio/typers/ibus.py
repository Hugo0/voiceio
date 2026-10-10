"""IBus text injection backend: atomic text insertion via input method."""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time

from voiceio.backends import ProbeResult
from voiceio.ibus import SOCKET_PATH, install
from voiceio.ibus.textlimit import commit_secs
from voiceio.typers import paste
from voiceio.typers.ydotool import ydotoold_running

log = logging.getLogger(__name__)
PING_TIMEOUT = 2.0


class IBusTyper:
    """IBus preedit for streaming preview, IBus commit for final text.

    Also copies committed text to clipboard so terminal users (where IBus
    doesn't reach) can paste it.  When the IBus engine reports that no client
    has focus (e.g. terminal emulators), we press the paste chord
    (``output.paste_keys``, see typers/paste.py) via ydotool/wtype/xdotool.
    """

    name = "ibus"

    # A healthy IBus setup does not change under our feet, and the full probe
    # spawns ~3 subprocesses. The health watchdog calls probe() every 10s, so
    # cache a successful result and only re-run the full check every few
    # minutes. Failures are never cached — they must be re-checked immediately.
    _PROBE_CACHE_TTL = 300.0

    def __init__(self, platform=None, paste_keys: str = "ctrl+v",
                 terminal_paste_keys: str = "", **kwargs):
        self._wl_copy = shutil.which("wl-copy")
        self._ydotool = shutil.which("ydotool")
        self._wtype = shutil.which("wtype")
        self._xdotool = shutil.which("xdotool")
        self._keys = paste.PasteKeys(paste_keys, terminal_paste_keys)
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
        self._commit(text)

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
        self._commit(text)

    def _commit(self, text: str) -> None:
        self._send(f"commit:{text}")
        self._copy_to_clipboard(text)
        if not self._engine_has_focus():
            self._simulate_paste()
            return
        # A long note goes out in timed chunks; returning early would let the
        # caller restore the input source while the tail is still pending.
        time.sleep(commit_secs(text))

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

    def _paste_tool(self) -> str | None:
        if self._ydotool and ydotoold_running():
            return "ydotool"
        wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        if wayland and self._wtype:
            return "wtype"
        if not wayland and self._xdotool:
            return "xdotool"
        return None

    def _simulate_paste(self) -> None:
        """Press the paste chord for an app IBus does not reach (terminals)."""
        tool = self._paste_tool()
        if tool is None:
            log.warning("No paste tool available (ydotoold not running, no wtype/xdotool)")
            return
        try:
            paste.press(tool, *self._keys.for_focus())
            log.debug("Pasted via %s", tool)
        except subprocess.CalledProcessError as e:
            log.warning("%s paste failed (rc=%d): %s", tool, e.returncode,
                        (e.stderr or b"").decode(errors="replace").strip())
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
