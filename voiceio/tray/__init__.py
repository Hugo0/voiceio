"""System tray icon with animated recording indicator.

Public API (called from app.py):
    tray.start(quit_callback, toggle_callback)  - launch tray icon
    tray.set_recording(True/False)              - switch visual state
    tray.stop()                                 - tear down

Backend selection:
    Linux: spawns _indicator.py under /usr/bin/python3 (system GTK3 +
           AppIndicator3, zero pip deps on Ubuntu/Fedora/Arch).
    macOS: in-process pystray (PyObjC is preinstalled).
"""
from __future__ import annotations

import atexit
import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None
_theme_dir: Path | None = None
_backend: str | None = None  # "indicator" | "pystray" | None
_stdout_thread: threading.Thread | None = None


_APPINDICATOR_PROBE = (
    "import gi; gi.require_version('Gtk','3.0'); "
    "from gi.repository import Gtk, GLib\n"
    "try:\n"
    "    gi.require_version('AyatanaAppIndicator3','0.1')\n"
    "    from gi.repository import AyatanaAppIndicator3\n"
    "except (ValueError, ImportError):\n"
    "    gi.require_version('AppIndicator3','0.1')\n"
    "    from gi.repository import AppIndicator3\n"
    "print('ok')\n"
)


def _find_system_python() -> str | None:
    """Find a system Python 3 that has GTK3 + AppIndicator3."""
    for py in ["/usr/bin/python3"]:
        if not shutil.which(py):
            continue
        try:
            result = subprocess.run(
                [py, "-c", _APPINDICATOR_PROBE],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return py
            log.debug("System python probe failed: %s", result.stderr.strip()[:200])
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.debug("System python probe error: %s", e)
            continue
    return None


def probe_availability() -> tuple[bool, str, str]:
    """Check if a tray icon backend is available.

    Returns (ok, reason, fix_hint) for use by health check / wizard.
    """
    if sys.platform.startswith("linux"):
        py = _find_system_python()
        if py:
            return True, "AppIndicator3 available", ""
        # Determine why it failed
        if not shutil.which("/usr/bin/python3"):
            return False, "system python3 not found", ""
        from voiceio.platform import pkg_install
        return (
            False,
            "system python3 missing GTK3/AppIndicator3 bindings",
            f"install: {pkg_install('gir1.2-ayatanaappindicator3-0.1')}",
        )
    elif sys.platform == "darwin":
        try:
            import pystray  # noqa: F401
            return True, "pystray available", ""
        except ImportError:
            return False, "pystray not installed", "pip install pystray"
    elif sys.platform == "win32":
        try:
            import pystray  # noqa: F401
            return True, "pystray available (Win32 tray)", ""
        except ImportError:
            return False, "pystray not installed", "pip install pystray Pillow"
    return False, "unsupported platform", ""


def _start_indicator(
    theme_dir: Path,
    icon_names: list[str],
    interval_ms: int,
    system_python: str,
) -> subprocess.Popen:
    from voiceio.tray._icons import FRAME_COUNT
    script = str(Path(__file__).parent / "_indicator.py")
    idle_icons = ",".join(icon_names[0:FRAME_COUNT])
    rec_icons = ",".join(icon_names[FRAME_COUNT:2 * FRAME_COUNT])
    proc_icons = ",".join(icon_names[2 * FRAME_COUNT:])
    return subprocess.Popen(
        [
            system_python, script,
            "--theme-dir", str(theme_dir),
            "--idle-icons", idle_icons,
            "--rec-icons", rec_icons,
            "--proc-icons", proc_icons,
            "--interval", str(interval_ms),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_stdout(proc: subprocess.Popen, toggle_cb: Callable[[], None]) -> None:
    """Read toggle commands from indicator subprocess stdout."""
    try:
        while proc.poll() is None and proc.stdout:
            line = proc.stdout.readline()
            if not line:
                break
            cmd = line.decode().strip()
            if cmd == "toggle":
                toggle_cb()
    except (OSError, ValueError):
        pass


def _send(command: str) -> None:
    if _proc is not None and _proc.stdin is not None and _proc.poll() is None:
        try:
            _proc.stdin.write(f"{command}\n".encode())
            _proc.stdin.flush()
        except (BrokenPipeError, OSError):
            log.debug("Tray subprocess pipe broken")


def _cleanup() -> None:
    global _theme_dir
    if _theme_dir is not None:
        shutil.rmtree(_theme_dir, ignore_errors=True)
        _theme_dir = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(
    quit_callback: Callable[[], None],
    toggle_callback: Callable[[], None] | None = None,
) -> None:
    """Start the tray icon using the best available backend."""
    global _proc, _theme_dir, _backend, _stdout_thread

    from voiceio.tray._icons import render_to_dir, ANIM_INTERVAL_MS

    _theme_dir, icon_names = render_to_dir()
    atexit.register(_cleanup)

    # Linux: AppIndicator subprocess
    if sys.platform.startswith("linux"):
        system_python = _find_system_python()
        if system_python:
            log.info("Tray: using AppIndicator via %s", system_python)
            _proc = _start_indicator(
                _theme_dir, icon_names, ANIM_INTERVAL_MS, system_python,
            )
            _backend = "indicator"

            if toggle_callback is not None:
                _stdout_thread = threading.Thread(
                    target=_read_stdout,
                    args=(_proc, toggle_callback),
                    daemon=True,
                    name="tray-stdout",
                )
                _stdout_thread.start()
            return
        else:
            log.info("Tray: system python3 GTK3 not found, trying pystray")

    # macOS or Linux fallback: in-process pystray
    from voiceio.tray._pystray import start as pystray_start

    # pystray needs flat PNG paths — extract from theme dir
    from voiceio.tray._icons import FRAME_COUNT
    apps_dir = _theme_dir / "hicolor"
    # Find the apps subdir
    for d in apps_dir.rglob("apps"):
        idle_path = d / f"{icon_names[0]}.png"
        frame_paths = [d / f"{n}.png" for n in icon_names[FRAME_COUNT:]]
        break
    else:
        log.warning("Tray: could not find icon PNGs")
        _cleanup()
        return

    if pystray_start(quit_callback, idle_path, frame_paths):
        _backend = "pystray"
        log.info("Tray: using pystray")
    else:
        log.warning("Tray: no backend available")
        _cleanup()


def set_recording(recording: bool) -> None:
    """Switch between idle and recording visual states."""
    if _backend == "indicator":
        _send("recording" if recording else "idle")
    elif _backend == "pystray":
        from voiceio.tray._pystray import set_recording as pystray_set
        pystray_set(recording)


def set_processing(processing: bool) -> None:
    """Show processing/finalizing animation in tray icon."""
    if _backend == "indicator":
        _send("processing" if processing else "idle")
    elif _backend == "pystray":
        from voiceio.tray._pystray import set_title
        set_title("voiceio - processing" if processing else "voiceio - idle")


def set_error(error: bool) -> None:
    """Show error state in tray icon (same as idle for now, just updates title)."""
    if _backend == "indicator":
        _send("error" if error else "idle")
    elif _backend == "pystray":
        from voiceio.tray._pystray import set_title
        set_title("voiceio - error" if error else "voiceio - idle")


def stop() -> None:
    """Stop the tray icon and clean up."""
    global _proc, _backend, _stdout_thread

    if _backend == "indicator" and _proc is not None:
        _send("quit")
        try:
            _proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _proc.kill()
        _proc = None
    elif _backend == "pystray":
        from voiceio.tray._pystray import stop as pystray_stop
        pystray_stop()

    _backend = None
    _stdout_thread = None
    _cleanup()
