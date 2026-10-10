"""System probes for setup: binaries, audio/GPU/evdev checks, system-package preflight."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from voiceio.setup.ui import Spinner, _ask_yn
from voiceio.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _install_pip_package(package: str) -> bool:
    """Offer to pip-install a package. Returns True if installed successfully."""
    if not _ask_yn(f"Install {package}? (pip install {package})"):
        return False
    with Spinner(f"Installing {package}...") as sp:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True, timeout=120,
            )
            if result.returncode == 0:
                sp.ok(f"{package} installed!")
                return True
            sp.fail(f"pip install failed (exit {result.returncode})")
            stderr = result.stderr.decode()[:200]
            if stderr:
                print(f"  {DIM}{stderr}{RESET}")
            return False
        except subprocess.TimeoutExpired:
            sp.fail("Installation timed out")
            return False


def _wait_for_service_ready(timeout: float = 30) -> bool:
    """Wait for the voiceio service to finish loading by checking for its socket/PID file."""
    from voiceio.config import PID_PATH
    from voiceio.hotkeys.socket_backend import SOCKET_PATH, _IS_WINDOWS

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _IS_WINDOWS:
            # On Windows there's no Unix socket; check PID file as readiness signal
            if PID_PATH.exists():
                return True
        else:
            if SOCKET_PATH.exists():
                return True
        time.sleep(0.5)
    return False


def _check_system() -> dict:
    """Check system dependencies and capabilities."""
    checks = {}

    _is_win = sys.platform == "win32"
    _is_mac = sys.platform == "darwin"
    checks["is_windows"] = _is_win
    checks["is_mac"] = _is_mac
    checks["is_linux"] = not _is_win and not _is_mac

    # Display server
    if _is_win:
        checks["display"] = "win32"
    elif _is_mac:
        checks["display"] = "darwin"
    else:
        checks["display"] = os.environ.get("XDG_SESSION_TYPE", "unknown")

    # pynput (Windows/macOS primary backend)
    checks["pynput"] = False
    if _is_win or _is_mac:
        try:
            import pynput  # noqa: F401
            checks["pynput"] = True
        except ImportError:
            pass

    # Linux-only typer binaries
    checks["xdotool"] = _check_binary("xdotool") if checks["is_linux"] else False
    checks["xclip"] = _check_binary("xclip") if checks["is_linux"] else False
    checks["ydotool"] = _check_binary("ydotool") if checks["is_linux"] else False
    checks["wtype"] = _check_binary("wtype") if checks["is_linux"] else False
    checks["ibus"] = _check_binary("ibus") if checks["is_linux"] else False

    # IBus Python bindings (check system Python, not venv)
    checks["ibus_gi"] = False
    if checks["ibus"]:
        from voiceio.ibus.install import has_ibus_gi
        checks["ibus_gi"] = has_ibus_gi()

    # Audio
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        checks["audio"] = len(input_devs) > 0
        checks["audio_devices"] = input_devs
    except Exception:
        checks["audio"] = False
        checks["audio_devices"] = []

    # GPU
    try:
        import ctranslate2
        checks["cuda"] = "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        checks["cuda"] = False

    # evdev keyboard access (Linux only): udev uaccess rule or input group
    checks["evdev_access"] = False
    if checks["is_linux"]:
        from voiceio.hotkeys.evdev import EvdevHotkey
        checks["evdev_access"] = EvdevHotkey().probe().ok

    return checks


def _preflight_deps(interactive: bool = True, install_deps: bool = False) -> None:
    """Detect missing system packages and offer a single install command.

    The #1 install funnel killer is a missing system dependency (a C toolchain
    for the evdev build, PortAudio, IBus + gir bindings). We surface one
    copy-pasteable command for THIS distro and, interactively, offer to run it.
    """
    from voiceio import platform as plat
    missing = plat.check_system_deps()
    if not missing:
        return

    cmd = plat.system_deps_install_cmd(missing)
    print(f"\n  {YELLOW}⚠  Missing system dependencies:{RESET}")
    for key in missing:
        print(f"    {RED}✗{RESET} {plat.dep_label(key)}")
    print(f"\n  {DIM}Install them all with:{RESET}")
    print(f"    {BOLD}{cmd}{RESET}\n")

    should_run = install_deps
    if interactive and not install_deps and cmd:
        should_run = _ask_yn(f"Run this now? ({cmd})", default=True)
    if not should_run or not cmd:
        if not interactive:
            print(f"  {DIM}Run the command above, then re-run setup.{RESET}")
        return

    print(f"\n  {CYAN}Running:{RESET} {cmd}\n")
    try:
        result = subprocess.run(cmd.split(), timeout=600)
        if result.returncode == 0:
            print(f"\n  {GREEN}✓{RESET} System dependencies installed.")
            # Some deps (evdev build) only take effect on a fresh Python process.
            still = plat.check_system_deps()
            if still:
                print(f"  {YELLOW}ℹ{RESET}  Re-run {BOLD}voiceio setup{RESET} to pick up: "
                      f"{', '.join(plat.dep_label(k) for k in still)}")
        else:
            print(f"\n  {RED}✗{RESET} Install failed (exit {result.returncode}). Run the command above manually.")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"\n  {RED}✗{RESET} Could not run installer: {e}")
