"""Platform detection: OS, display server, desktop environment, available tools."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Platform:
    os: str                    # "linux", "darwin", "windows"
    display_server: str        # "x11", "wayland", "quartz", "unknown"
    desktop: str               # "gnome", "kde", "sway", "hyprland", "macos", "unknown"

    # Tool availability
    has_xdotool: bool = False
    has_ydotool: bool = False
    has_wtype: bool = False
    has_xclip: bool = False
    has_wl_copy: bool = False
    has_dotool: bool = False
    has_ibus: bool = False

    # Permissions
    has_input_group: bool = False
    has_uinput_access: bool = False

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"

    @property
    def is_mac(self) -> bool:
        return self.os == "darwin"

    @property
    def is_wayland(self) -> bool:
        return self.display_server == "wayland"

    @property
    def is_x11(self) -> bool:
        return self.display_server == "x11"

    @property
    def is_windows(self) -> bool:
        return self.os == "windows"

    @property
    def is_gnome(self) -> bool:
        return self.desktop in ("gnome", "unity")


def _detect_os() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    return "unknown"


def _detect_display_server() -> str:
    plat = _detect_os()
    if plat == "darwin":
        return "quartz"
    if plat == "windows":
        return "win32"

    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        return "wayland"
    if session == "x11":
        return "x11"

    # Fallback heuristics
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"

    return "unknown"


def _detect_desktop() -> str:
    plat = _detect_os()
    if plat == "darwin":
        return "macos"
    if plat == "windows":
        return "windows"

    raw = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

    if "gnome" in raw:
        return "gnome"
    if "kde" in raw or "plasma" in raw:
        return "kde"
    if "sway" in raw:
        return "sway"
    if "hyprland" in raw:
        return "hyprland"
    if raw:
        return raw.split(":")[0]  # take first component

    return "unknown"


def _check_input_group() -> bool:
    try:
        import grp
        input_gid = grp.getgrnam("input").gr_gid
        return input_gid in os.getgroups()
    except (KeyError, ImportError):
        return False


def _check_uinput_access() -> bool:
    try:
        with open("/dev/uinput", "rb"):
            pass
        return True
    except (PermissionError, FileNotFoundError, OSError):
        return False


_INSTALL_CMDS: dict[str, str] = {
    "apt": "sudo apt install",
    "dnf": "sudo dnf install",
    "pacman": "sudo pacman -S",
    "zypper": "sudo zypper install",
    "brew": "brew install",
}

# Package name overrides per package manager (when they differ from apt names)
_PKG_NAMES: dict[str, dict[str, str]] = {
    "dnf": {
        "gir1.2-ibus-1.0": "ibus-libs",
        "python3-gi": "python3-gobject",
        "gir1.2-ayatanaappindicator3-0.1": "libayatana-appindicator-gtk3",
        "espeak-ng": "espeak-ng",
    },
    "pacman": {
        "gir1.2-ibus-1.0": "ibus",
        "python3-gi": "python-gobject",
        "portaudio19-dev": "portaudio",
        "gir1.2-ayatanaappindicator3-0.1": "libayatana-appindicator",
        "espeak-ng": "espeak-ng",
    },
    "zypper": {
        "gir1.2-ibus-1.0": "typelib-1_0-IBus-1_0",
        "python3-gi": "python3-gobject",
        "gir1.2-ayatanaappindicator3-0.1": "typelib-1_0-AyatanaAppIndicator3-0_1",
        "espeak-ng": "espeak-ng",
    },
}


@lru_cache(maxsize=1)
def _detect_pkg_manager() -> str:
    """Detect the system package manager."""
    for mgr in ("apt", "dnf", "pacman", "zypper", "brew"):
        if shutil.which(mgr):
            return mgr
    return "apt"  # fallback to apt for hint text


def pkg_install(*packages: str) -> str:
    """Return an install command string for the detected package manager.

    Example: pkg_install("ibus", "gir1.2-ibus-1.0")
    → "sudo apt install ibus gir1.2-ibus-1.0"  (on Debian)
    → "sudo dnf install ibus ibus-libs"         (on Fedora)
    → "sudo pacman -S ibus"                     (on Arch)
    """
    mgr = _detect_pkg_manager()
    prefix = _INSTALL_CMDS.get(mgr, f"sudo {mgr} install")
    overrides = _PKG_NAMES.get(mgr, {})
    mapped = []
    seen = set()
    for pkg in packages:
        resolved = overrides.get(pkg, pkg)
        if resolved not in seen:
            seen.add(resolved)
            mapped.append(resolved)
    return f"{prefix} {' '.join(mapped)}"


def open_in_terminal(cmd: list[str]) -> bool:
    """Launch a command in the user's terminal emulator. Returns success."""
    import subprocess

    plat_os = _detect_os()

    if plat_os == "darwin":
        # macOS: use open -a Terminal
        try:
            subprocess.Popen(["open", "-a", "Terminal", "--args", *cmd])
            return True
        except OSError:
            log.warning("Failed to open Terminal.app")
            return False

    if plat_os == "windows":
        try:
            subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", *cmd])
            return True
        except OSError:
            log.warning("Failed to open cmd.exe")
            return False

    # Linux: try common terminal emulators
    term = os.environ.get("TERMINAL")
    if term and shutil.which(term):
        try:
            subprocess.Popen([term, "-e", *cmd])
            return True
        except OSError:
            pass

    # x-terminal-emulator (Debian/Ubuntu alternative)
    if shutil.which("x-terminal-emulator"):
        try:
            subprocess.Popen(["x-terminal-emulator", "-e", *cmd])
            return True
        except OSError:
            pass

    # Try specific terminals with their flags
    _TERMINALS = [
        (["gnome-terminal", "--"], "gnome-terminal"),
        (["konsole", "-e"], "konsole"),
        (["alacritty", "-e"], "alacritty"),
        (["kitty", "--"], "kitty"),
        (["foot", "--"], "foot"),
        (["wezterm", "start", "--"], "wezterm"),
        (["xfce4-terminal", "-e"], "xfce4-terminal"),
        (["xterm", "-e"], "xterm"),
    ]
    for prefix, binary in _TERMINALS:
        if shutil.which(binary):
            try:
                subprocess.Popen([*prefix, *cmd])
                return True
            except OSError:
                continue

    log.warning("No terminal emulator found")
    return False


@lru_cache(maxsize=1)
def detect() -> Platform:
    """Detect the current platform. Cached, safe to call multiple times."""
    plat_os = _detect_os()
    ds = _detect_display_server()
    desktop = _detect_desktop()
    log.debug("Platform detected: os=%s display_server=%s desktop=%s", plat_os, ds, desktop)
    return Platform(
        os=plat_os,
        display_server=ds,
        desktop=desktop,
        has_xdotool=shutil.which("xdotool") is not None,
        has_ydotool=shutil.which("ydotool") is not None,
        has_wtype=shutil.which("wtype") is not None,
        has_xclip=shutil.which("xclip") is not None,
        has_wl_copy=shutil.which("wl-copy") is not None,
        has_dotool=shutil.which("dotool") is not None,
        has_ibus=shutil.which("ibus") is not None,
        has_input_group=_check_input_group() if plat_os == "linux" else False,
        has_uinput_access=_check_uinput_access() if plat_os == "linux" else False,
    )
