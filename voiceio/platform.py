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


# ── System-dependency preflight ──────────────────────────────────────────────
#
# The #1 install funnel killer is missing *system* packages: a C toolchain to
# build the evdev wheel, PortAudio for microphone capture, and IBus + its
# GObject-introspection bindings for text injection. This is the single source
# of truth mapping each requirement to the exact package names per distro.

SYSTEM_DEPS: dict[str, dict[str, list[str]]] = {
    # Builds the evdev C-extension (no manylinux wheel is published for evdev).
    "compiler": {
        "apt": ["build-essential", "python3-dev"],
        "dnf": ["gcc", "gcc-c++", "make", "python3-devel"],
        "pacman": ["base-devel"],
        "zypper": ["gcc", "gcc-c++", "make", "python3-devel"],
    },
    # PortAudio backs sounddevice (microphone capture).
    "portaudio": {
        "apt": ["portaudio19-dev"],
        "dnf": ["portaudio-devel"],
        "pacman": ["portaudio"],
        "zypper": ["portaudio-devel"],
    },
    # IBus engine + GObject bindings — the only reliable Wayland text injection.
    "ibus": {
        "apt": ["ibus", "gir1.2-ibus-1.0", "python3-gi"],
        "dnf": ["ibus", "ibus-libs", "python3-gobject"],
        "pacman": ["ibus", "python-gobject"],
        "zypper": ["ibus", "typelib-1_0-IBus-1_0", "python3-gobject"],
    },
}

_DEP_LABELS: dict[str, str] = {
    "compiler": "C compiler + Python headers (builds the evdev hotkey backend)",
    "portaudio": "PortAudio (microphone capture)",
    "ibus": "IBus + GObject bindings (text injection)",
}


def _probe_compiler() -> bool:
    # If evdev already imports, the toolchain did its job at install time and
    # is no longer needed at runtime.
    try:
        import evdev  # noqa: F401
        return True
    except ImportError:
        pass
    return shutil.which("cc") is not None or shutil.which("gcc") is not None


def _probe_portaudio() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except (OSError, ImportError):
        return False


def _probe_ibus() -> bool:
    if shutil.which("ibus") is None:
        return False
    try:
        from voiceio.typers.ibus import _has_ibus_gi
        return _has_ibus_gi()
    except Exception:
        return False


_DEP_PROBES = {
    "compiler": _probe_compiler,
    "portaudio": _probe_portaudio,
    "ibus": _probe_ibus,
}


def check_system_deps() -> list[str]:
    """Return canonical keys of missing/broken system dependencies (Linux only).

    Each key indexes both SYSTEM_DEPS (package names) and _DEP_LABELS (human
    text). Empty list on non-Linux or when everything is satisfied.
    """
    if _detect_os() != "linux":
        return []
    missing: list[str] = []
    for key, probe in _DEP_PROBES.items():
        try:
            ok = probe()
        except Exception:
            ok = False
        if not ok:
            missing.append(key)
    return missing


def dep_label(key: str) -> str:
    """Human-readable label for a system-dependency key."""
    return _DEP_LABELS.get(key, key)


def system_deps_install_cmd(keys: list[str] | None = None) -> str:
    """One copy-pasteable install command covering ``keys`` on this distro.

    With no keys, covers every requirement in SYSTEM_DEPS. Package names are
    de-duplicated and ordered; falls back to apt names for unknown managers.
    """
    if keys is None:
        keys = list(SYSTEM_DEPS)
    mgr = _detect_pkg_manager()
    prefix = _INSTALL_CMDS.get(mgr, f"sudo {mgr} install")
    pkgs: list[str] = []
    seen: set[str] = set()
    for key in keys:
        by_mgr = SYSTEM_DEPS.get(key, {})
        for pkg in by_mgr.get(mgr, by_mgr.get("apt", [])):
            if pkg not in seen:
                seen.add(pkg)
                pkgs.append(pkg)
    if not pkgs:
        return ""
    return f"{prefix} {' '.join(pkgs)}"


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
