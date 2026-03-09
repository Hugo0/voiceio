"""Platform detection: OS, display server, desktop environment, available tools."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache


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
        return "unknown"

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


@lru_cache(maxsize=1)
def detect() -> Platform:
    """Detect the current platform. Cached, safe to call multiple times."""
    plat_os = _detect_os()
    return Platform(
        os=plat_os,
        display_server=_detect_display_server(),
        desktop=_detect_desktop(),
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
