"""Platform detection: OS, display server, desktop environment, available tools."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

# Microphone, hotkeys, tray and pynput typing ship in the `desktop` extra so a
# headless server install stays free of audio/input libraries.
DESKTOP_EXTRA_INSTALL = "pipx install --force 'python-voiceio[desktop]'"


class LazySounddevice:
    """`sounddevice`, imported on first use. Its import initializes PortAudio,
    which raises when libpulse is installed but no sound server runs (CI,
    headless boxes): imported eagerly, that broke `import voiceio.app` and
    every command behind it."""

    def __getattr__(self, name: str):
        if name.startswith("_"):  # introspection (mock, inspect) must not import it
            raise AttributeError(name)
        try:
            import sounddevice
        except ImportError as e:
            raise ImportError(desktop_dep_hint("sounddevice")) from e
        return getattr(sounddevice, name)


def desktop_dep_hint(package: str) -> str:
    """One-line message for a desktop dependency that failed to import."""
    from voiceio.install_env import running_ephemeral
    if running_ephemeral():
        # `uvx python-voiceio` resolves the base package only.
        fix = ("run it as uvx 'python-voiceio[desktop]' ..., or install it: "
               "uv tool install 'python-voiceio[desktop]'")
    else:
        fix = (f"{DESKTOP_EXTRA_INSTALL} (or uv tool install --force "
               "'python-voiceio[desktop]', or pip install 'python-voiceio[desktop]' in your venv)")
    return f"{package} is not installed. The dictation app needs voiceio's 'desktop' extra: {fix}"


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
    input_method: str = ""     # "ibus", "fcitx", or "" (from the session's IM env)

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


def _detect_input_method() -> str:
    """The session's input-method framework, as advertised to apps via env.

    fcitx5 (Omarchy's default, common for CJK users) and IBus are mutually
    exclusive: starting ibus-daemon under fcitx5 hijacks every text field.
    """
    for var in ("XMODIFIERS", "GTK_IM_MODULE", "QT_IM_MODULE"):
        val = os.environ.get(var, "").lower()
        if "fcitx" in val:
            return "fcitx"
        if "ibus" in val:
            return "ibus"
    return ""


def is_nixos() -> bool:
    return os.path.exists("/etc/NIXOS")


# Interpreters that may carry distro-packaged GObject bindings (gi). Order:
# FHS distros, then the NixOS system profile.
_SYSTEM_PYTHON_PATHS = (
    "/usr/bin/python3",
    "/usr/bin/python",
    "/run/current-system/sw/bin/python3",
)


_system_python_hits: dict[str, str] = {}


def find_system_python(probe: str) -> str | None:
    """First interpreter that runs the ``probe`` snippet successfully.

    GObject bindings ship as distro packages, so the venv/pipx interpreter
    running voiceio usually can't import them. ``$VOICEIO_SYSTEM_PYTHON``
    wins; then the running interpreter (Nix/Guix wrappers and venvs with
    PyGObject have voiceio's deps too), then well-known system paths and
    ``python3`` on PATH. Hits are cached; misses are not, so installing
    bindings takes effect without a restart.
    """
    import subprocess

    if probe in _system_python_hits:
        return _system_python_hits[probe]

    candidates = [
        os.environ.get("VOICEIO_SYSTEM_PYTHON", ""),
        sys.executable,
        *_SYSTEM_PYTHON_PATHS,
        shutil.which("python3") or "",
    ]
    seen: set[str] = set()
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        if not (os.path.isfile(py) and os.access(py, os.X_OK)):
            continue
        try:
            result = subprocess.run(
                [py, "-c", probe], capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("System python probe error (%s): %s", py, e)
            continue
        if result.returncode == 0:
            _system_python_hits[probe] = py
            return py
        log.debug("System python probe failed (%s): %s", py, result.stderr.strip()[:200])
    return None


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
    "xbps": "sudo xbps-install -S",
    "apk": "sudo apk add",
    # emerge asks before merging; _assume_yes drops --ask when nobody can answer.
    "emerge": "sudo emerge --ask",
    # NixOS has no imperative install; point at the declarative config.
    "nix": "add to environment.systemPackages:",
    "brew": "brew install",
}

# A package manager asks "continue? [Y/n]" on stdin; with no terminal to
# answer it the install aborts. Only added when the caller has no terminal and
# the user already said yes (setup --install-deps, a doctor --fix prompt).
_ASSUME_YES: dict[str, str] = {
    "apt": "-y", "apt-get": "-y", "dnf": "-y", "zypper": "-y",
    "pacman": "--noconfirm", "xbps-install": "-y",
}


def _has_tty() -> bool:
    """Can sudo ask for a password here? It reads /dev/tty, not stdin."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR)
    except OSError:
        return False
    os.close(fd)
    return True


def _runs_without_password(tool: str) -> bool:
    """`sudo -n true` / `doas -n true`: cached credentials or a no-password
    rule, so the tool works with no terminal to ask."""
    import subprocess
    try:
        return subprocess.run([tool, "-n", "true"], capture_output=True,
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _sudo_without_password() -> bool:
    return _runs_without_password("sudo")


def _doas_without_password() -> bool:
    return _runs_without_password("doas")


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


# Flags that make a package manager prompt; dropped instead (emerge has no
# subcommand to put a yes-flag after, and without --ask it never asks).
_ASK_FLAGS: dict[str, tuple[str, ...]] = {"emerge": ("--ask", "-a")}


def _assume_yes(argv: list[str]) -> list[str]:
    """`apt install x` -> `apt install -y x` (flag after the subcommand)."""
    if argv and (ask := _ASK_FLAGS.get(os.path.basename(argv[0]))):
        return [a for a in argv if a not in ask]
    flag = _ASSUME_YES.get(os.path.basename(argv[0])) if argv else None
    if not flag or flag in argv or len(argv) < 2:
        return argv
    return [argv[0], argv[1], flag, *argv[2:]]


def privileged_argv(argv: list[str]) -> list[str] | None:
    """How to run a command that needs root from here, or None if it can't be.

    A command needs root when it starts with sudo, or is a shell script that
    calls sudo. Commands are written with sudo; this picks the tool that
    exists. Already root: run it without sudo. With a terminal: sudo if
    installed, else doas (Gentoo, Alpine), else pkexec, each asking for the
    password. Without one (setup driven by an agent, a launcher, a GUI)
    nothing can ask on a terminal: `sudo -n` or `doas -n` when they need no
    password; else pkexec, whose polkit agent shows the desktop's password
    dialog. Else None, and the caller prints the command for the user to run
    as root (see privilege_unavailable_message).
    Ask the user before calling this: it never asks.
    """
    if not argv:
        return argv
    direct = argv[0] == "sudo"
    if not direct and not any("sudo " in a for a in argv[1:]):
        return argv
    # A script with inner sudo runs whole under the tool; as root its sudo
    # needs no password.
    rest = argv[1:] if direct else argv
    if _is_root() and direct:
        return rest
    has_sudo, has_doas = bool(shutil.which("sudo")), bool(shutil.which("doas"))
    has_pkexec = bool(shutil.which("pkexec"))
    if _has_tty():
        if has_sudo:
            return argv
        if has_doas:
            return ["doas", *rest]
        return ["pkexec", *rest] if has_pkexec else None
    rest = _assume_yes(rest) if direct else argv
    if has_sudo and _sudo_without_password():
        return ["sudo", "-n", *rest] if direct else argv
    if has_doas and _doas_without_password():
        return ["doas", "-n", *rest]
    if has_pkexec and (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        return ["pkexec", *rest]
    return None


def has_privilege_tool() -> bool:
    return any(shutil.which(t) for t in ("sudo", "doas", "pkexec"))


def as_root_command(argv: list[str]) -> str:
    """The command to type in a root shell: the argv without its sudo."""
    import re
    import shlex
    if argv and argv[0] == "sudo":
        argv = argv[1:]
    return shlex.join(re.sub(r"\bsudo (-n )?", "", a) for a in argv)


def privilege_unavailable_message(argv: list[str]) -> str:
    """Why privileged_argv gave up, and what to run instead (indented lines)."""
    import shlex
    if not has_privilege_tool():
        return ("  No sudo, doas or pkexec found. Run this as root "
                "(e.g. in a shell from `su -`):\n"
                f"    {as_root_command(argv)}")
    return ("  No terminal to ask for your password, and no pkexec. Run it yourself:\n"
            f"    {shlex.join(argv)}")


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
        "cudnn": "cudnn",
    },
    "zypper": {
        "gir1.2-ibus-1.0": "typelib-1_0-IBus-1_0",
        "python3-gi": "python3-gobject",
        "gir1.2-ayatanaappindicator3-0.1": "typelib-1_0-AyatanaAppIndicator3-0_1",
        "espeak-ng": "espeak-ng",
    },
    "xbps": {
        "gir1.2-ibus-1.0": "ibus",
        "python3-gi": "python3-gobject",
        "portaudio19-dev": "portaudio-devel",
        "gir1.2-ayatanaappindicator3-0.1": "libayatana-appindicator",
    },
    "apk": {
        "gir1.2-ibus-1.0": "ibus",
        "python3-gi": "py3-gobject3",
        "portaudio19-dev": "portaudio-dev",
        "gir1.2-ayatanaappindicator3-0.1": "libayatana-appindicator",
    },
    # Gentoo atoms. Names not listed stay bare (`emerge ydotool`), which
    # emerge resolves when the name is unique in the tree.
    "apt": {
        "cudnn": "nvidia-cudnn",
    },
    "emerge": {
        "ibus": "app-i18n/ibus",
        "gir1.2-ibus-1.0": "app-i18n/ibus",  # typelib comes with USE=introspection
        "python3-gi": "dev-python/pygobject",
        "portaudio19-dev": "media-libs/portaudio",
        "gir1.2-ayatanaappindicator3-0.1": "dev-libs/libayatana-appindicator",
        "espeak-ng": "app-accessibility/espeak-ng",
        "xdotool": "x11-misc/xdotool",
        "xclip": "x11-misc/xclip",
        "wl-clipboard": "gui-apps/wl-clipboard",
        "wtype": "gui-apps/wtype",
        "cudnn": "dev-libs/cudnn",
    },
    "nix": {
        "gir1.2-ibus-1.0": "ibus",
        "python3-gi": "python3Packages.pygobject3",
        "portaudio19-dev": "portaudio",
        "gir1.2-ayatanaappindicator3-0.1": "libayatana-appindicator",
    },
}


@lru_cache(maxsize=1)
def _detect_pkg_manager() -> str:
    """Detect the system package manager."""
    if is_nixos():
        return "nix"
    for mgr, binary in (
        ("apt", "apt"), ("dnf", "dnf"), ("pacman", "pacman"), ("zypper", "zypper"),
        ("xbps", "xbps-install"), ("apk", "apk"), ("emerge", "emerge"), ("brew", "brew"),
    ):
        if shutil.which(binary):
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


# ── CUDA libraries ───────────────────────────────────────────────────────────

# What CTranslate2 4.x's CUDA build loads at the first encode.
CUDA_LIBS = ("libcublas.so.12", "libcudnn.so.9")


def missing_cuda_libs() -> list[str] | None:
    """The CUDA_LIBS the loader can't open, or None when CTranslate2 sees no
    NVIDIA GPU (or has no CUDA build). [] means the GPU is usable."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            return None
    except Exception:  # noqa: BLE001 — no faster-whisper, or no CUDA build
        return None
    import ctypes
    missing = []
    for lib in CUDA_LIBS:
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    return missing


def nvidia_gpu_present() -> bool:
    """An NVIDIA GPU with its kernel driver loaded, asked of the OS rather
    than of CTranslate2 (which reports 0 devices when libcuda won't load)."""
    if os.path.exists("/proc/driver/nvidia/version") or os.path.exists("/dev/nvidia0"):
        return True
    if not shutil.which("nvidia-smi"):
        return False
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "GPU" in r.stdout


def cuda_device_problem() -> str:
    """Why CTranslate2 sees no CUDA device on a machine that has an NVIDIA
    GPU, or "" (no GPU, no CTranslate2, or it sees the device)."""
    try:
        import ctranslate2
    except Exception:  # noqa: BLE001 — no faster-whisper installed
        return ""
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return ""
    except Exception:  # noqa: BLE001 — treat a failing query as no device
        pass
    if not nvidia_gpu_present():
        return ""
    import ctypes
    try:
        ctypes.CDLL("libcuda.so.1")
        why = ""
    except OSError:
        why = " (libcuda.so.1 can't be loaded)"
    return (f"NVIDIA GPU found, but CTranslate2 sees no CUDA device{why}: check that "
            "the driver's libcuda.so.1 loads (ldconfig -p | grep libcuda.so.1) and "
            "that nvidia-smi works.")


def cuda_libs_hint(missing: list[str]) -> str:
    """Where to look for the libraries missing_cuda_libs() named."""
    hint = ""
    if any("cudnn" in lib for lib in missing):
        hint = "cuDNN 9 is a separate package from the CUDA toolkit"
        # Only the managers whose package name we know; others stay generic.
        mgr = _detect_pkg_manager()
        if "cudnn" in _PKG_NAMES.get(mgr, {}):
            # Older Debian/Ubuntu releases package cuDNN 8; NVIDIA's repo has 9.
            alt = " or NVIDIA's CUDA repo" if mgr == "apt" else ""
            hint += f" ({pkg_install('cudnn')}{alt})"
        hint += ". "
    return (hint + "The libraries must be on the loader path: "
            "ldconfig -p | grep -E 'cublas.so.12|cudnn.so.9'")


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
        "xbps": ["base-devel", "python3-devel"],
        "apk": ["build-base", "python3-dev", "linux-headers"],
        "emerge": ["sys-devel/gcc"],  # Python headers ship with dev-lang/python
        "nix": ["gcc"],
    },
    # PortAudio backs sounddevice (microphone capture).
    "portaudio": {
        "apt": ["portaudio19-dev"],
        "dnf": ["portaudio-devel"],
        "pacman": ["portaudio"],
        "zypper": ["portaudio-devel"],
        "xbps": ["portaudio-devel"],
        "apk": ["portaudio-dev"],
        "emerge": ["media-libs/portaudio"],
        "nix": ["portaudio"],
    },
    # IBus engine + GObject bindings — the only reliable Wayland text injection.
    "ibus": {
        "apt": ["ibus", "gir1.2-ibus-1.0", "python3-gi"],
        "dnf": ["ibus", "ibus-libs", "python3-gobject"],
        "pacman": ["ibus", "python-gobject"],
        "zypper": ["ibus", "typelib-1_0-IBus-1_0", "python3-gobject"],
        "xbps": ["ibus", "python3-gobject"],
        "apk": ["ibus", "py3-gobject3"],
        "emerge": ["app-i18n/ibus", "dev-python/pygobject"],
        "nix": ["ibus", "python3Packages.pygobject3"],
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


def _portaudio_lib_present() -> bool:
    """Is the PortAudio shared library installed? The lookup sounddevice does."""
    import ctypes.util
    return ctypes.util.find_library("portaudio") is not None


def _probe_portaudio() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except OSError:
        return False  # sounddevice is there, the PortAudio library is not
    except ImportError:
        # The pip package is missing (no [desktop] extra), which says nothing
        # about the system library: audio_problem() names the extra instead.
        return _portaudio_lib_present()


def audio_problem() -> str:
    """Why sounddevice can't be used, or "" when it imports. A missing pip
    package (the desktop extra) and a missing system PortAudio need
    different fixes, so they get different messages."""
    try:
        import sounddevice  # noqa: F401
    except OSError:
        return "PortAudio library not found — install: " + system_deps_install_cmd(["portaudio"])
    except ImportError:
        return desktop_dep_hint("sounddevice")
    except Exception as e:  # PortAudioError: library there, sound server isn't
        return f"PortAudio could not start: {e}"
    return ""


def _probe_ibus() -> bool:
    if shutil.which("ibus") is None:
        return False
    try:
        from voiceio.ibus.install import has_ibus_gi
        return has_ibus_gi()
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


def open_in_terminal(cmd: list[str], *, hold: bool = False) -> bool:
    """Launch a command in the user's terminal emulator. Returns success.

    ``hold=True`` keeps the window open after the command exits — required for
    print-and-exit commands (history, doctor, logs) that would otherwise flash
    a terminal that closes the instant they finish.
    """
    import shlex
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

    # Linux: try common terminal emulators. When holding, wrap the command in a
    # shell that pauses at the end so the window survives a print-and-exit.
    launch = cmd
    if hold:
        inner = " ".join(shlex.quote(c) for c in cmd)
        launch = [
            "sh", "-c",
            f'{inner}; printf "\\n[voiceio] Press Enter to close…"; read _',
        ]

    term = os.environ.get("TERMINAL")
    if term and shutil.which(term):
        try:
            subprocess.Popen([term, "-e", *launch])
            return True
        except OSError:
            pass

    # x-terminal-emulator (Debian/Ubuntu alternative)
    if shutil.which("x-terminal-emulator"):
        try:
            subprocess.Popen(["x-terminal-emulator", "-e", *launch])
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
                subprocess.Popen([*prefix, *launch])
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
        input_method=_detect_input_method() if plat_os == "linux" else "",
        has_input_group=_check_input_group() if plat_os == "linux" else False,
        has_uinput_access=_check_uinput_access() if plat_os == "linux" else False,
    )


def redetect() -> Platform:
    """Clear the cached Platform and re-run detection.

    detect() is @lru_cache'd, so the first call freezes the result.
    When voiceio starts before the desktop session has exported its env
    vars, that first call returns display='unknown' and we'd be stuck
    with that forever. Always clear the cache before re-detecting.
    """
    detect.cache_clear()
    return detect()


# Env vars needed for clipboard/tray/typing on graphical sessions.
# XDG_CURRENT_DESKTOP is required for is_gnome() detection, which gates
# IBus input source configuration.
_GRAPHICAL_ENV_VARS = (
    "DISPLAY",
    # Without the X auth cookie, X11 tools (xclip via the XWayland bridge)
    # fail silently inside the daemon and clipboard copies fall back to
    # wl-copy — whose focus steal discards any visible IBus preedit.
    "XAUTHORITY",
    "WAYLAND_DISPLAY",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
)

_graphical_env_complete = False


def import_graphical_env() -> None:
    """Pull graphical session env vars from the systemd user manager.

    When started as a systemd user service, the process may launch before
    the desktop session imports DISPLAY/WAYLAND_DISPLAY. This queries
    ``systemctl --user show-environment`` to pick them up after the fact.

    Once all expected vars are present, sets ``_graphical_env_complete``
    and becomes a free no-op so callers in the health loop don't spawn
    a subprocess every 10s.
    """
    import subprocess

    global _graphical_env_complete
    if _graphical_env_complete:
        return
    missing = [v for v in _GRAPHICAL_ENV_VARS if v not in os.environ]
    if not missing:
        _graphical_env_complete = True
        return
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show-environment"],
            text=True, timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return
    for line in out.splitlines():
        key, _, val = line.partition("=")
        if key in missing and val:
            os.environ[key] = val
            log.info("Imported %s=%s from systemd user env", key, val)
    if not [v for v in _GRAPHICAL_ENV_VARS if v not in os.environ]:
        _graphical_env_complete = True
