"""Service installation and management.

Linux: systemd user service.
Windows: Startup folder .bat shortcut.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

LOCAL_BIN = Path.home() / ".local" / "bin"
# `voiceio-toggle` stays for 1.x: GNOME shortcuts created before 1.0 call it.
SCRIPT_NAMES = ["voiceio", "voiceio-toggle"]
# v0 entry points; their symlinks dangle (or run old code) after an upgrade.
_LEGACY_SCRIPT_NAMES = ["voiceio-doctor", "voiceio-setup", "voiceio-test"]
_PATH_HINT_ADDED = False  # track if we already printed the PATH hint

SERVICE_NAME = "voiceio.service"
SERVICE_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_PATH = SERVICE_DIR / SERVICE_NAME

# Nightly `voiceio learn maintain` (opt-in: `voiceio learn schedule on`)
LEARN_SERVICE_NAME = "voiceio-learn.service"
LEARN_TIMER_NAME = "voiceio-learn.timer"
LEARN_SERVICE_PATH = SERVICE_DIR / LEARN_SERVICE_NAME
LEARN_TIMER_PATH = SERVICE_DIR / LEARN_TIMER_NAME

# Pre-1.0 weekly cloud mining timer — only referenced to remove it
CORRECT_SERVICE_NAME = "voiceio-correct.service"
CORRECT_TIMER_NAME = "voiceio-correct.timer"
CORRECT_SERVICE_PATH = SERVICE_DIR / CORRECT_SERVICE_NAME
CORRECT_TIMER_PATH = SERVICE_DIR / CORRECT_TIMER_NAME


def has_systemd() -> bool:
    """Check if systemd is available on this system."""
    return shutil.which("systemctl") is not None


def _find_voiceio_bin() -> str:
    """Find the voiceio binary path."""
    # Check venv first (Scripts/ on Windows, bin/ elsewhere)
    scripts_dir = "Scripts" if _IS_WINDOWS else "bin"
    venv_bin = Path(sys.prefix) / scripts_dir / "voiceio"
    if venv_bin.exists():
        return str(venv_bin.resolve())
    found = shutil.which("voiceio")
    if found:
        return str(Path(found).resolve())
    return "voiceio"


def _service_unit(bin_path: str) -> str:
    """Generate the systemd unit file content."""
    return f"""\
[Unit]
Description=VoiceIO - voice-to-text input
Documentation=https://github.com/Hugo0/voiceio
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart={bin_path} run
# Dictation is interactive: someone is waiting on every decode. Not Nice=10 —
# nice only ranks processes inside one cgroup, so it never shielded the
# compositor (session.slice) and only made decodes lose to every terminal and
# browser in app.slice (25s for 8s of audio at load 26). A cgroup weight wins
# bursts against those siblings and is a no-op where cpu isn't delegated.
CPUWeight=400
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1
PassEnvironment=DISPLAY WAYLAND_DISPLAY XDG_SESSION_TYPE XDG_RUNTIME_DIR

[Install]
WantedBy=graphical-session.target
"""


def _learn_units(bin_path: str, auto_promote: bool = False) -> tuple[str, str]:
    """(service, timer) for nightly learning on this machine."""
    flags = " --auto-promote" if auto_promote else ""
    service = f"""\
[Unit]
Description=VoiceIO - learn from today's dictation (label, retrain when enough is new)
Documentation=https://github.com/Hugo0/voiceio#make-it-yours

[Service]
Type=oneshot
ExecStart={bin_path} learn maintain{flags}
# Background work: it gets the CPU only when nothing else wants it, so
# dictation and everything else stay fast. The unit's own cgroup already
# does what `learn` would otherwise re-exec into a scope for.
CPUWeight=idle
IOWeight=10
Nice=19
Environment=VOICEIO_IDLE_SCOPE=1 PYTHONUNBUFFERED=1
"""
    timer = """\
[Unit]
Description=VoiceIO - nightly learning

[Timer]
OnCalendar=*-*-* 03:30
RandomizedDelaySec=30min
# A laptop that was asleep at 03:30 catches up when it next runs.
Persistent=true

[Install]
WantedBy=timers.target
"""
    return service, timer


def _systemctl(*args: str) -> bool:
    try:
        return subprocess.run(["systemctl", "--user", *args], capture_output=True,
                              timeout=10).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_learn_timer(auto_promote: bool = False) -> bool:
    """Run `voiceio learn maintain` nightly. True when the timer is enabled."""
    if _IS_WINDOWS or not has_systemd():
        return False
    service, timer = _learn_units(_find_voiceio_bin(), auto_promote)
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    LEARN_SERVICE_PATH.write_text(service)
    LEARN_TIMER_PATH.write_text(timer)
    _systemctl("daemon-reload")
    return _systemctl("enable", "--now", LEARN_TIMER_NAME)


def uninstall_learn_timer() -> None:
    _systemctl("disable", "--now", LEARN_TIMER_NAME)
    for p in (LEARN_TIMER_PATH, LEARN_SERVICE_PATH):
        p.unlink(missing_ok=True)
    _systemctl("daemon-reload")


def learn_timer_enabled() -> bool:
    return LEARN_TIMER_PATH.exists() and _systemctl("is-enabled", "--quiet", LEARN_TIMER_NAME)


def learn_timer_auto_promotes() -> bool:
    try:
        return "--auto-promote" in LEARN_SERVICE_PATH.read_text()
    except OSError:
        return False


def restart_when_idle() -> bool:
    """Restart the daemon once no dictation is in progress."""
    from voiceio.cli.servicecmd import wait_until_idle
    wait_until_idle()
    return _systemctl("restart", SERVICE_NAME)


def uninstall_correct_timer() -> None:
    """Remove the pre-1.0 weekly cloud-mining timer (`correct --auto --batch`).
    It is gone in 1.0: mining moved to the local, opt-in `voiceio learn`."""
    if _IS_WINDOWS:
        return
    try:
        subprocess.run(["systemctl", "--user", "disable", "--now", CORRECT_TIMER_NAME],
                       capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    for p in (CORRECT_TIMER_PATH, CORRECT_SERVICE_PATH):
        if p.exists():
            p.unlink()
            log.info("Removed %s", p)


def _windows_startup_dir() -> Path:
    """Return the Windows Startup folder path."""
    appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _windows_startup_bat() -> Path:
    return _windows_startup_dir() / "voiceio.bat"


def install_windows_startup() -> bool:
    """Create a .bat file in the Windows Startup folder."""
    bat_path = _windows_startup_bat()
    voiceio_bin = _find_voiceio_bin()
    bat_path.parent.mkdir(parents=True, exist_ok=True)
    bat_path.write_text(f'@echo off\nstart /B "" "{voiceio_bin}"\n')
    log.info("Windows startup script created: %s", bat_path)
    return True


def uninstall_windows_startup() -> bool:
    """Remove the voiceio .bat from the Windows Startup folder."""
    bat_path = _windows_startup_bat()
    try:
        bat_path.unlink()
        log.info("Removed Windows startup script: %s", bat_path)
        return True
    except FileNotFoundError:
        return False


def install_service() -> bool:
    """Install autostart (systemd on Linux, Startup folder on Windows).

    Returns True if installed successfully.
    """
    if _IS_WINDOWS:
        return install_windows_startup()

    if not has_systemd():
        log.warning("systemctl not found: cannot install service")
        return False

    bin_path = _find_voiceio_bin()

    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    SERVICE_PATH.write_text(_service_unit(bin_path))
    log.info("Installed systemd service to %s", SERVICE_PATH)

    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True, timeout=5,
        )
        # disable before enable so a prior [Install] target (older versions
        # wired WantedBy=default.target, which never restarts the service on a
        # GNOME logout/login) is unlinked instead of left as a stale want.
        subprocess.run(
            ["systemctl", "--user", "disable", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
        log.info("Enabled %s", SERVICE_NAME)
        uninstall_correct_timer()
        install_symlinks()  # the CLI must run the same install as the service
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Could not enable service: %s", e)
        return False


def uninstall_service() -> bool:
    """Disable and remove autostart."""
    if _IS_WINDOWS:
        return uninstall_windows_startup()

    try:
        subprocess.run(
            ["systemctl", "--user", "disable", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["systemctl", "--user", "stop", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    uninstall_correct_timer()

    if SERVICE_PATH.exists():
        SERVICE_PATH.unlink()
        log.info("Removed %s", SERVICE_PATH)

    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return True


def is_installed() -> bool:
    """Check if autostart is installed."""
    if _IS_WINDOWS:
        return _windows_startup_bat().exists()
    return SERVICE_PATH.exists()


def is_running() -> bool:
    """Check if the service is currently running."""
    if _IS_WINDOWS:
        # Check PID file — works for both pip installs (python.exe) and .exe builds
        from voiceio.config import PID_PATH
        try:
            pid = int(PID_PATH.read_text().strip())
            # On Windows, check if process with this PID exists
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_service() -> bool:
    """Start the voiceio systemd user service."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _is_pipx_install() -> bool:
    """Check if voiceio is running from a pipx-managed venv."""
    return "pipx/venvs" in sys.prefix


def _local_bin_on_path() -> bool:
    """Check if LOCAL_BIN is in the current PATH."""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    return str(LOCAL_BIN) in path_dirs or str(LOCAL_BIN.resolve()) in path_dirs


def install_symlinks() -> list[str]:
    """Create symlinks in ~/.local/bin/ pointing to venv scripts.

    Returns list of names successfully linked.
    For pipx installs, scripts are already in ~/.local/bin/ as real files,
    so we skip symlink creation. On Windows, pip/pipx handles PATH placement.
    """
    if _IS_WINDOWS:
        # On Windows, pip/pipx places scripts in Scripts/ which is on PATH
        found = [name for name in SCRIPT_NAMES if shutil.which(name)]
        log.debug("Windows: found %d scripts on PATH: %s", len(found), found)
        return found

    if _is_pipx_install():
        # pipx already placed scripts in ~/.local/bin/, nothing to do
        return [name for name in SCRIPT_NAMES if (LOCAL_BIN / name).exists()]

    venv_bin = Path(sys.prefix) / "bin"
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    for name in _LEGACY_SCRIPT_NAMES:
        dest = LOCAL_BIN / name
        if dest.is_symlink():
            dest.unlink()
            log.info("Removed legacy %s", dest)
    linked = []
    for name in SCRIPT_NAMES:
        src = venv_bin / name
        dest = LOCAL_BIN / name
        if not src.exists():
            continue
        # Already correct
        if dest.is_symlink() and dest.resolve() == src.resolve():
            linked.append(name)
            continue
        # Remove stale symlink; skip regular files
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink():
                log.warning("Skipping %s: regular file exists at %s", name, dest)
                continue
            dest.unlink()
        dest.symlink_to(src.resolve())
        log.info("Linked %s → %s", dest, src.resolve())
        linked.append(name)

    # On macOS (or any system where ~/.local/bin isn't on PATH), add it to shell profile
    if linked and not _local_bin_on_path():
        _add_local_bin_to_path()

    return linked


def _add_local_bin_to_path() -> None:
    """Add ~/.local/bin to PATH in the user's active shell rc file."""
    global _PATH_HINT_ADDED
    line = '\nexport PATH="$HOME/.local/bin:$PATH"\n'

    # Detect the user's actual shell and target its rc file.
    # .profile is sourced by login shells only — interactive terminals
    # (especially zsh on Ubuntu) may never read it.
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        targets = [".zshrc", ".profile"]
    elif "bash" in shell:
        targets = [".bashrc", ".profile"]
    else:
        targets = [".zshrc", ".bashrc", ".profile"]

    for rc_name in targets:
        rc = Path.home() / rc_name
        if rc.exists():
            content = rc.read_text()
            if ".local/bin" in content:
                continue  # already in this file, check next
            rc.write_text(content + line)
            log.info("Added ~/.local/bin to PATH in %s", rc)
            _PATH_HINT_ADDED = True
            return

    # No existing rc found — create the shell-appropriate one
    rc = Path.home() / (targets[0] if targets else ".profile")
    rc.write_text(line)
    log.info("Created %s with PATH entry", rc)
    _PATH_HINT_ADDED = True
    _PATH_HINT_ADDED = True


def symlinks_installed() -> bool:
    """Check if voiceio is accessible as a command."""
    # Check if it's anywhere on PATH
    if shutil.which("voiceio"):
        return True
    # Check if symlink exists (even if not on PATH yet, a new shell will pick it up)
    dest = LOCAL_BIN / "voiceio"
    return dest.exists()


def path_hint_needed() -> bool:
    """Return True if we modified shell profile and user needs to restart shell."""
    return _PATH_HINT_ADDED
