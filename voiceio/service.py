"""Systemd user service installation and management."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

LOCAL_BIN = Path.home() / ".local" / "bin"
SCRIPT_NAMES = ["voiceio", "voiceio-toggle", "voiceio-doctor", "voiceio-setup", "voiceio-test"]
_PATH_HINT_ADDED = False  # track if we already printed the PATH hint

SERVICE_NAME = "voiceio.service"
SERVICE_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_PATH = SERVICE_DIR / SERVICE_NAME


def has_systemd() -> bool:
    """Check if systemd is available on this system."""
    return shutil.which("systemctl") is not None


def _find_voiceio_bin() -> str:
    """Find the voiceio binary path."""
    # Check venv first
    venv_bin = Path(sys.prefix) / "bin" / "voiceio"
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
Description=VoiceIO — voice-to-text input
Documentation=https://github.com/hugomontenegro/voiceio
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart={bin_path}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def install_service() -> bool:
    """Install and enable the voiceio systemd user service.

    Returns True if installed successfully.
    """
    if not has_systemd():
        log.warning("systemctl not found — cannot install service")
        return False

    bin_path = _find_voiceio_bin()

    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    SERVICE_PATH.write_text(_service_unit(bin_path))
    log.info("Installed systemd service to %s", SERVICE_PATH)

    # Reload systemd and enable the service
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", SERVICE_NAME],
            capture_output=True, timeout=5,
        )
        log.info("Enabled %s", SERVICE_NAME)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Could not enable service: %s", e)
        return False


def uninstall_service() -> bool:
    """Disable and remove the voiceio systemd user service."""
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
    """Check if the systemd service is installed."""
    return SERVICE_PATH.exists()


def is_running() -> bool:
    """Check if the systemd service is currently running."""
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
    so we skip symlink creation.
    """
    if _is_pipx_install():
        # pipx already placed scripts in ~/.local/bin/ — nothing to do
        return [name for name in SCRIPT_NAMES if (LOCAL_BIN / name).exists()]

    venv_bin = Path(sys.prefix) / "bin"
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
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
                log.warning("Skipping %s — regular file exists at %s", name, dest)
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
    """Add ~/.local/bin to PATH via shell profile (for macOS etc.)."""
    global _PATH_HINT_ADDED
    line = '\nexport PATH="$HOME/.local/bin:$PATH"\n'
    # Try .zshrc first (macOS default), then .bashrc
    for rc_name in (".zshrc", ".bashrc", ".profile"):
        rc = Path.home() / rc_name
        if rc.exists():
            content = rc.read_text()
            if ".local/bin" in content:
                return  # already there
            rc.write_text(content + line)
            log.info("Added ~/.local/bin to PATH in %s", rc)
            _PATH_HINT_ADDED = True
            return
    # No shell rc found — create .profile
    rc = Path.home() / ".profile"
    rc.write_text(line)
    log.info("Created %s with PATH entry", rc)
    _PATH_HINT_ADDED = True


def symlinks_installed() -> bool:
    """Check if voiceio is accessible as a command."""
    # Check if it's anywhere on PATH
    if shutil.which("voiceio"):
        return True
    # Check if symlink exists (even if not on PATH yet — new shell will pick it up)
    dest = LOCAL_BIN / "voiceio"
    return dest.exists()


def path_hint_needed() -> bool:
    """Return True if we modified shell profile and user needs to restart shell."""
    return _PATH_HINT_ADDED
