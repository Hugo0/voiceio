"""`voiceio uninstall` — remove every system integration voiceio made."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voiceio.cli.doctor import _confirm_privileged


def register(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("uninstall", help="Remove all voiceio system integrations").set_defaults(
        func=lambda a: _cmd_uninstall())


def _cmd_uninstall() -> None:
    """Remove all voiceio system integrations."""
    import os
    import shutil
    import signal
    import subprocess

    home = Path.home()
    removed: list[str] = []

    answer = input("This will remove all voiceio system files. Continue? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # 1. Stop running daemons
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "voiceio.exe"],
                capture_output=True, timeout=5,
            )
            removed.append("Running voiceio process(es)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Remove Windows startup shortcut
        from voiceio.service import uninstall_windows_startup
        if uninstall_windows_startup():
            removed.append("Windows startup shortcut")
    else:
        # Kill the running voiceio daemon via its PID file (not pgrep, which
        # can match unrelated processes or this very uninstall command).
        from voiceio.config import PID_PATH
        try:
            pid = int(PID_PATH.read_text().strip())
            if pid and pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGTERM)
                    removed.append(f"Running voiceio daemon (pid {pid})")
                except ProcessLookupError:
                    pass  # already gone
                except PermissionError:
                    pass
            PID_PATH.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass

        # Kill any running IBus engine process
        from voiceio.ibus import engine_pids
        pids = engine_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        if pids:
            removed.append("Running IBus engine(s)")

        # Stop and disable systemd service
        service_path = home / ".config" / "systemd" / "user" / "voiceio.service"
        if service_path.exists():
            try:
                subprocess.run(
                    ["systemctl", "--user", "stop", "voiceio.service"],
                    capture_output=True, timeout=5,
                )
                subprocess.run(
                    ["systemctl", "--user", "disable", "voiceio.service"],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            service_path.unlink(missing_ok=True)
            try:
                subprocess.run(
                    ["systemctl", "--user", "daemon-reload"],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            removed.append(str(service_path))

        # Remove the weekly mining timer + its oneshot unit.
        from voiceio import service as _service
        if _service.CORRECT_TIMER_PATH.exists() or _service.CORRECT_SERVICE_PATH.exists():
            _service.uninstall_correct_timer()
            removed.append(str(_service.CORRECT_TIMER_PATH))
        if _service.LEARN_TIMER_PATH.exists():
            _service.uninstall_learn_timer()
            removed.append(str(_service.LEARN_TIMER_PATH))

        # 2. Remove IBus component and launcher
        ibus_component = home / ".local" / "share" / "ibus" / "component" / "voiceio.xml"
        ibus_launcher = home / ".local" / "share" / "voiceio" / "voiceio-ibus-engine"
        if ibus_component.exists():
            ibus_component.unlink()
            removed.append(str(ibus_component))
        if ibus_launcher.exists():
            ibus_launcher.unlink()
            removed.append(str(ibus_launcher))
            launcher_dir = ibus_launcher.parent
            try:
                launcher_dir.rmdir()
                removed.append(str(launcher_dir))
            except OSError:
                pass

        # 3. Remove GNOME input source entry
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and "'ibus', 'voiceio'" in result.stdout:
                import ast
                sources = ast.literal_eval(result.stdout.strip())
                new_sources = [s for s in sources if s != ("ibus", "voiceio")]
                formatted = "[" + ", ".join(f"({s[0]!r}, {s[1]!r})" for s in new_sources) + "]"
                subprocess.run(
                    ["gsettings", "set", "org.gnome.desktop.input-sources", "sources", formatted],
                    capture_output=True, timeout=3,
                )
                removed.append("GNOME input source ('ibus', 'voiceio')")
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, SyntaxError):
            pass

        # 4. Remove environment.d file
        env_file = home / ".config" / "environment.d" / "voiceio.conf"
        if env_file.exists():
            env_file.unlink()
            removed.append(str(env_file))

        # 5. Remove CLI symlinks from ~/.local/bin/
        local_bin = home / ".local" / "bin"
        symlink_names = ["voiceio", "voiceio-toggle", "voiceio-doctor", "voiceio-setup",
                         "voiceio-test"]  # includes pre-1.0 aliases, for cleanup
        for name in symlink_names:
            link = local_bin / name
            if link.is_symlink():
                link.unlink()
                removed.append(str(link))

        # 6. Remove the uinput + keyboard udev rules (need sudo — explicit consent).
        from voiceio.hotkeys.evdev import UACCESS_RULE_PATH
        from voiceio.typers.ydotool import UINPUT_UDEV_RULE_PATH
        for rule_path, what in ((UINPUT_UDEV_RULE_PATH, "uinput"),
                                (UACCESS_RULE_PATH, "keyboard-access")):
            if not Path(rule_path).exists():
                continue
            cmd = f"sudo rm {rule_path}"
            if _confirm_privileged(f"Remove the voiceio {what} udev rule", cmd):
                try:
                    subprocess.run(["sudo", "rm", rule_path], timeout=30)
                    removed.append(rule_path)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    print(f"  Failed. Run manually: {cmd}")

    # 7. Optionally remove config and all local state (platform-aware paths)
    from voiceio.config import CONFIG_DIR, LOG_DIR
    if CONFIG_DIR.exists():
        answer = input("Remove config too (config.toml, corrections, consent)? [y/N] ").strip().lower()
        if answer == "y":
            shutil.rmtree(CONFIG_DIR)
            removed.append(str(CONFIG_DIR))

    if LOG_DIR.exists():
        answer = input("Remove all local state too (history, recordings, logs, metrics)? [y/N] ").strip().lower()
        if answer == "y":
            shutil.rmtree(LOG_DIR)
            removed.append(str(LOG_DIR))

    # Print summary
    if removed:
        print("\nRemoved:")
        for item in removed:
            print(f"  - {item}")
    else:
        print("\nNothing to remove. voiceio was not installed on this system.")

    # Offer to uninstall the Python package itself
    from voiceio.config import PYPI_NAME
    is_pipx = "pipx" in sys.prefix
    if is_pipx:
        answer = input("\nAlso uninstall the voiceio Python package (pipx uninstall)? [Y/n] ").strip().lower()
        if answer in ("y", "yes", ""):
            try:
                subprocess.run(["pipx", "uninstall", PYPI_NAME], timeout=30)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(f"Failed. Run manually: pipx uninstall {PYPI_NAME}")
    else:
        # Dev install or pip install: check if voiceio is still reachable
        voiceio_bin = shutil.which("voiceio")
        if voiceio_bin:
            print(f"\nNote: 'voiceio' is still available at {voiceio_bin}")
            if ".venv" in str(voiceio_bin) or "site-packages" in str(voiceio_bin):
                print("This is a development install. To fully remove:")
                print(f"  pip uninstall {PYPI_NAME}")
            else:
                print("To fully remove the package:")
                print(f"  pip uninstall {PYPI_NAME}")
        else:
            print("\nvoiceio fully removed.")
