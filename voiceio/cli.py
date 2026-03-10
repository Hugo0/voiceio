"""CLI entry point with subcommands."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path


def main() -> None:
    """Main entry point: voiceio [command] [options]."""
    from voiceio import __version__

    parser = argparse.ArgumentParser(
        prog="voiceio",
        description="Voice-to-text. Speak naturally, and text appears at your cursor.",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    # ── voiceio (no subcommand) = run daemon ──────────────────────────
    # These args apply to the default (run) mode
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--model", type=str, default=None,
                        help="Whisper model name (tiny, base, small, medium, large-v3)")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (en, es, fr, ...) or 'auto'")
    parser.add_argument("--method", type=str, default=None,
                        help="Typer backend (auto, ibus, ydotool, clipboard, ...)")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable streaming (type all text at end)")
    parser.add_argument("--notify-clipboard", action="store_true", default=None,
                        help="Show desktop notification on commit")
    parser.add_argument("--no-notify-clipboard", action="store_true", default=None,
                        help="Disable desktop notification on commit")

    # ── voiceio setup ─────────────────────────────────────────────────
    sub.add_parser("setup", help="Run interactive setup wizard")

    # ── voiceio doctor ────────────────────────────────────────────────
    p_doctor = sub.add_parser("doctor", help="Run diagnostic health check")
    p_doctor.add_argument("--fix", action="store_true",
                          help="Attempt to auto-fix issues")

    # ── voiceio toggle ────────────────────────────────────────────────
    sub.add_parser("toggle", help="Toggle recording on a running daemon")

    # ── voiceio test ──────────────────────────────────────────────────
    sub.add_parser("test", help="Run a quick microphone + transcription test")

    # ── voiceio service ────────────────────────────────────────────────
    p_service = sub.add_parser("service", help="Manage systemd autostart service")
    p_service.add_argument("action", nargs="?", default="status",
                           choices=["install", "uninstall", "start", "stop", "status"],
                           help="Action to perform (default: status)")

    # ── voiceio update ──────────────────────────────────────────────────
    sub.add_parser("update", help="Update voiceio to the latest version")

    # ── voiceio uninstall ──────────────────────────────────────────────
    sub.add_parser("uninstall", help="Remove all voiceio system integrations")

    # ── voiceio logs ───────────────────────────────────────────────────
    sub.add_parser("logs", help="Show recent log output")

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup()
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "toggle":
        _cmd_toggle()
    elif args.command == "test":
        _cmd_test()
    elif args.command == "service":
        _cmd_service(args)
    elif args.command == "update":
        _cmd_update()
    elif args.command == "uninstall":
        _cmd_uninstall()
    elif args.command == "logs":
        _cmd_logs()
    else:
        _cmd_run(args)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the voiceio daemon (default command)."""
    from voiceio import config
    cfg = config.load(path=Path(args.config) if args.config else None)
    if args.verbose:
        cfg.daemon.log_level = "DEBUG"
    if args.model:
        cfg.model.name = args.model
    if args.language:
        cfg.model.language = args.language
    if args.method:
        cfg.output.method = args.method
    if args.no_streaming:
        cfg.output.streaming = False
    if args.notify_clipboard:
        cfg.feedback.notify_clipboard = True
    elif args.no_notify_clipboard:
        cfg.feedback.notify_clipboard = False

    # Console: show voiceio messages at configured level
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    ))
    console.setLevel(logging.WARNING)

    # File: always log DEBUG to rotating file
    from voiceio.config import LOG_DIR, LOG_PATH
    from logging.handlers import RotatingFileHandler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        str(LOG_PATH), maxBytes=2_000_000, backupCount=2,
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
    ))
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])
    logging.getLogger("voiceio").setLevel(getattr(logging, cfg.daemon.log_level))

    log = logging.getLogger("voiceio")
    from voiceio import __version__, platform as plat
    p = plat.detect()
    log.info("=== voiceio v%s startup ===", __version__)
    log.info("Python %s on %s", sys.version.split()[0], sys.platform)
    log.info("Detected: os=%s display=%s desktop=%s", p.os, p.display_server, p.desktop)
    from voiceio.config import CONFIG_PATH as _default_cfg
    log.info("Config: %s", args.config or _default_cfg)
    log.info("Logs: %s", LOG_PATH)

    from voiceio.app import VoiceIO
    app = VoiceIO(cfg)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: app.request_shutdown())
    app.run()


def _cmd_setup() -> None:
    """Run interactive setup wizard."""
    from voiceio.wizard import run_wizard
    run_wizard()


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Run diagnostic health check, offer to fix issues."""
    from voiceio.health import check_health, format_report
    report = check_health()
    print(format_report(report))

    fixable = [b for b in report.hotkey_backends + report.typer_backends
               if not b.ok and b.fix_cmd]

    if not args.fix:
        if fixable:
            names = ", ".join(b.name for b in fixable)
            print(f"\nRun 'voiceio doctor --fix' to auto-fix: {names}")
        sys.exit(0 if report.all_ok else 1)

    # Auto-fix mode
    print("\nAttempting fixes...\n")
    import subprocess

    fixed_any = False
    for b in report.hotkey_backends + report.typer_backends:
        if not b.ok and b.fix_cmd:
            cmd_str = " ".join(b.fix_cmd)
            print(f"  Fixing {b.name}: {cmd_str}")
            try:
                subprocess.run(b.fix_cmd, check=True, timeout=10)
                print("  Done.")
                fixed_any = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"  Failed: {e}")

    # CLI symlinks
    if not report.cli_in_path:
        print("  Fixing CLI: creating symlinks in ~/.local/bin/")
        try:
            from voiceio.service import install_symlinks
            linked = install_symlinks()
            if linked:
                print(f"  Done, linked: {', '.join(linked)}")
                fixed_any = True
            else:
                print("  Failed: no scripts found to link")
        except Exception as e:
            print(f"  Failed: {e}")

    # IBus-specific fixes
    ibus_broken = [b for b in report.typer_backends
                   if b.name == "ibus" and not b.ok]
    for b in ibus_broken:
        if "input sources" in b.reason.lower():
            print(f"  Fixing {b.name}: adding VoiceIO to GNOME input sources")
            try:
                from voiceio.typers.ibus import _ensure_gnome_input_source
                _ensure_gnome_input_source()
                print("  Done.")
                fixed_any = True
            except Exception as e:
                print(f"  Failed: {e}")
        elif "component" in b.reason.lower():
            print(f"  Fixing {b.name}: installing IBus component")
            try:
                from voiceio.typers.ibus import install_component
                install_component()
                print("  Done.")
                fixed_any = True
            except Exception as e:
                print(f"  Failed: {e}")

    if fixed_any:
        print("\nRe-checking...")
        report = check_health()
        print(format_report(report))

    sys.exit(0 if report.all_ok else 1)


def _cmd_toggle() -> None:
    """Send toggle command to running daemon."""
    from voiceio.hotkeys.socket_backend import send_toggle
    if not send_toggle():
        print("voiceio daemon is not running. Start it with: voiceio", file=sys.stderr)
        sys.exit(1)


def _cmd_test() -> None:
    """Run a quick microphone + transcription test."""
    from voiceio.wizard import run_test
    run_test()


def _cmd_service(args: argparse.Namespace) -> None:
    """Manage the autostart service (systemd on Linux, Startup folder on Windows)."""
    from voiceio.service import (
        install_service, uninstall_service, is_installed, is_running,
        start_service, SERVICE_PATH,
    )
    import subprocess

    action = args.action

    if action == "status":
        installed = is_installed()
        running = is_running()
        print(f"Service installed: {'yes' if installed else 'no'}")
        if installed:
            print(f"Service running:   {'yes' if running else 'no'}")
            if sys.platform != "win32":
                print(f"Service file:      {SERVICE_PATH}")
        else:
            print("Run 'voiceio service install' to set up autostart.")
        sys.exit(0 if installed else 1)

    elif action == "install":
        if install_service():
            print("Service installed and enabled. It will start on next login.")
            if sys.platform != "win32":
                print("Start now: systemctl --user start voiceio")
        else:
            print("Failed to install service.", file=sys.stderr)
            sys.exit(1)

    elif action == "uninstall":
        uninstall_service()
        print("Service disabled and removed.")

    elif action == "start":
        if not is_installed():
            print("Service not installed. Run 'voiceio service install' first.", file=sys.stderr)
            sys.exit(1)
        if start_service():
            print("Service started.")
        else:
            print("Failed to start service.", file=sys.stderr)
            sys.exit(1)

    elif action == "stop":
        if sys.platform == "win32":
            print("On Windows, close the voiceio window or use Task Manager.", file=sys.stderr)
            return
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", "voiceio.service"],
                capture_output=True, timeout=5,
            )
            print("Service stopped.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("Failed to stop service.", file=sys.stderr)
            sys.exit(1)


def _cmd_update() -> None:
    """Update voiceio to the latest PyPI version."""
    import subprocess
    from voiceio import __version__
    from voiceio.config import PYPI_NAME

    is_pipx = "pipx" in sys.prefix
    if is_pipx:
        print(f"Current version: {__version__}")
        print("Checking for updates...")
        try:
            result = subprocess.run(
                ["pipx", "upgrade", PYPI_NAME],
                capture_output=True, text=True, timeout=60,
            )
            print(result.stdout.strip())
            if result.returncode != 0 and result.stderr.strip():
                print(result.stderr.strip(), file=sys.stderr)
                sys.exit(1)
        except FileNotFoundError:
            print("pipx not found. Update manually: pipx upgrade " + PYPI_NAME, file=sys.stderr)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("Update timed out.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Not a pipx install. Update manually:")
        print(f"  pip install --upgrade {PYPI_NAME}")
        sys.exit(1)


def _cmd_uninstall() -> None:
    """Remove all voiceio system integrations."""
    import os
    import shutil
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
        # Kill any running voiceio daemon (manual or systemd)
        try:
            result = subprocess.run(
                ["pgrep", "-f", "voiceio.cli"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                my_pid = str(os.getpid())
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid and pid != my_pid:
                        subprocess.run(["kill", pid], capture_output=True, timeout=3)
                removed.append("Running voiceio daemon(s)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Kill any running IBus engine process
        try:
            result = subprocess.run(
                ["pgrep", "-f", "voiceio.ibus.engine"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().split("\n"):
                    pid = pid.strip()
                    if pid:
                        subprocess.run(["kill", pid], capture_output=True, timeout=3)
                removed.append("Running IBus engine(s)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

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
        symlink_names = ["voiceio", "voiceio-toggle", "voiceio-doctor", "voiceio-setup", "voiceio-test"]
        for name in symlink_names:
            link = local_bin / name
            if link.is_symlink():
                link.unlink()
                removed.append(str(link))

    # 6. Optionally remove config and logs (use platform-aware paths)
    from voiceio.config import CONFIG_DIR, LOG_DIR
    if CONFIG_DIR.exists():
        answer = input("Remove config too? [y/N] ").strip().lower()
        if answer == "y":
            shutil.rmtree(CONFIG_DIR)
            removed.append(str(CONFIG_DIR))

    if LOG_DIR.exists():
        answer = input("Remove logs too? [y/N] ").strip().lower()
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


def _cmd_logs() -> None:
    """Show recent log output (last 50 lines)."""
    from voiceio.config import LOG_PATH
    if not LOG_PATH.exists():
        print("No log file found. Start voiceio first.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
        for line in lines[-50:]:
            print(line, end="")
    except OSError as e:
        print(f"Cannot read log file: {e}", file=sys.stderr)
        sys.exit(1)


# Legacy entry points for voiceio-doctor (parses its own --fix flag)
def _cmd_doctor_legacy() -> None:
    parser = argparse.ArgumentParser(prog="voiceio-doctor")
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    _cmd_doctor(parser.parse_args())
