"""CLI entry point with subcommands."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Per-run cap on candidates sent to LLM adjudication. The lower-frequency tail
# is deferred to the next run so one huge backlog can't blow up token cost.
_ADJUDICATE_CAP = 150


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

    # ── voiceio correct ─────────────────────────────────────────────────
    p_correct = sub.add_parser("correct", help="Manage corrections dictionary")
    p_correct.add_argument("wrong", nargs="?", help="Misheard word/phrase")
    p_correct.add_argument("right", nargs="?", help="Correct replacement")
    p_correct.add_argument("--list", action="store_true", help="List all corrections")
    p_correct.add_argument("--remove", metavar="WORD", help="Remove a correction")
    p_correct.add_argument("--flagged", action="store_true",
                           help="Show words flagged by 'correct that'")
    p_correct.add_argument("--clear-flagged", action="store_true",
                           help="Clear flagged words")
    p_correct.add_argument("--auto", action="store_true",
                           help="Scan history with LLM to find and fix Whisper mistakes")
    p_correct.add_argument("--full", action="store_true",
                           help="Rescan all history (ignore the last-scan cursor)")
    p_correct.add_argument("--batch", action="store_true",
                           help="Non-interactive: apply safety-gated fixes and "
                                "vocabulary, notify about items needing review "
                                "(used by the weekly systemd timer)")

    # ── voiceio history ──────────────────────────────────────────────────
    p_history = sub.add_parser("history", help="View transcription history")
    p_history.add_argument("-n", "--limit", type=int, default=20,
                           help="Number of entries to show (default: 20, 0=all)")
    p_history.add_argument("-s", "--search", type=str, default=None,
                           help="Search history by keyword")
    p_history.add_argument("--clear", action="store_true",
                           help="Clear all history")
    p_history.add_argument("--path", action="store_true",
                           help="Print history file path")

    # ── voiceio demo ──────────────────────────────────────────────────
    sub.add_parser("demo", help="Interactive guided tour of voiceio features")

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
    elif args.command == "correct":
        _cmd_correct(args)
    elif args.command == "history":
        _cmd_history(args)
    elif args.command == "demo":
        _cmd_demo()
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

    # Console: show voiceio messages at configured level (visible in journalctl)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    ))
    console.setLevel(getattr(logging, cfg.daemon.log_level))

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
    from voiceio.hints import hint
    hint("test", "Run 'voiceio test' to try a quick recording")


def _is_privileged_cmd(cmd: list[str]) -> bool:
    """True if a fix command runs sudo or pipes a remote install script."""
    joined = " ".join(cmd)
    return "sudo" in cmd or "sudo " in joined or "curl" in joined or "| sh" in joined


def _confirm_privileged(description: str, command_str: str) -> bool:
    """Show the exact privileged command and require an explicit y/N (default N).

    Used for every sudo / remote-script action in `doctor --fix` so nothing
    touches the system or runs as root without the user seeing and approving
    the exact command first.
    """
    print(f"\n  {description} needs to run:")
    print(f"    {command_str}")
    try:
        answer = input("  Run this now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Run diagnostic health check, offer to fix issues."""
    if sys.platform in ("win32", "darwin"):
        _plat = "Windows" if sys.platform == "win32" else "macOS"
        print(f"Warning: {_plat} is an experimental, untested platform. "
              f"voiceio is developed on Linux and may be broken here.\n",
              file=sys.stderr)
    from voiceio.health import check_health, format_report
    from voiceio import config as _config
    report = check_health()
    print(format_report(report))

    # Permission probe: nothing voiceio persists should be group/world-readable.
    perm_issues = _config.check_permissions()
    if perm_issues:
        print("\nInsecure file permissions (should be 0600 files / 0700 dirs):")
        for path, mode, expected in perm_issues:
            print(f"  {oct(mode)[2:]:>4} → {oct(expected)[2:]:>4}  {path}")

    fixable = [b for b in report.hotkey_backends + report.typer_backends
               if not b.ok and b.fix_cmd]

    if not args.fix:
        if fixable:
            names = ", ".join(b.name for b in fixable)
            print(f"\nRun 'voiceio doctor --fix' to auto-fix: {names}")
        if perm_issues:
            print("Run 'voiceio doctor --fix' to tighten file permissions.")
        from voiceio.hints import hint
        hint("correct_auto", "Run 'voiceio correct --auto' to scan for Whisper mistakes")
        sys.exit(0 if (report.all_ok and not perm_issues) else 1)

    # Auto-fix mode
    print("\nAttempting fixes...\n")
    import subprocess

    fixed_any = False

    if perm_issues:
        changed = _config.harden_permissions()
        print(f"  Tightened permissions on {changed} path(s).")
        fixed_any = fixed_any or changed > 0

    for b in report.hotkey_backends + report.typer_backends:
        if not b.ok and b.fix_cmd:
            cmd_str = " ".join(b.fix_cmd)
            if _is_privileged_cmd(b.fix_cmd):
                if not _confirm_privileged(f"Fix {b.name}", cmd_str):
                    print(f"  Skipped {b.name} (declined).")
                    continue
            else:
                print(f"  Fixing {b.name}: {cmd_str}")
            try:
                subprocess.run(b.fix_cmd, check=True, timeout=30)
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

    # LLM fixes
    from voiceio.config import load as _load_cfg
    _cfg = _load_cfg()
    if _cfg.llm.enabled:
        from voiceio.llm import OllamaStatus, diagnose_ollama, install_ollama, start_ollama, pull_model
        llm_status, _ = diagnose_ollama(_cfg.llm)
        if llm_status == OllamaStatus.NOT_INSTALLED:
            if sys.platform == "linux":
                if _confirm_privileged(
                    "Install Ollama (local LLM runtime)",
                    "curl -fsSL https://ollama.com/install.sh | sh",
                ):
                    print("  Installing Ollama...")
                    if install_ollama():
                        print("  Done.")
                        fixed_any = True
                    else:
                        print("  Failed. Visit https://ollama.com")
                else:
                    print("  Skipped Ollama install (declined). Manual: https://ollama.com/download")
            else:
                print("  LLM: Install Ollama from https://ollama.com")
        elif llm_status == OllamaStatus.NOT_RUNNING:
            print("  Fixing LLM: starting Ollama...")
            if start_ollama():
                print("  Done.")
                fixed_any = True
            else:
                print("  Failed. Try: ollama serve")
        elif llm_status == OllamaStatus.MODEL_NOT_FOUND:
            model = _cfg.llm.model or "phi3:mini"
            print(f"  Fixing LLM: pulling model '{model}'...")
            if pull_model(model):
                print("  Done.")
                fixed_any = True
            else:
                print(f"  Failed. Try: ollama pull {model}")

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
    from voiceio.hints import hint
    hint("service", "Run 'voiceio service install' to start on boot")


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

        # Remove the weekly mining timer + its oneshot unit.
        from voiceio import service as _service
        if _service.CORRECT_TIMER_PATH.exists() or _service.CORRECT_SERVICE_PATH.exists():
            _service.uninstall_correct_timer()
            removed.append(str(_service.CORRECT_TIMER_PATH))

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

        # 6. Remove the uinput udev rule (needs sudo — explicit consent).
        from voiceio.typers.ydotool import UINPUT_UDEV_RULE_PATH
        if Path(UINPUT_UDEV_RULE_PATH).exists():
            cmd = f"sudo rm {UINPUT_UDEV_RULE_PATH}"
            if _confirm_privileged("Remove the voiceio uinput udev rule", cmd):
                try:
                    subprocess.run(["sudo", "rm", UINPUT_UDEV_RULE_PATH], timeout=30)
                    removed.append(UINPUT_UDEV_RULE_PATH)
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


def _cmd_correct(args: argparse.Namespace) -> None:
    """Manage the corrections dictionary."""
    from voiceio.corrections import CorrectionDict
    cd = CorrectionDict()

    if args.list:
        corrections = cd.list_all()
        if not corrections:
            print("No corrections configured.")
            print("\nAdd one: voiceio correct \"wrong\" \"right\"")
            return
        for wrong, right in sorted(corrections.items()):
            print(f"  {wrong} → {right}")
        print(f"\n{len(corrections)} correction(s)")
        return

    if args.remove:
        if cd.remove(args.remove):
            print(f"Removed: {args.remove}")
        else:
            print(f"Not found: {args.remove}")
            sys.exit(1)
        return

    if args.flagged:
        flagged = cd.list_flagged()
        if not flagged:
            print("No flagged words. Say 'correct that' during dictation to flag a word.")
            return
        for w in flagged:
            print(f"  {w}")
        print(f"\n{len(flagged)} flagged word(s)")
        print("Fix with: voiceio correct \"wrong\" \"right\"")
        print("Clear:    voiceio correct --clear-flagged")
        return

    if args.clear_flagged:
        cd.clear_flagged()
        print("Flagged words cleared.")
        return

    if args.wrong and args.right:
        # Direct manual add stays ungated, but warn if it would have been
        # blocked by the safety gate (e.g. "correcting" a real word).
        from voiceio.autocorrect import gate_correction
        from voiceio.config import load as _load_cfg
        _cfg = _load_cfg()
        reason = gate_correction(
            args.wrong, args.right,
            language=_cfg.model.language if _cfg.model.language != "auto" else "en",
            protect_languages=_cfg.autocorrect.protect_languages,
        )
        if reason:
            print(f"Warning: {reason}")
        cd.add(args.wrong, args.right)
        print(f"Added: {args.wrong} → {args.right}")
        return

    # --auto or bare `voiceio correct` both run the scan flow
    if args.auto or (not args.wrong):
        _cmd_correct_auto(
            cd,
            full=getattr(args, "full", False),
            batch=getattr(args, "batch", False),
        )
        return

    print("Usage: voiceio correct \"wrong\" \"right\"")
    print("       voiceio correct --list")
    print("       voiceio correct --remove \"wrong\"")
    print("       voiceio correct --flagged")
    sys.exit(1)


def _cmd_correct_auto(cd, *, full: bool = False, batch: bool = False) -> None:
    """Scan history for Whisper mistakes, auto-fix with LLM, or review manually.

    batch=True (systemd timer): non-interactive and queue-free. Apply the
    safety-gated auto_fix bucket, then run evidence-based *adjudication* over
    the ambiguous remainder (each candidate voted on multiple times with full
    context) — applying unanimous corrections, learning unanimous terms, and
    silently deferring the rest for later re-adjudication. The scan cursor
    ALWAYS advances; deferred items carry their own revisit logic. A desktop
    notification fires only as an informational summary when something changed.
    Nothing is ever left pending for a human to triage.

    Interactive mode keeps the review UI for users who want to inspect, seeded
    with the same adjudication outcomes as pre-filled suggestions.
    """
    import time as _time

    from voiceio import history
    from voiceio.autocorrect import (
        ReviewResult, find_suspicious_words, gate_correction,
        rank_review_items, review_suspicious,
    )
    from voiceio.autocorrect_state import load_state, save_state
    from voiceio.config import load as load_cfg
    from voiceio.llm_api import resolve_api_key
    from voiceio.vocabulary import load_vocabulary
    from voiceio.wizard import (
        BOLD, CYAN, DIM, GREEN, LOGO_CORRECT, MAGENTA, RESET, YELLOW,
        Spinner, _rl_prompt,
    )

    cfg = load_cfg()
    state = load_state()
    run_ts = _time.time()

    # Snapshot corrections + vocabulary before mining mutates them, so a later
    # drift audit can roll back a bad batch. Never let this break mining.
    audit_snapshot = None
    if batch:
        try:
            from voiceio.snapshots import snapshot
            audit_snapshot = snapshot("pre-mining")
        except Exception:
            logging.getLogger("voiceio").warning("pre-mining snapshot failed", exc_info=True)

    # ── Banner + stats ──────────────────────────────────────────────────
    existing = cd.list_all()
    vocab_str = load_vocabulary(cfg.model)
    vocab_words = set(vocab_str.split(", ")) if vocab_str else set()
    flagged = cd.list_flagged()
    all_entries = history.read(limit=0)

    # Scan cursor: only mine entries newer than the last successful run,
    # unless --full was passed. Entries without a `ts` are treated as new.
    if full or not state.last_scan_ts:
        entries = all_entries
    else:
        entries = [e for e in all_entries if e.get("ts", 0) > state.last_scan_ts]

    print(LOGO_CORRECT)
    stats = []
    if full or not state.last_scan_ts:
        stats.append(f"{len(all_entries)} history entries")
    else:
        stats.append(f"{len(entries)} new / {len(all_entries)} entries")
    stats.append(f"{len(existing)} correction(s)")
    stats.append(f"{len(vocab_words)} vocabulary term(s)")
    if state.dismissed:
        stats.append(f"{len(state.dismissed)} dismissed")
    if flagged:
        stats.append(f"{YELLOW}{len(flagged)} flagged{RESET}")
    print(f"  {DIM}{' · '.join(stats)}{RESET}")

    if not all_entries:
        print(f"\n  {DIM}No history yet. Start dictating to build history.{RESET}")
        return

    if not entries:
        print(f"\n  {GREEN}✓{RESET} No new history since last scan. "
              f"{DIM}Use --full to rescan everything.{RESET}")
        save_state(state)  # refresh nothing but keep file present
        return

    # ── API key check ───────────────────────────────────────────────────
    has_api = bool(resolve_api_key(cfg.autocorrect))
    has_ollama = cfg.llm.enabled

    if batch and not has_api:
        # Unattended runs can't do manual review and Ollama isn't reliable
        # for classification — bail quietly rather than fail the timer.
        print("No API key configured — skipping batch mining run.")
        return

    if not has_api and not has_ollama:
        print(f"\n  {YELLOW}⚠{RESET}  No LLM configured — review will be manual only.")
        print(f"  {DIM}→ Set OPENROUTER_API_KEY for smart autocorrect{RESET}")
        print(f"  {DIM}→ Or re-run 'voiceio setup' to configure Ollama{RESET}")

    # ── Scan ────────────────────────────────────────────────────────────
    language = cfg.model.language if cfg.model.language != "auto" else "en"

    # Deferred words still in cooldown are skipped this run — they get a fresh
    # look (with accumulated context) only once their cooldown expires and they
    # recur in new dictation. Ready deferred words fall through normally.
    # Words retired by the teacher audit are never re-learned automatically.
    from voiceio.audit import load_audit_state
    skip_words = (set(state.dismissed) | state.cooldown_words(run_ts)
                  | load_audit_state().retired)

    with Spinner("Scanning history...") as sp:
        suspicious = find_suspicious_words(
            entries, language,
            existing_corrections=set(existing.keys()),
            vocabulary=vocab_words,
            dismissed=skip_words,
        )
        n = len(suspicious)
        sp.ok(f"Scanned {len(entries)} entries — {n} uncommon word(s)")

    if not suspicious:
        print(f"\n  {GREEN}✓{RESET} Your dictation history looks clean!")
        return

    # ── LLM review ──────────────────────────────────────────────────────
    if has_api or has_ollama:
        provider = cfg.autocorrect.model if has_api else "Ollama"
        with Spinner(f"Analyzing with {provider}...") as sp:
            def _progress(done: int, total: int) -> None:
                sp.set_message(f"Analyzing with {provider}... ({done}/{total})")
            result = review_suspicious(cfg, suspicious, on_progress=_progress)
            classified_n = (len(result.auto_fix) + len(result.ask_user)
                            + len(result.vocabulary))
            if classified_n == 0:
                sp.warn("LLM returned no classifications "
                        "(check 'voiceio logs'). Falling back to manual review.")
            else:
                sp.ok(f"Analyzed {len(suspicious)} word(s) with {provider} "
                      f"— {classified_n} classified")
    else:
        result = ReviewResult()

    # Build lookup for O(1) access to suspicious word metadata
    sw_by_word = {sw.word: sw for sw in suspicious}

    # ── Bucket 1: Auto-fix (high confidence, safety-gated) ──────────────
    # Pairs failing the gate (real word being "corrected", or target that is
    # itself junk) are downgraded to manual review rather than persisted.
    auto_fixed = 0
    downgraded: list[dict] = []
    if result.auto_fix:
        gated = []
        for fix in result.auto_fix:
            reason = gate_correction(
                fix["wrong"], fix["right"],
                vocabulary=vocab_words, language=language,
                protect_languages=cfg.autocorrect.protect_languages,
            )
            if reason:
                downgraded.append({
                    "wrong": fix["wrong"], "right": fix["right"],
                    "reason": f"needs confirmation — {reason}",
                })
            else:
                gated.append(fix)
        if gated:
            print(f"\n{BOLD}Auto-corrected{RESET} {DIM}({len(gated)}){RESET}")
            for fix in gated:
                sw = sw_by_word.get(fix["wrong"])
                count = f" {DIM}({sw.count}x){RESET}" if sw else ""
                cd.add(fix["wrong"], fix["right"])
                print(f"  {GREEN}✓{RESET} {fix['wrong']} → {BOLD}{fix['right']}{RESET}{count}")
                auto_fixed += 1

    # ── Bucket 3: Vocabulary (proper nouns/terms) — bulk confirm ─────────
    # In batch mode terms are added without prompting; add_terms() still
    # applies its sanity checks (junk, near-misspellings of existing terms).
    vocab_added = 0
    if result.vocabulary:
        vocab_added = _confirm_add_vocabulary(
            cfg, result.vocabulary, state, silent=batch,
        )

    # ── Bucket 2: Ask user (ambiguous) ──────────────────────────────────
    to_review = downgraded + list(result.ask_user)

    # If no LLM was used, put all suspicious words into manual review
    if not has_api and not has_ollama:
        for sw in suspicious:
            to_review.append({
                "wrong": sw.word,
                "right": "",
                "reason": sw.reason,
            })
    # Add words the LLM didn't mention (fell through)
    elif has_api or has_ollama:
        classified = set()
        for fix in result.auto_fix:
            classified.add(fix["wrong"].lower())
        for fix in result.ask_user:
            classified.add(fix["wrong"].lower())
        for v in result.vocabulary:
            classified.add(v.lower())
        for sw in suspicious:
            if sw.word.lower() not in classified:
                to_review.append({
                    "wrong": sw.word,
                    "right": "",
                    "reason": "not classified by LLM",
                })

    # Rank: most-likely-error words first (LLM-suggested fixes, words near
    # common dictionary words, high-frequency entries) — push acronyms / tech
    # terms toward the end where the user can quickly skip or quit.
    to_review = rank_review_items(to_review, sw_by_word)

    reviewed = 0
    skipped = 0
    user_quit = False

    # ── Evidence-based adjudication (replaces the human review queue) ─────
    # Cap per run by frequency so a huge backlog can't blow up token cost in a
    # single run; the lower-frequency tail is capacity-deferred (no penalty,
    # no cooldown) so it's picked up on the very next run.
    from voiceio.autocorrect import adjudicate

    def _cand_count(item: dict) -> int:
        sw = sw_by_word.get(item.get("wrong", ""))
        return sw.count if sw else 0

    to_review.sort(key=lambda it: -_cand_count(it))
    if len(to_review) > _ADJUDICATE_CAP:
        capped = to_review[_ADJUDICATE_CAP:]
        to_review = to_review[:_ADJUDICATE_CAP]
        for it in capped:
            state.defer(it["wrong"], failure=False)
        log.info(
            "Adjudication cap hit: adjudicating %d, deferring %d to next run",
            len(to_review), len(capped),
        )
        print(f"  {DIM}{len(capped)} lower-frequency item(s) deferred "
              f"to next run{RESET}")

    adj = adjudicate(cfg, to_review, sw_by_word,
                     vocabulary=vocab_words, language=language)

    # Unanimous corrections (already gate-passed inside adjudicate).
    adj_applied = 0
    if adj.apply:
        print(f"\n{BOLD}Adjudicated{RESET} {DIM}({len(adj.apply)}){RESET}")
        for fix in adj.apply:
            sw = sw_by_word.get(fix["wrong"])
            count = f" {DIM}({sw.count}x){RESET}" if sw else ""
            cd.add(fix["wrong"], fix["right"])
            print(f"  {GREEN}✓{RESET} {fix['wrong']} → "
                  f"{BOLD}{fix['right']}{RESET}{count}")
            adj_applied += 1

    # Unanimous real terms → vocabulary.
    adj_vocab = 0
    if adj.vocabulary:
        adj_vocab = _confirm_add_vocabulary(cfg, adj.vocabulary, state, silent=True)

    corrections_learned = auto_fixed + adj_applied
    terms_learned = vocab_added + adj_vocab

    # ── Batch: no queue, ever. Defer the rest, advance the cursor, notify ─
    if batch:
        for it in adj.deferred:
            state.defer(it["wrong"], votes=it.get("votes"))
        if corrections_learned or terms_learned:
            from voiceio.feedback import notify
            parts = []
            if corrections_learned:
                parts.append(f"{corrections_learned} correction"
                             f"{'s' if corrections_learned != 1 else ''}")
            if terms_learned:
                parts.append(f"{terms_learned} term"
                             f"{'s' if terms_learned != 1 else ''}")
            notify(
                "VoiceIO learned from your dictation",
                f"Learned {' and '.join(parts)} this week.",
            )
        state.last_scan_ts = run_ts
        save_state(state)
        return

    # ── Interactive: inspect the deferred remainder (pre-filled) ─────────
    to_review = rank_review_items(adj.deferred, sw_by_word)

    if to_review:
        import re as _re
        import readline as _rl

        def _context_snippet(text: str, word: str, width: int = 80) -> str:
            """Extract a snippet centered on the word occurrence."""
            idx = text.lower().find(word.lower())
            if idx == -1:
                return text[:width]
            # Center the word in the window
            half = (width - len(word)) // 2
            start = max(0, idx - half)
            end = min(len(text), start + width)
            start = max(0, end - width)  # re-adjust if near the end
            snippet = text[start:end]
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return f"{prefix}{snippet}{suffix}"

        def _fmt_context(text: str, word: str) -> str:
            """Format a context snippet with the word highlighted."""
            snippet = _context_snippet(text, word)
            # Highlight the word (case-insensitive)
            import re as _re2
            highlighted = _re2.sub(
                f"({_re2.escape(word)})",
                rf"{BOLD}{YELLOW}\1{RESET}",
                snippet, count=1, flags=_re2.IGNORECASE,
            )
            return highlighted

        def _input_prefill(prompt: str, prefill: str = "") -> str:
            """input() with pre-filled editable text."""
            if prefill:
                _rl.set_startup_hook(lambda: _rl.insert_text(prefill))
            try:
                return input(_rl_prompt(prompt))
            finally:
                _rl.set_startup_hook()

        def _key(letter: str, rest: str) -> str:
            return f"[{BOLD}{CYAN}{letter}{RESET}]{rest}"

        print(f"\n{BOLD}Review{RESET} {DIM}({len(to_review)}){RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")

        # Build actions legend (shown below each prompt)
        actions = [_key("a", "ccept"), _key("v", "ocab"), _key("s", "kip"),
                   _key("c", "ontext"), _key("q", "uit")]
        if not has_api:
            actions.append(_key("k", "ey"))
        actions.append(f"{DIM}or type correction{RESET}")
        legend = f"  {' '.join(actions)}"

        # ANSI: save pos, move down, print, restore pos
        CUU = "\033[A"     # cursor up one line
        EL = "\033[2K"     # erase entire line

        for i, item in enumerate(to_review, 1):
            wrong = item["wrong"]
            right = item.get("right", "")
            reason = item.get("reason", "")

            sw_match = sw_by_word.get(wrong)
            # Deduplicate contexts
            contexts: list[str] = []
            if sw_match and sw_match.contexts:
                seen: set[str] = set()
                for c in sw_match.contexts:
                    if c not in seen:
                        seen.add(c)
                        contexts.append(c)
            ctx_idx = 0

            # Word header
            count_str = f" {DIM}({sw_match.count}x){RESET}" if sw_match else ""
            print(f"\n  {MAGENTA}[{i}/{len(to_review)}]{RESET} {BOLD}\"{wrong}\"{RESET}{count_str}")

            # Reason
            if reason:
                print(f"  {DIM}{reason}{RESET}")

            # Suggestion from LLM
            if right:
                print(f"  {DIM}suggestion:{RESET} {GREEN}{right}{RESET}")

            # Context (first one)
            if contexts:
                ctx_label = f" [1/{len(contexts)}]" if len(contexts) > 1 else ""
                print(f"  {DIM}context{ctx_label}:{RESET} \"{_fmt_context(contexts[0], wrong)}\"")

            # Single input line — pre-fill with suggestion if available
            prompt_hint = f"  {CYAN}›{RESET} "
            quit_requested = False

            def _clear_legend() -> None:
                """Erase the legend lines below the prompt."""
                sys.stdout.write(f"\n{EL}\n{EL}{CUU}{CUU}\r")
                sys.stdout.flush()

            while True:
                # Show legend below with spacing, then move cursor back up
                sys.stdout.write(f"\n\n{legend}{CUU}{CUU}\r")
                sys.stdout.flush()
                try:
                    raw = _input_prefill(prompt_hint, right).strip()
                except (EOFError, KeyboardInterrupt):
                    _clear_legend()
                    print()
                    quit_requested = True
                    break
                _clear_legend()

                # Strip ANSI escape sequences (arrow keys, etc.)
                choice = _re.sub(r"\x1b\[[A-D]", "", raw).strip()

                # Arrow keys → treat as context cycling
                if "\x1b[A" in raw or "\x1b[B" in raw or (not choice and raw):
                    choice = "c"

                cl = choice.lower()

                if cl in ("c", "context"):
                    if contexts and ctx_idx < len(contexts) - 1:
                        ctx_idx += 1
                        print(f"  {DIM}context [{ctx_idx + 1}/{len(contexts)}]:"
                              f"{RESET} \"{_fmt_context(contexts[ctx_idx], wrong)}\"")
                    elif not contexts:
                        print(f"  {DIM}no contexts available{RESET}")
                    else:
                        print(f"  {DIM}no more contexts{RESET}")
                    continue

                if cl in ("k", "key"):
                    try:
                        key = input(_rl_prompt(f"  {CYAN}›{RESET} Paste OPENROUTER_API_KEY: ")).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        quit_requested = True
                        break
                    if not key:
                        continue
                    _save_api_key(cfg, key)
                    has_api = True
                    cfg = load_cfg()  # reload so resolve_api_key sees the new key
                    print(f"  {GREEN}✓{RESET} API key saved to config")

                    # Drop [k]ey from the legend now that we have one.
                    legend_actions = [_key("a", "ccept"), _key("v", "ocab"),
                                      _key("s", "kip"), _key("c", "ontext"),
                                      _key("q", "uit"),
                                      f"{DIM}or type correction{RESET}"]
                    legend = f"  {' '.join(legend_actions)}"

                    # Re-run LLM analysis on words not yet reviewed (skip current).
                    remaining_words = {to_review[j]["wrong"] for j in range(i, len(to_review))}
                    remaining_sw = [sw for sw in suspicious if sw.word in remaining_words]
                    if remaining_sw:
                        try:
                            with Spinner(f"Analyzing with {cfg.autocorrect.model}...") as sp:
                                def _resume_progress(done: int, total: int) -> None:
                                    sp.set_message(
                                        f"Analyzing with {cfg.autocorrect.model}... ({done}/{total})",
                                    )
                                new_result = review_suspicious(
                                    cfg, remaining_sw, on_progress=_resume_progress,
                                )
                                sp.ok(f"Analyzed {len(remaining_sw)} word(s) with {cfg.autocorrect.model}")
                        except Exception as exc:
                            print(f"  {YELLOW}⚠{RESET}  LLM analysis failed: {exc}")
                            continue

                        resume_downgraded: list[dict] = []
                        if new_result.auto_fix:
                            gated = []
                            for fix in new_result.auto_fix:
                                reason = gate_correction(
                                    fix["wrong"], fix["right"],
                                    vocabulary=vocab_words, language=language,
                                    protect_languages=cfg.autocorrect.protect_languages,
                                )
                                if reason:
                                    resume_downgraded.append({
                                        "wrong": fix["wrong"], "right": fix["right"],
                                        "reason": f"needs confirmation — {reason}",
                                    })
                                else:
                                    gated.append(fix)
                            if gated:
                                print(f"\n{BOLD}Auto-corrected{RESET} {DIM}({len(gated)}){RESET}")
                                for fix in gated:
                                    sw = sw_by_word.get(fix["wrong"])
                                    count = f" {DIM}({sw.count}x){RESET}" if sw else ""
                                    cd.add(fix["wrong"], fix["right"])
                                    auto_fixed += 1
                                    print(f"  {GREEN}✓{RESET} {fix['wrong']} → {BOLD}{fix['right']}{RESET}{count}")

                        if new_result.vocabulary:
                            vocab_added += _confirm_add_vocabulary(
                                cfg, new_result.vocabulary, state,
                            )

                        classified = ({f["wrong"].lower() for f in new_result.auto_fix}
                                      | {v.lower() for v in new_result.vocabulary})
                        new_remaining = resume_downgraded + list(new_result.ask_user)
                        seen_lc = {a["wrong"].lower() for a in new_remaining}
                        for sw in remaining_sw:
                            wlow = sw.word.lower()
                            if wlow not in classified and wlow not in seen_lc:
                                new_remaining.append({
                                    "wrong": sw.word, "right": "",
                                    "reason": "not classified by LLM",
                                })
                                seen_lc.add(wlow)
                        new_remaining = rank_review_items(new_remaining, sw_by_word)
                        to_review[i:] = new_remaining
                        if new_remaining:
                            print(f"\n{DIM}Continuing review with smart suggestions ({len(new_remaining)} remaining)...{RESET}")
                    continue

                if cl in ("q", "quit"):
                    quit_requested = True
                    break
                elif cl in ("a", "accept", "y", "yes") and right:
                    reason = gate_correction(
                        wrong, right, vocabulary=vocab_words, language=language,
                        protect_languages=cfg.autocorrect.protect_languages,
                    )
                    if reason:
                        print(f"  {YELLOW}⚠{RESET}  {reason}")
                    cd.add(wrong, right)
                    print(f"  {GREEN}✓{RESET} {wrong} → {BOLD}{right}{RESET}")
                    reviewed += 1
                    reviewed += _offer_cluster_apply(
                        cd, wrong, right, to_review, i, sw_by_word, _rl_prompt,
                    )
                elif cl in ("v", "vocab"):
                    added = _confirm_add_vocabulary(
                        cfg, [wrong], state, silent=True,
                    )
                    vocab_added += added
                    if added:
                        print(f"  {CYAN}+{RESET} Added \"{wrong}\" to vocabulary")
                    else:
                        print(f"  {DIM}Skipped \"{wrong}\" (not a valid vocabulary term){RESET}")
                elif cl in ("s", "skip", ""):
                    # Rejecting a word means never propose it again.
                    state.dismiss(wrong)
                    skipped += 1
                else:
                    # Typed a word — use it as the correction directly.
                    reason = gate_correction(
                        wrong, choice, vocabulary=vocab_words, language=language,
                        protect_languages=cfg.autocorrect.protect_languages,
                    )
                    if reason:
                        print(f"  {YELLOW}⚠{RESET}  {reason}")
                    cd.add(wrong, choice)
                    print(f"  {GREEN}✓{RESET} {wrong} → {BOLD}{choice}{RESET}")
                    reviewed += 1
                    reviewed += _offer_cluster_apply(
                        cd, wrong, choice, to_review, i, sw_by_word, _rl_prompt,
                    )
                break

            if quit_requested:
                user_quit = True
                break

    # ── Persist scan state ──────────────────────────────────────────────
    # Advance the cursor only on a completed run — quitting mid-review must
    # leave the un-reviewed (new) entries eligible for the next scan. (Batch
    # mode already advanced the cursor and returned above.) The dismissed set
    # is always persisted.
    if not user_quit:
        state.last_scan_ts = run_ts
    save_state(state)

    # ── Summary ─────────────────────────────────────────────────────────
    auto_total = auto_fixed + adj_applied
    vocab_total = vocab_added + adj_vocab
    parts = []
    if auto_total:
        parts.append(f"{GREEN}{auto_total} auto-corrected{RESET}")
    if reviewed:
        parts.append(f"{GREEN}{reviewed} reviewed{RESET}")
    if vocab_total:
        parts.append(f"{CYAN}{vocab_total} vocabulary{RESET}")
    if skipped:
        parts.append(f"{DIM}{skipped} skipped{RESET}")
    if parts:
        print(f"\n{BOLD}Summary{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        print(f"  {' · '.join(parts)}")
    if auto_total or reviewed:
        print(f"  {DIM}Corrections will apply to future dictations automatically.{RESET}")
    if auto_total or reviewed or vocab_total:
        from voiceio.hints import hint
        hint("correct_list", "Run 'voiceio correct --list' to see all corrections")

    # ── Teacher-model audit (weekly batch only) ─────────────────────────
    # Runs after mining so freshly-applied rules are measured against ground
    # truth. Guarded so an audit failure can never break the mining timer, and
    # skipped when there's no retained audio to transcribe.
    if batch:
        try:
            from voiceio.config import RECORDINGS_DIR
            if any(RECORDINGS_DIR.glob("*.wav")):
                from voiceio.audit import run_audit
                run_audit(cfg, snapshot_dir=audit_snapshot)
        except Exception:
            logging.getLogger("voiceio").warning("teacher audit failed", exc_info=True)


def _offer_cluster_apply(
    cd, wrong: str, right: str, to_review: list, current_i: int,
    sw_by_word: dict, rl_prompt,
) -> int:
    """After a correction, offer to apply it to Levenshtein-close remaining items.

    Returns the number of additional corrections applied. Mutates `to_review`
    in place by removing items the user batch-accepted.
    """
    from voiceio.autocorrect import _levenshtein
    from voiceio.wizard import BOLD, CYAN, DIM, GREEN, RESET

    threshold = 2
    matches: list[tuple[int, str]] = []
    wrong_lc = wrong.lower()
    for j in range(current_i, len(to_review)):
        other = to_review[j].get("wrong", "")
        if not other or other.lower() == wrong_lc:
            continue
        if _levenshtein(wrong_lc, other.lower()) <= threshold:
            matches.append((j, other))

    if not matches:
        return 0

    preview_n = 5
    shown = ", ".join(m[1] for m in matches[:preview_n])
    if len(matches) > preview_n:
        shown += f", +{len(matches) - preview_n} more"
    print(f"  {DIM}→ Found {len(matches)} similar variant(s): {shown}{RESET}")
    try:
        yn = input(rl_prompt(
            f"  {CYAN}›{RESET} Apply '{BOLD}{right}{RESET}' to all? [Y/n] ",
        )).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if yn not in ("", "y", "yes"):
        return 0

    # Remove back-to-front so earlier indices stay valid
    applied = 0
    for j, other in reversed(matches):
        cd.add(other, right)
        sw = sw_by_word.get(other)
        count = f" {DIM}({sw.count}x){RESET}" if sw else ""
        print(f"  {GREEN}✓{RESET} {other} → {BOLD}{right}{RESET}{count}")
        del to_review[j]
        applied += 1
    return applied


def _confirm_add_vocabulary(cfg, terms: list[str], state, *, silent: bool = False) -> int:
    """Offer to bulk-add vocabulary `terms`, returning the count added.

    In the `--auto` flow the LLM's vocabulary bucket is presented as a single
    one-key confirm ("add N terms? [Y/n]") with the list shown. Declined terms
    are dismissed so they aren't re-proposed. `silent=True` skips the prompt
    (used for the single-word [v]ocab action inside manual review).

    Persistence and sanity-checking (dedupe, junk/misspelling skip) live in
    `vocabulary.add_terms`.
    """
    from voiceio.vocabulary import add_terms

    terms = [t for t in terms if t and t.strip()]
    if not terms:
        return 0

    if not silent:
        from voiceio.wizard import BOLD, CYAN, DIM, RESET, _rl_prompt
        shown = ", ".join(terms[:12])
        if len(terms) > 12:
            shown += f", +{len(terms) - 12} more"
        print(f"\n{BOLD}Vocabulary{RESET} {DIM}({len(terms)}){RESET}")
        print(f"  {DIM}{shown}{RESET}")
        try:
            yn = input(_rl_prompt(
                f"  {CYAN}›{RESET} Add {len(terms)} term(s) to vocabulary? [Y/n] ",
            )).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            yn = "n"
        if yn not in ("", "y", "yes"):
            for t in terms:
                state.dismiss(t)
            return 0

    return add_terms(terms, cfg.model)


def _save_api_key(cfg, key: str) -> None:
    """Save an API key + auto-detected provider to the config file."""
    from voiceio.config import CONFIG_PATH
    from voiceio.llm_api import detect_provider

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_url, model = detect_provider(key)

    if CONFIG_PATH.exists():
        content = CONFIG_PATH.read_text(encoding="utf-8")
    else:
        content = ""

    import re

    def _set_field(content: str, field: str, value: str) -> str:
        """Set a field under [autocorrect], adding it if missing."""
        if re.search(r"^\[autocorrect\]", content, re.MULTILINE):
            # Try to replace existing line (commented or not)
            new, n = re.subn(
                rf"(^\[autocorrect\][^\[]*?)^#?\s*{field}\s*=.*$",
                rf'\1{field} = "{value}"',
                content, count=1, flags=re.MULTILINE,
            )
            if n:
                return new
            # Field didn't exist — append after [autocorrect]
            return re.sub(
                r"^(\[autocorrect\].*)$",
                rf'\1\n{field} = "{value}"',
                content, count=1, flags=re.MULTILINE,
            )
        else:
            return content + f'\n[autocorrect]\n{field} = "{value}"\n'

    content = _set_field(content, "api_key", key)
    content = _set_field(content, "base_url", base_url)
    content = _set_field(content, "model", model)

    from voiceio import config as _config
    from voiceio import consent
    # config.toml now holds a secret — write it 0600 in a 0700 dir.
    _config.secure_write(CONFIG_PATH, content)
    # Deliberately configuring a cloud key is explicit cloud consent.
    consent.record_consent(source="api-key")


def _cmd_history(args: argparse.Namespace) -> None:
    """View or manage transcription history."""
    from voiceio import history
    from voiceio.config import HISTORY_PATH

    if args.path:
        print(HISTORY_PATH)
        return

    if args.clear:
        history.clear()
        print("History cleared.")
        return

    if args.search:
        entries = history.search(args.search)
    else:
        entries = history.read(limit=args.limit)

    if not entries:
        if args.search:
            print(f"No matches for: {args.search}")
        else:
            print("No history yet. Start dictating to build history.")
        return

    import time as _time
    for e in reversed(entries):  # show oldest first (chronological)
        ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(e.get("ts", 0)))
        print(f"  [{ts}] {e.get('text', '')}")

    print(f"\n{len(entries)} entry/entries shown")
    if not args.search and args.limit and len(entries) == args.limit:
        print(f"Show more: voiceio history -n {args.limit * 2}")
    if len(entries) >= 20:
        from voiceio.hints import hint
        hint("correct_auto", "Run 'voiceio correct --auto' to scan for Whisper mistakes")


def _cmd_demo() -> None:
    """Run the interactive guided tour."""
    from voiceio.demo import run_demo
    run_demo()


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


def _entry_point() -> None:
    """PyInstaller entry point.

    On Linux/macOS the console_scripts wrapper that setuptools generates
    from ``project.scripts`` calls ``main()`` directly, so module-level
    code here never runs. On Windows we ship a PyInstaller bundle that
    runs ``cli.py`` as ``__main__`` — without this wrapper, the exe would
    load all definitions and exit silently (which is exactly what users
    reported: "installed, clicked, nothing happened").

    This wrapper also catches any unhandled exception and writes it to
    ``crash.log`` before exiting, so a crash before logging is configured
    still leaves a diagnostic trail. On Windows consoles we pause so the
    cmd.exe window stays open long enough to read the error.
    """
    # Force UTF-8 on Windows stdout/stderr so we can print Unicode
    # symbols (✓ / ✗ / emoji in doctor output, foreign characters in
    # transcriptions) without hitting UnicodeEncodeError on the default
    # cp1252 code page. Python 3.7+ has PYTHONUTF8 / reconfigure.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        # Best-effort crash log: write to the standard log dir, and also
        # stderr. Use a fresh import path in case config import itself
        # was the thing that crashed.
        try:
            import os
            from pathlib import Path
            if sys.platform == "win32":
                log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "voiceio" / "logs"
            else:
                log_dir = Path.home() / ".local" / "state" / "voiceio"
            log_dir.mkdir(parents=True, exist_ok=True)
            crash_path = log_dir / "crash.log"
            with open(crash_path, "a", encoding="utf-8") as f:
                import datetime
                f.write(f"\n===== {datetime.datetime.now().isoformat()} =====\n")
                f.write(f"argv: {sys.argv}\n")
                f.write(f"platform: {sys.platform}\n")
                f.write(tb)
            print(f"\n[voiceio crashed — wrote traceback to {crash_path}]",
                  file=sys.stderr)
        except Exception:
            pass  # absolute last-resort: nothing we can do
        print(tb, file=sys.stderr)
        # On Windows, keep the console window open so the user can read
        # the error instead of watching cmd.exe flash and close.
        if sys.platform == "win32" and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("\nPress Enter to close...")
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exit(1)


if __name__ == "__main__":
    _entry_point()
