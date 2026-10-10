"""CLI entry point with subcommands."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

log = logging.getLogger(__name__)

def main() -> None:
    """voiceio COMMAND [options] — see `voiceio --help`."""
    from voiceio import __version__
    from voiceio.cli import configcmd, corrections, history, learn, models, vocab

    parser = argparse.ArgumentParser(
        prog="voiceio",
        description="Local voice dictation that learns your words.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_run = sub.add_parser("run", help="Run the dictation daemon (what the service starts)")
    p_run.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    p_run.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    p_run.add_argument("--model", type=str, default=None, help="Model name or path")
    p_run.add_argument("--language", type=str, default=None, help="Language code or 'auto'")
    p_run.add_argument("--method", type=str, default=None,
                       help="Typer backend (auto, ibus, ydotool, wtype, clipboard, ...)")
    p_run.add_argument("--no-streaming", action="store_true", help="Type all text at the end")
    p_run.set_defaults(func=_cmd_run)

    sub.add_parser("toggle", help="Start/stop recording on the running daemon").set_defaults(
        func=lambda a: _cmd_toggle())

    p_setup = sub.add_parser("setup", help="Configure voiceio (interactive wizard)")
    p_setup.add_argument("--defaults", action="store_true",
                         help="Non-interactive: accept all recommended defaults")
    p_setup.add_argument("--answers", type=str, default=None, metavar="JSON",
                         help="Non-interactive: JSON object of explicit answers")
    p_setup.add_argument("--yes", "-y", action="store_true",
                         help="Run the system-dependency install command without prompting")
    p_setup.set_defaults(func=_cmd_setup)

    p_doctor = sub.add_parser("doctor", help="Check that everything works")
    p_doctor.add_argument("--fix", action="store_true", help="Try to fix what is broken")
    p_doctor.add_argument("--live", action="store_true",
                          help="Also record from the mic and transcribe it")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_service = sub.add_parser("service", help="Autostart service (systemd user unit)")
    p_service.add_argument("action", nargs="?", default="status",
                           choices=["install", "uninstall", "start", "stop", "restart",
                                    "status", "logs"])
    p_service.add_argument("-f", "--follow", action="store_true", help="logs: follow")
    p_service.add_argument("-n", "--lines", type=int, default=50, help="logs: line count")
    p_service.set_defaults(func=_cmd_service)

    models.register(sub)
    vocab.register(sub)
    corrections.register(sub)
    learn.register(sub)
    history.register(sub)
    configcmd.register(sub)

    sub.add_parser("demo", help="Guided tour of the features").set_defaults(
        func=lambda a: __import__("voiceio.demo", fromlist=["run_demo"]).run_demo())
    sub.add_parser("uninstall", help="Remove all voiceio system integrations").set_defaults(
        func=lambda a: _cmd_uninstall())

    args = parser.parse_args()
    if args.command is None:
        # A pre-1.0 systemd unit runs bare `voiceio`; keep it working until
        # `voiceio service install` (or doctor --fix) rewrites it.
        if os.environ.get("INVOCATION_ID") and not sys.stdin.isatty():
            return _cmd_run(p_run.parse_args([]))
        parser.print_help()
        return
    args.func(args)


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

    from voiceio import service
    if service.CORRECT_TIMER_PATH.exists():  # pre-1.0 cloud-mining timer
        service.uninstall_correct_timer()
    from voiceio.app import VoiceIO
    app = VoiceIO(cfg)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: app.request_shutdown())
    app.run()


def _cmd_setup(args: argparse.Namespace) -> None:
    """Run setup: interactive wizard by default, or non-interactive for agents."""
    non_interactive = getattr(args, "defaults", False) or getattr(args, "answers", None) is not None
    if non_interactive:
        import json
        answers: dict = {}
        if getattr(args, "answers", None):
            try:
                answers = json.loads(args.answers)
            except json.JSONDecodeError as e:
                print(f"[voiceio-setup] step=validate status=error reason='invalid JSON: {e}'",
                      file=sys.stderr)
                sys.exit(2)
            if not isinstance(answers, dict):
                print("[voiceio-setup] step=validate status=error reason='--answers must be a JSON object'",
                      file=sys.stderr)
                sys.exit(2)
        from voiceio.setup import run_setup_noninteractive
        sys.exit(run_setup_noninteractive(answers, install_deps=getattr(args, "yes", False)))

    from voiceio.setup import run_wizard
    run_wizard()
    from voiceio.hints import hint
    hint("test", "Run 'voiceio doctor --live' to try a quick recording")


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
    if not sys.platform.startswith("linux"):
        print("Note: voiceio targets Linux; other platforms are best-effort.\n",
              file=sys.stderr)
    from voiceio.health import check_health, format_report, format_usage, usage_findings
    from voiceio import config as _config
    report = check_health()
    print(format_report(report))
    print(format_usage(usage_findings()))
    if args.live:
        from voiceio.setup import run_test
        run_test()

    # Permission probe: nothing voiceio persists should be group/world-readable.
    perm_issues = _config.check_permissions()
    if perm_issues:
        print("\nInsecure file permissions (should be 0600 files / 0700 dirs):")
        for path, mode, expected in perm_issues:
            print(f"  {oct(mode)[2:]:>4} → {oct(expected)[2:]:>4}  {path}")

    # System-dependency preflight: the #1 install funnel killer.
    from voiceio import platform as plat
    missing_deps = plat.check_system_deps()
    dep_cmd = plat.system_deps_install_cmd(missing_deps) if missing_deps else ""
    if missing_deps:
        print("\nMissing system dependencies:")
        for key in missing_deps:
            print(f"  ✗ {plat.dep_label(key)}")
        print(f"\nInstall them all with:\n  {dep_cmd}")

    fixable = [b for b in report.hotkey_backends + report.typer_backends
               if not b.ok and b.fix_cmd]

    if not args.fix:
        if fixable or missing_deps:
            names = ", ".join(b.name for b in fixable)
            if missing_deps:
                names = (names + ", " if names else "") + "system packages"
            print(f"\nRun 'voiceio doctor --fix' to auto-fix: {names}")
        if perm_issues:
            print("Run 'voiceio doctor --fix' to tighten file permissions.")
        from voiceio.hints import hint
        hint("vocab_suggest", "Run 'voiceio vocab suggest' to find names it keeps mishearing")
        sys.exit(0 if (report.all_ok and not perm_issues and not missing_deps) else 1)

    # Auto-fix mode
    print("\nAttempting fixes...\n")
    import subprocess

    if missing_deps and dep_cmd:
        print(f"  Fixing system packages: {dep_cmd}")
        try:
            subprocess.run(dep_cmd.split(), check=True, timeout=600)
            print("  Done.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Failed: {e} — run the command above manually.")

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
        print("voiceio daemon is not running. Start it with: voiceio service start "
              "(or `voiceio run` in a terminal)", file=sys.stderr)
        sys.exit(1)


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

    elif action in ("stop", "restart"):
        try:
            subprocess.run(["systemctl", "--user", action, "voiceio.service"],
                           capture_output=True, timeout=10, check=True)
            print(f"Service {'stopped' if action == 'stop' else 'restarted'}.")
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print(f"Failed to {action} the service.", file=sys.stderr)
            sys.exit(1)

    elif action == "logs":
        from voiceio.config import LOG_PATH
        if not LOG_PATH.exists():
            print("No log file yet. Start voiceio first.", file=sys.stderr)
            sys.exit(1)
        cmd = ["tail", "-n", str(args.lines)] + (["-F"] if args.follow else []) + [str(LOG_PATH)]
        try:
            subprocess.call(cmd)
        except KeyboardInterrupt:
            pass


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


def _entry_point() -> None:
    """Console-script entry: run the CLI, and leave a crash log behind if it
    dies before logging is configured."""
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        import datetime
        import traceback
        tb = traceback.format_exc()
        try:
            from voiceio.config import LOG_DIR
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / "crash.log", "a", encoding="utf-8") as f:
                f.write(f"\n===== {datetime.datetime.now().isoformat()} =====\n"
                        f"argv: {sys.argv}\n{tb}")
            print(f"\n[voiceio crashed — traceback in {LOG_DIR / 'crash.log'}]", file=sys.stderr)
        except Exception:  # noqa: BLE001 — last resort
            pass
        print(tb, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _entry_point()
