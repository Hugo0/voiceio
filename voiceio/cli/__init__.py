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
    from voiceio.cli import (
        configcmd, corrections, doctor, history, learn, models, servicecmd, uninstall, vocab,
    )

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

    doctor.register(sub)
    servicecmd.register(sub)
    models.register(sub)
    vocab.register(sub)
    corrections.register(sub)
    learn.register(sub)
    history.register(sub)
    configcmd.register(sub)

    sub.add_parser("demo", help="Guided tour of the features").set_defaults(
        func=lambda a: __import__("voiceio.demo", fromlist=["run_demo"]).run_demo())
    uninstall.register(sub)

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


def _cmd_toggle() -> None:
    """Send toggle command to running daemon."""
    from voiceio.hotkeys.socket_backend import send_toggle
    if not send_toggle():
        print("voiceio daemon is not running. Start it with: voiceio service start "
              "(or `voiceio run` in a terminal)", file=sys.stderr)
        sys.exit(1)


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
