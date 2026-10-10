"""`voiceio service` — the systemd user unit."""
from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("service", help="Autostart service (systemd user unit)")
    p.add_argument("action", nargs="?", default="status",
                   choices=["install", "uninstall", "start", "stop", "restart",
                            "status", "logs"])
    p.add_argument("-f", "--follow", action="store_true", help="logs: follow")
    p.add_argument("-n", "--lines", type=int, default=50, help="logs: line count")
    p.set_defaults(func=_cmd_service)


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
