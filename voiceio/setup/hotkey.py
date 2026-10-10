"""Hotkey wiring: GNOME custom shortcut, tiling-WM bind lines, live hotkey test."""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

from voiceio.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW


def _wm_bind_lines(hotkey: str) -> list[tuple[str, str]]:
    """Config lines binding ``hotkey`` to `voiceio toggle` per tiling WM."""
    *mods, key = hotkey.lower().split("+")
    hypr = {"super": "SUPER", "meta": "SUPER", "win": "SUPER", "ctrl": "CTRL",
            "alt": "ALT", "shift": "SHIFT"}
    sway = {"super": "Mod4", "meta": "Mod4", "win": "Mod4", "ctrl": "Ctrl",
            "alt": "Alt", "shift": "Shift"}
    hypr_mods = " ".join(hypr.get(m, m.upper()) for m in mods)
    sway_combo = "+".join([*(sway.get(m, m) for m in mods), key])
    return [
        ("hyprland", f"bind = {hypr_mods}, {key.upper()}, exec, voiceio toggle"),
        ("i3/sway", f"bindsym {sway_combo} exec voiceio toggle"),
    ]


def _toggle_command() -> str | None:
    """Absolute `voiceio toggle` command: GNOME doesn't know about venvs."""
    venv_path = Path(sys.prefix) / "bin" / "voiceio"
    if venv_path.exists():
        return f"{venv_path.resolve()} toggle"
    found = shutil.which("voiceio")
    return f"{Path(found).resolve()} toggle" if found else None


def _setup_gnome_shortcut(hotkey: str) -> bool:
    toggle_path = _toggle_command()
    if not toggle_path:
        print(f"  {RED}✗{RESET} voiceio not found on PATH")
        return False

    print(f"  {DIM}Command: {toggle_path}{RESET}")

    schema = "org.gnome.settings-daemon.plugins.media-keys"
    path = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/voiceio/"

    # Convert "super+v" -> "<Super>v"
    parts = hotkey.split("+")
    gnome_combo = "".join(f"<{p.capitalize()}>" for p in parts[:-1]) + parts[-1]

    try:
        subprocess.run([
            "gsettings", "set", f"{schema}.custom-keybinding:{path}",
            "name", "voiceio toggle"
        ], check=True, capture_output=True)
        subprocess.run([
            "gsettings", "set", f"{schema}.custom-keybinding:{path}",
            "command", toggle_path
        ], check=True, capture_output=True)
        subprocess.run([
            "gsettings", "set", f"{schema}.custom-keybinding:{path}",
            "binding", gnome_combo
        ], check=True, capture_output=True)

        result = subprocess.run(
            ["gsettings", "get", schema, "custom-keybindings"],
            capture_output=True, text=True, check=True
        )
        current = result.stdout.strip()
        if path not in current:
            if current == "@as []":
                new = f"['{path}']"
            else:
                new = current.rstrip("]") + f", '{path}']"
            subprocess.run([
                "gsettings", "set", schema, "custom-keybindings", new
            ], check=True, capture_output=True)

        return True
    except Exception as e:
        print(f"  {RED}✗{RESET} Failed: {e}")
        return False


def _test_hotkey(hotkey: str, backend: str) -> bool:
    """Start voiceio daemon briefly and test that the hotkey triggers."""
    from voiceio.hotkeys.socket_backend import SOCKET_PATH

    print(f"\n  {CYAN}Testing hotkey: {BOLD}{hotkey}{RESET}")
    print(f"  {DIM}Backend: {backend}{RESET}")

    if backend == "socket":
        # For socket backend, we test if the GNOME shortcut triggers voiceio toggle
        # Start a temporary socket listener
        import socket as sock

        SOCKET_PATH.unlink(missing_ok=True)

        s = sock.socket(sock.AF_UNIX, sock.SOCK_DGRAM)
        s.bind(str(SOCKET_PATH))
        s.settimeout(10.0)

        print(f"\n  {YELLOW}Press {BOLD}{hotkey}{RESET}{YELLOW} now (in any window)...{RESET}", end="", flush=True)

        try:
            data = s.recv(64)
            if data == b"toggle":
                print(f"\r  {GREEN}✓{RESET} Hotkey works!                    ")
                s.close()
                SOCKET_PATH.unlink(missing_ok=True)
                return True
        except sock.timeout:
            print(f"\r  {RED}✗{RESET} No hotkey detected after 10s      ")
            s.close()
            SOCKET_PATH.unlink(missing_ok=True)

            # Diagnostic: test if voiceio toggle itself works
            print(f"\n  {DIM}Diagnosing...{RESET}")
            venv_toggle = Path(sys.prefix) / "bin" / "voiceio"
            print(f"  {DIM}Running voiceio toggle directly...{RESET}", end="", flush=True)

            # Re-bind socket for the direct test
            s2 = sock.socket(sock.AF_UNIX, sock.SOCK_DGRAM)
            s2.bind(str(SOCKET_PATH))
            s2.settimeout(3.0)

            try:
                subprocess.run([str(venv_toggle), "toggle"], capture_output=True, timeout=3)
                data = s2.recv(64)
                if data == b"toggle":
                    print(f" {GREEN}OK{RESET}")
                    print(f"\n  {YELLOW}ℹ{RESET}  voiceio toggle works, but the DE shortcut didn't fire.")
                    print("    The GNOME shortcut may need a moment to register, or")
                    print(f"    {BOLD}{hotkey}{RESET} may conflict with an existing shortcut.")
            except Exception:
                print(f" {RED}FAIL{RESET}")
                print(f"\n  {YELLOW}ℹ{RESET}  voiceio toggle itself failed. This is unexpected.")

            s2.close()
            SOCKET_PATH.unlink(missing_ok=True)
            return False
    else:
        # For evdev/x11 backends, use the native listener
        triggered = threading.Event()

        def on_trigger():
            triggered.set()

        try:
            from voiceio.hotkeys import chain as hotkey_chain
            from voiceio import platform as plat
            platform = plat.detect()
            hk = hotkey_chain.select(platform, override=backend)
            hk.start(hotkey, on_trigger)
            stop = hk.stop
        except Exception as e:
            print(f"\n  {RED}✗{RESET} Backend failed: {e}")
            return False

        print(f"\n  {YELLOW}Press {BOLD}{hotkey}{RESET}{YELLOW} now...{RESET}", end="", flush=True)

        ok = triggered.wait(timeout=10.0)
        stop()

        if ok:
            print(f"\r  {GREEN}✓{RESET} Hotkey works!                    ")
            return True
        else:
            print(f"\r  {RED}✗{RESET} No hotkey detected after 10s      ")
            return False
