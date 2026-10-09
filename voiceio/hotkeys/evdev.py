"""Evdev hotkey backend: reads /dev/input directly."""
from __future__ import annotations

import logging
import shlex
import threading
import time
from pathlib import Path
from typing import Callable

from voiceio.backends import ProbeResult

log = logging.getLogger(__name__)

DEBOUNCE_SECS = 0.8

UACCESS_RULE_NAME = "60-voiceio-uaccess.rules"
UACCESS_RULE_PATH = f"/etc/udev/rules.d/{UACCESS_RULE_NAME}"
_UACCESS_RULE_SRC = Path(__file__).resolve().parent.parent / "data" / UACCESS_RULE_NAME


def uaccess_rule() -> str:
    """The rule line(s) of the shipped udev rule, comments stripped."""
    lines = _UACCESS_RULE_SRC.read_text().splitlines()
    return "".join(f"{ln}\n" for ln in lines if ln.strip() and not ln.startswith("#"))


def uaccess_install_cmd() -> list[str]:
    """argv that installs the uaccess rule and applies it (runs via sudo).

    One ``sh -c`` so the whole privileged action shows as a single string in
    doctor's consent prompt. `trigger` re-runs the rules for existing input
    devices, so access is granted immediately — no re-login, unlike the
    `input` group.
    """
    script = (
        f"printf %s {shlex.quote(uaccess_rule())} "
        f"| sudo tee {UACCESS_RULE_PATH} > /dev/null "
        "&& sudo udevadm control --reload-rules "
        "&& sudo udevadm trigger --subsystem-match=input --action=change"
    )
    return ["sh", "-c", script]


def nixos_hint() -> str:
    return (
        "in configuration.nix: services.udev.extraRules = ''"
        + uaccess_rule().strip()
        + "''; (or enable the voiceio flake's programs.voiceio), then rebuild"
    )


def access_hint() -> str:
    """Short human hint for granting evdev access, used by app and wizard."""
    from voiceio.platform import is_nixos
    if is_nixos():
        return nixos_hint()
    return (
        f"`voiceio doctor --fix` installs {UACCESS_RULE_PATH} (works at once). "
        "Fallback: sudo usermod -aG input $USER, then log out/in"
    )


class EvdevHotkey:
    """Hotkey detection via Linux evdev (needs read access to /dev/input)."""

    name = "evdev"

    def probe(self) -> ProbeResult:
        try:
            import evdev  # noqa: F401
        except ImportError:
            from voiceio.platform import desktop_dep_hint
            return ProbeResult(ok=False, reason="evdev package not installed",
                               fix_hint=desktop_dep_hint("evdev"))

        # OK if *any* device is readable. A per-device ACL (udev `uaccess`)
        # grants keyboards only, so the first node being unreadable proves
        # nothing.
        denied = False
        for path in sorted(Path("/dev/input/").glob("event*")):
            try:
                with open(path, "rb"):
                    return ProbeResult(ok=True)
            except PermissionError:
                denied = True
            except OSError:
                continue
        if denied:
            from voiceio.platform import is_nixos
            if is_nixos():
                # /etc is read-only on NixOS; the rule must be declared.
                return ProbeResult(
                    ok=False, reason="No permission to read /dev/input",
                    fix_hint=nixos_hint(),
                )
            return ProbeResult(
                ok=False, reason="No permission to read /dev/input",
                fix_hint=(
                    f"install udev rule {UACCESS_RULE_PATH} (grants the logged-in "
                    "user its keyboards, no re-login). Fallback: "
                    "sudo usermod -aG input $USER, then log out/in"
                ),
                fix_cmd=uaccess_install_cmd(),
            )
        return ProbeResult(ok=False, reason="No input devices found")

    def start(self, combo: str, on_trigger: Callable[[], None]) -> None:
        import evdev
        from evdev import ecodes

        MODIFIER_MAP = {
            "super": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},
            "ctrl": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
            "alt": {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT},
            "shift": {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT},
        }

        KEY_MAP = {
            **{chr(c): getattr(ecodes, f"KEY_{chr(c).upper()}") for c in range(ord("a"), ord("z") + 1)},
            **{str(i): getattr(ecodes, f"KEY_{i}") for i in range(10)},
            **{f"f{i}": getattr(ecodes, f"KEY_F{i}") for i in range(1, 13)},
            "space": ecodes.KEY_SPACE,
            "pause": ecodes.KEY_PAUSE,
            "insert": ecodes.KEY_INSERT,
            "scroll_lock": ecodes.KEY_SCROLLLOCK,
            "print_screen": ecodes.KEY_SYSRQ,
        }

        parts = [p.strip().lower() for p in combo.split("+")]
        required_mods: set[int] = set()
        for mod in parts[:-1]:
            required_mods.update(MODIFIER_MAP.get(mod, set()))
        key_code = KEY_MAP.get(parts[-1])
        if key_code is None:
            raise ValueError(f"Unknown key: {parts[-1]}")

        keyboards = []
        for path in sorted(Path("/dev/input/").glob("event*")):
            try:
                dev = evdev.InputDevice(str(path))
                caps = dev.capabilities(verbose=False)
                if ecodes.EV_KEY in caps:
                    keys = caps[ecodes.EV_KEY]
                    if ecodes.KEY_A in keys and ecodes.KEY_Z in keys:
                        keyboards.append(dev)
            except (PermissionError, OSError):
                continue

        if not keyboards:
            raise RuntimeError("No keyboard devices accessible")

        self._running = threading.Event()
        self._running.set()
        pressed: set[int] = set()
        pressed_lock = threading.Lock()
        last_trigger = [0.0]

        def check_mods() -> bool:
            for codes in MODIFIER_MAP.values():
                if codes & required_mods and not (codes & pressed):
                    return False
            return True

        def read_device(dev: evdev.InputDevice) -> None:
            try:
                for event in dev.read_loop():
                    if not self._running.is_set():
                        break
                    if event.type != ecodes.EV_KEY:
                        continue
                    key_event = evdev.categorize(event)
                    should_trigger = False
                    with pressed_lock:
                        if key_event.keystate == evdev.KeyEvent.key_down:
                            pressed.add(event.code)
                            if event.code == key_code and check_mods():
                                now = time.monotonic()
                                since = now - last_trigger[0]
                                if since >= DEBOUNCE_SECS:
                                    last_trigger[0] = now
                                    should_trigger = True
                        elif key_event.keystate == evdev.KeyEvent.key_up:
                            pressed.discard(event.code)
                    if should_trigger:
                        on_trigger()
            except OSError:
                pass

        self._threads = []
        for dev in keyboards:
            t = threading.Thread(target=read_device, args=(dev,), daemon=True)
            t.start()
            self._threads.append(t)

        log.info("Evdev hotkey listener started on %d keyboards", len(keyboards))

    def stop(self) -> None:
        if hasattr(self, "_running"):
            self._running.clear()
