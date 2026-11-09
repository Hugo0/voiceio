"""On X11 pynput covers the hotkey: evdev access (a root-only udev rule) is
optional there, and setup/doctor must not fail on it or offer the install."""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voiceio import config, health
from voiceio import platform as plat
from voiceio.backends import ProbeResult
from voiceio.hotkeys import evdev as evdev_mod
from voiceio.platform import Platform

_DENIED = ProbeResult(ok=False, reason="No permission to read /dev/input",
                      fix_hint="install udev rule", fix_cmd=evdev_mod.uaccess_install_cmd())


def _cfg(backend="auto"):
    return SimpleNamespace(hotkey=SimpleNamespace(backend=backend),
                           llm=SimpleNamespace(enabled=False))


def _report(display, pynput_ok=True, backend="auto", monkeypatch=None):
    monkeypatch.setattr(config, "load", lambda *a, **k: _cfg(backend))
    p = Platform(os="linux", display_server=display, desktop="kde")
    chain = ([("pynput", MagicMock(), ProbeResult(ok=pynput_ok, reason="pynput missing"))]
             if display == "x11" else [])
    chain += [("evdev", MagicMock(), _DENIED), ("socket", MagicMock(), ProbeResult(ok=True))]
    with patch("voiceio.health.hotkey_chain.resolve", lambda platform: chain), \
         patch("voiceio.health.typer_chain.resolve",
               lambda platform: [("xdotool", MagicMock(), ProbeResult(ok=True))]), \
         patch.dict("sys.modules", {"sounddevice": MagicMock()}):
        return health.check_health(p)


def _evdev(report):
    return next(b for b in report.hotkey_backends if b.name == "evdev")


@pytest.mark.parametrize("display,pynput_ok,backend,expected", [
    ("x11", True, "auto", True),
    ("x11", True, "pynput", True),
    ("x11", True, "evdev", False),      # pinned: evdev is the hotkey
    ("x11", False, "auto", False),      # no pynput: evdev is next in the chain
    ("wayland", True, "auto", False),   # evdev is the only physical listener
    ("unknown", True, "auto", False),
])
def test_evdev_optional(display, pynput_ok, backend, expected):
    assert evdev_mod.evdev_optional(display, pynput_ok, backend) is expected


def test_doctor_shows_evdev_optional_on_x11_with_pynput(monkeypatch):
    report = _report("x11", monkeypatch=monkeypatch)
    b = _evdev(report)
    assert b.optional and not b.fix_cmd and not b.fix_hint
    assert report.all_ok
    line = next(ln for ln in health.format_report(report).splitlines() if "evdev" in ln)
    assert "pynput handles the hotkey on X11" in line
    assert "✗" not in line
    assert "udev" not in health.format_report(report)


@pytest.mark.parametrize("display,pynput_ok,backend", [
    ("wayland", True, "auto"), ("x11", False, "auto"), ("x11", True, "evdev"),
])
def test_doctor_keeps_evdev_fix_elsewhere(monkeypatch, display, pynput_ok, backend):
    b = _evdev(_report(display, pynput_ok, backend, monkeypatch=monkeypatch))
    assert not b.optional
    assert b.fix_cmd == evdev_mod.uaccess_install_cmd()


def _run_doctor_fix(monkeypatch, report):
    from voiceio import install_env
    from voiceio.cli import doctor
    privileged, ran = [], []
    monkeypatch.setattr(health, "check_health", lambda: report)
    monkeypatch.setattr(health, "format_report", lambda r: "")
    monkeypatch.setattr(health, "usage_findings", lambda: [])
    monkeypatch.setattr(plat, "check_system_deps", lambda: [])
    monkeypatch.setattr(config, "check_permissions", lambda: [])
    monkeypatch.setattr(install_env, "stale_install_paths", lambda: [])
    monkeypatch.setattr(install_env, "running_ephemeral", lambda: False)
    monkeypatch.setattr(doctor, "_offer_report", lambda *a: False)
    monkeypatch.setattr(doctor, "_run_privileged", lambda *a, **k: privileged.append(a))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))
    report.cli_in_path = True
    with pytest.raises(SystemExit):
        doctor._cmd_doctor(argparse.Namespace(live=False, fix=True, report=False))
    return privileged, ran


def test_doctor_fix_skips_udev_install_on_x11_with_pynput(monkeypatch):
    privileged, ran = _run_doctor_fix(monkeypatch, _report("x11", monkeypatch=monkeypatch))
    assert privileged == [] and ran == []


def test_doctor_fix_still_offers_udev_install_on_wayland(monkeypatch):
    privileged, _ = _run_doctor_fix(monkeypatch, _report("wayland", monkeypatch=monkeypatch))
    assert [a[0] for a in privileged] == ["Fix evdev"]


def test_probe_drops_udev_fix_when_pynput_covers(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: False)
    monkeypatch.setattr(evdev_mod, "_pynput_covers_hotkey", lambda: True)
    (tmp_path / "event0").write_text("")
    monkeypatch.setattr(Path, "glob", lambda self, pat: [tmp_path / "event0"])

    def deny(*a, **k):
        raise PermissionError
    monkeypatch.setattr("builtins.open", deny)
    result = evdev_mod.EvdevHotkey().probe()
    assert not result.ok and not result.fix_cmd
    assert result.fix_hint == evdev_mod.NOT_NEEDED_ON_X11


def test_pynput_covers_hotkey_reads_display_and_config(monkeypatch):
    from voiceio.hotkeys import pynput_backend
    monkeypatch.setattr(pynput_backend.PynputHotkey, "probe", lambda self: ProbeResult(ok=True))
    monkeypatch.setattr(plat, "_detect_display_server", lambda: "x11")
    monkeypatch.setattr(config, "load", lambda *a, **k: _cfg("auto"))
    assert evdev_mod._pynput_covers_hotkey()
    monkeypatch.setattr(config, "load", lambda *a, **k: _cfg("evdev"))
    assert not evdev_mod._pynput_covers_hotkey()
    monkeypatch.setattr(config, "load", lambda *a, **k: _cfg("auto"))
    monkeypatch.setattr(plat, "_detect_display_server", lambda: "wayland")
    assert not evdev_mod._pynput_covers_hotkey()


# ── setup ────────────────────────────────────────────────────────────────

_CHECKS = {
    "is_windows": False, "is_mac": False, "is_linux": True, "display": "x11",
    "audio": True, "audio_devices": [object()], "pynput": True, "xdotool": True,
    "xclip": False, "ydotool": False, "wtype": False, "ibus": False, "ibus_gi": False,
    "cuda": False, "evdev_access": False, "evdev_optional": True,
}


def test_setup_shows_evdev_optional_on_x11(capsys):
    from voiceio.setup import steps
    steps._print_capability_checks(dict(_CHECKS))
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "evdev" in ln)
    assert "pynput handles the hotkey on X11" in line
    assert "✗" not in line and "doctor --fix" not in line


def test_setup_keeps_evdev_hint_on_wayland(capsys, monkeypatch):
    from voiceio.setup import steps
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: False)
    steps._print_capability_checks(dict(_CHECKS, display="wayland", pynput=False,
                                        evdev_optional=False))
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "evdev" in ln)
    assert "✗" in line and "doctor --fix" in line


def test_check_system_detects_pynput_on_startx_x11(monkeypatch):
    """XDG_SESSION_TYPE=tty under startx: DISPLAY still says X11."""
    from voiceio.hotkeys import pynput_backend
    from voiceio.setup import system
    monkeypatch.setenv("XDG_SESSION_TYPE", "tty")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(plat, "_detect_os", lambda: "linux")
    monkeypatch.setattr(pynput_backend.PynputHotkey, "probe", lambda self: ProbeResult(ok=True))
    monkeypatch.setattr(evdev_mod.EvdevHotkey, "probe", lambda self: _DENIED)
    checks = system._check_system()
    assert checks["display"] == "x11"
    assert checks["pynput"] and not checks["evdev_access"] and checks["evdev_optional"]


def test_setup_picks_auto_backend_on_x11_without_evdev(monkeypatch):
    from voiceio.setup import steps
    monkeypatch.setattr("voiceio.ibus.install.ensure", lambda: True)
    state = steps.SetupState(checks=dict(_CHECKS))
    steps._select_output_backend(state)
    assert state.backend == "auto"     # pynput first, not the socket-only fallback


# ── install commands ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd,rule_path", [
    (evdev_mod.uaccess_install_cmd(), evdev_mod.UACCESS_RULE_PATH),
])
def test_install_cmd_creates_rules_dir_first(cmd, rule_path):
    script = cmd[-1]
    assert cmd[:3] == ["sudo", "sh", "-c"]
    assert script.index("mkdir -p /etc/udev/rules.d") < script.index(f"> {rule_path}")
    assert "sudo" not in script


def test_ydotool_install_cmd_creates_rules_dir_first():
    from voiceio.typers import ydotool
    test_install_cmd_creates_rules_dir_first(ydotool.uinput_udev_install_cmd(),
                                             ydotool.UINPUT_UDEV_RULE_PATH)
