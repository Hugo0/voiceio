"""Health check / diagnostic report for all backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from voiceio import platform as plat
from voiceio.hotkeys import chain as hotkey_chain
from voiceio.typers import chain as typer_chain

log = logging.getLogger(__name__)


@dataclass
class BackendStatus:
    name: str
    ok: bool
    reason: str = ""
    fix_hint: str = ""
    fix_cmd: list[str] = field(default_factory=list)


@dataclass
class IBusStatus:
    component_installed: bool = False
    daemon_running: bool = False
    socket_reachable: bool = False
    gnome_source_configured: bool = False
    env_persisted: bool = False

@dataclass
class HealthReport:
    platform: plat.Platform
    hotkey_backends: list[BackendStatus] = field(default_factory=list)
    typer_backends: list[BackendStatus] = field(default_factory=list)
    audio_ok: bool = False
    audio_reason: str = ""
    cli_in_path: bool = False
    ibus_checks: IBusStatus | None = None
    tray_ok: bool = False
    tray_reason: str = ""
    tray_fix_hint: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        has_hotkey = any(b.ok for b in self.hotkey_backends)
        has_typer = any(b.ok for b in self.typer_backends)
        return has_hotkey and has_typer and self.audio_ok


def check_health(p: plat.Platform | None = None) -> HealthReport:
    """Run all probes and return a health report."""
    if p is None:
        p = plat.detect()

    report = HealthReport(platform=p)

    # Probe hotkey backends
    for name, backend, probe in hotkey_chain.resolve(p):
        report.hotkey_backends.append(BackendStatus(
            name=name, ok=probe.ok, reason=probe.reason,
            fix_hint=probe.fix_hint, fix_cmd=probe.fix_cmd,
        ))

    # Probe typer backends
    for name, backend, probe in typer_chain.resolve(p):
        report.typer_backends.append(BackendStatus(
            name=name, ok=probe.ok, reason=probe.reason,
            fix_hint=probe.fix_hint, fix_cmd=probe.fix_cmd,
        ))

    # Check audio
    try:
        import sounddevice as _sd
        _sd.query_devices(kind="input")
        report.audio_ok = True
    except Exception as e:
        report.audio_reason = str(e)

    # Check CLI in PATH
    from voiceio.service import symlinks_installed
    report.cli_in_path = symlinks_installed()

    # Tray icon
    from voiceio.tray import probe_availability
    report.tray_ok, report.tray_reason, report.tray_fix_hint = probe_availability()

    # IBus-specific checks (only if IBus is in the typer chain)
    ibus_backends = [b for b in report.typer_backends if b.name == "ibus"]
    if ibus_backends and ibus_backends[0].ok:
        report.ibus_checks = _check_ibus()

    # Platform-specific warnings
    import sys
    if sys.platform == "win32":
        active_hotkey = next((b.name for b in report.hotkey_backends if b.ok), None)
        if active_hotkey == "pynput":
            report.warnings.append(
                "Windows antivirus software may block pynput keyboard hooks. "
                "If hotkeys stop working, add voiceio to your antivirus exclusions."
            )
    elif sys.platform == "darwin":
        active_hotkey = next((b.name for b in report.hotkey_backends if b.ok), None)
        if active_hotkey == "pynput":
            report.warnings.append(
                "macOS requires Accessibility permission for pynput. "
                "Grant it in System Settings → Privacy & Security → Accessibility."
            )

    return report


def _check_ibus() -> IBusStatus:
    """Run detailed IBus health checks."""
    from pathlib import Path
    from voiceio.ibus import SOCKET_PATH
    from voiceio.typers.ibus import (
        _ibus_daemon_running, _component_installed,
    )

    status = IBusStatus()
    status.component_installed = _component_installed()
    status.daemon_running = _ibus_daemon_running()
    status.socket_reachable = SOCKET_PATH.exists()

    # Check GNOME input source
    try:
        from voiceio.platform import detect
        if detect().is_gnome:
            import subprocess
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"],
                capture_output=True, text=True, timeout=3,
            )
            status.gnome_source_configured = (
                result.returncode == 0 and "('ibus', 'voiceio')" in result.stdout
            )
    except Exception:
        pass

    # Check env persistence
    env_file = Path.home() / ".config" / "environment.d" / "voiceio.conf"
    status.env_persisted = env_file.exists()

    return status


def _icon(ok: bool, warn: bool = False) -> str:
    if ok:
        return "\u2714"  # ✔
    if warn:
        return "\u26A0"  # ⚠
    return "\u2718"  # ✘


def _backend_line(b: BackendStatus, active: bool = False) -> list[str]:
    lines = []
    icon = _icon(b.ok)
    suffix = " \u25C0 active" if active else ""
    line = f"  {icon} {b.name}{suffix}"
    if not b.ok:
        line += f"  \u2014 {b.reason}"
    lines.append(line)
    if b.fix_hint and not b.ok:
        lines.append(f"    \u2192 {b.fix_hint}")
    return lines


def format_report(report: HealthReport) -> str:
    """Format a health report as a human-readable string."""
    lines = []
    lines.append(f"Platform: {report.platform.os} / {report.platform.display_server} / {report.platform.desktop}")
    lines.append("")

    # Find which backends are active (first OK in each list)
    active_hotkey = next((b.name for b in report.hotkey_backends if b.ok), None)
    active_typer = next((b.name for b in report.typer_backends if b.ok), None)

    lines.append("Hotkey backends:")
    for b in report.hotkey_backends:
        lines.extend(_backend_line(b, active=(b.name == active_hotkey)))

    lines.append("")
    lines.append("Typer backends:")
    for b in report.typer_backends:
        lines.extend(_backend_line(b, active=(b.name == active_typer)))

    lines.append("")
    icon = _icon(report.audio_ok)
    line = f"Audio: {icon}"
    if not report.audio_ok:
        line += f"  \u2014 {report.audio_reason}"
    lines.append(line)

    icon = _icon(report.cli_in_path)
    line = f"CLI in PATH: {icon}"
    if not report.cli_in_path:
        line += "  \u2014 run 'voiceio doctor --fix' to create ~/.local/bin/ symlinks"
    lines.append(line)

    icon = _icon(report.tray_ok, warn=True)
    line = f"Tray icon: {icon}"
    if not report.tray_ok:
        line += f"  \u2014 {report.tray_reason}"
    lines.append(line)
    if report.tray_fix_hint and not report.tray_ok:
        lines.append(f"  \u2192 {report.tray_fix_hint}")

    if report.ibus_checks is not None:
        ibus = report.ibus_checks
        lines.append("")
        lines.append("IBus pipeline:")
        for label, ok, hint in [
            ("Component installed", ibus.component_installed, "voiceio setup"),
            ("Daemon running", ibus.daemon_running, "ibus-daemon -drxR"),
            ("Engine socket", ibus.socket_reachable, "start voiceio to activate"),
            ("GNOME input source", ibus.gnome_source_configured, "voiceio setup"),
            ("Env persisted (reboot-safe)", ibus.env_persisted, "voiceio setup"),
        ]:
            icon = _icon(ok, warn=True)
            line = f"  {icon} {label}"
            if not ok:
                line += f"  \u2014 {hint}"
            lines.append(line)

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  \u26A0 {w}")

    return "\n".join(lines)
