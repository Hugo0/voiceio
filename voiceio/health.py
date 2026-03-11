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
class FeatureGroups:
    processing: dict[str, str] = field(default_factory=dict)
    accuracy: dict[str, str] = field(default_factory=dict)

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
    features: FeatureGroups = field(default_factory=FeatureGroups)
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

    # Check features
    report.features = _check_features()

    # IBus-specific checks (only if IBus is in the typer chain)
    ibus_backends = [b for b in report.typer_backends if b.name == "ibus"]
    if ibus_backends and ibus_backends[0].ok:
        report.ibus_checks = _check_ibus()

    # Platform-specific warnings
    active_hotkey = next((b.name for b in report.hotkey_backends if b.ok), None)
    if active_hotkey == "pynput":
        if report.platform.is_windows:
            report.warnings.append(
                "Windows antivirus software may block pynput keyboard hooks. "
                "If hotkeys stop working, add voiceio to your antivirus exclusions."
            )
        elif report.platform.is_mac:
            report.warnings.append(
                "macOS requires Accessibility permission for pynput. "
                "Grant it in System Settings → Privacy & Security → Accessibility."
            )

    return report


def _check_features() -> FeatureGroups:
    """Check which features are active based on config and files."""
    groups = FeatureGroups()
    try:
        from voiceio.config import load, CORRECTIONS_PATH, HISTORY_PATH
        cfg = load()

        # ── Processing toggles ──
        groups.processing["Voice commands"] = "on" if cfg.commands.enabled else "off"
        groups.processing["Smart punctuation"] = "on" if cfg.output.punctuation_cleanup else "off"
        groups.processing["Number conversion"] = "on" if cfg.output.number_conversion else "off"
        groups.processing["Silero VAD"] = cfg.audio.vad_backend

        # LLM
        if cfg.llm.enabled:
            from voiceio.llm import OllamaStatus, diagnose_ollama
            status, _ = diagnose_ollama(cfg.llm)
            if status == OllamaStatus.OK:
                groups.processing["LLM cleanup"] = f"on ({cfg.llm.model or 'auto'} via Ollama)"
            elif status == OllamaStatus.NOT_INSTALLED:
                groups.processing["LLM cleanup"] = "enabled but Ollama not installed (ollama.com)"
            elif status == OllamaStatus.NOT_RUNNING:
                groups.processing["LLM cleanup"] = "enabled but Ollama not running (ollama serve)"
            elif status == OllamaStatus.MODEL_NOT_FOUND:
                model = cfg.llm.model or "any"
                groups.processing["LLM cleanup"] = f"enabled but model '{model}' not found (ollama pull {cfg.llm.model})"
        else:
            groups.processing["LLM cleanup"] = "off (enable in config or re-run 'voiceio setup')"

        # TTS
        if cfg.tts.enabled:
            from voiceio.tts import probe_all as tts_probe_all
            results = tts_probe_all(cfg.tts)
            active = next((name for name, probe in results if probe.ok), None)
            if active:
                groups.processing["Text-to-speech"] = f"on ({active})"
            else:
                groups.processing["Text-to-speech"] = "enabled but no engine available"
        else:
            groups.processing["Text-to-speech"] = "off"

        # ── Accuracy (learning stack) ──

        # History
        if HISTORY_PATH.exists():
            try:
                raw_lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
                lines = [ln for ln in raw_lines if ln.strip()]
                if lines:
                    groups.accuracy["History"] = f"{len(lines)} entries"
                else:
                    groups.accuracy["History"] = "empty"
            except OSError:
                groups.accuracy["History"] = "active"
        else:
            groups.accuracy["History"] = "no file yet (starts on first dictation)"

        # Corrections dictionary
        if CORRECTIONS_PATH.exists():
            import json
            try:
                raw = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
                n = len(raw) if isinstance(raw, dict) else 0
                groups.accuracy["Corrections"] = f"{n} rule(s)" if n else "file exists, empty"
            except (json.JSONDecodeError, OSError):
                groups.accuracy["Corrections"] = "file exists, unreadable"
        else:
            groups.accuracy["Corrections"] = "none"

        # Flagged words
        from voiceio.config import FLAGGED_PATH
        if FLAGGED_PATH.exists():
            try:
                raw = FLAGGED_PATH.read_text(encoding="utf-8").splitlines()
                flines = [w.strip() for w in raw if w.strip()]
                if flines:
                    groups.accuracy["Flagged words"] = f"{len(flines)} word(s)"
            except OSError:
                pass

        # Vocabulary
        if cfg.model.vocabulary_file:
            groups.accuracy["Vocabulary"] = cfg.model.vocabulary_file
        else:
            from voiceio.config import CONFIG_DIR
            default = CONFIG_DIR / "vocabulary.txt"
            if default.exists():
                groups.accuracy["Vocabulary"] = str(default)
            else:
                groups.accuracy["Vocabulary"] = "none"

        # Autocorrect
        from voiceio.llm_api import resolve_api_key
        api_key = resolve_api_key(cfg.autocorrect)
        if api_key:
            groups.accuracy["Autocorrect"] = f"API ({cfg.autocorrect.model})"
        elif cfg.llm.enabled:
            groups.accuracy["Autocorrect"] = "Ollama fallback"
        else:
            groups.accuracy["Autocorrect"] = "no API key (set in config or OPENROUTER_API_KEY env var)"
    except Exception:
        pass
    return groups


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


# ── ANSI colors (shared with wizard.py) ──────────────────────────────────────

from voiceio.wizard import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402


def _icon(ok: bool, warn: bool = False) -> str:
    if ok:
        return f"{GREEN}✓{RESET}"
    if warn:
        return f"{YELLOW}⚠{RESET}"
    return f"{RED}✗{RESET}"


def _section(title: str) -> str:
    return f"\n{BOLD}{title}{RESET}\n{DIM}{'─' * 40}{RESET}"


def _check_line(label: str, ok: bool, detail: str = "",
                warn: bool = False) -> str:
    icon = _icon(ok, warn=warn)
    extra = f"  {DIM}{detail}{RESET}" if detail else ""
    return f"  {icon} {label}{extra}"


def _backend_line(b: BackendStatus, active: bool = False) -> list[str]:
    lines = []
    tag = f"  {CYAN}◀ active{RESET}" if active else ""
    detail = ""
    if not b.ok:
        detail = b.reason
    lines.append(_check_line(f"{b.name}{tag}", b.ok, detail))
    if b.fix_hint and not b.ok:
        lines.append(f"    {DIM}→ {b.fix_hint}{RESET}")
    return lines


def _feature_line(name: str, status: str) -> str:
    """Format a feature line with colored status indicator."""
    s = status.lower()
    if s.startswith("on") or s.startswith("active") or s.startswith("silero"):
        icon = f"{GREEN}✓{RESET}"
    elif s.startswith("off") or s.startswith("none") or s.startswith("no ") \
            or s.startswith("empty"):
        icon = f"{DIM}○{RESET}"
    elif "not " in s or "error" in s:
        icon = f"{RED}✗{RESET}"
    elif "enabled but" in s:
        icon = f"{YELLOW}⚠{RESET}"
    else:
        icon = f"{GREEN}✓{RESET}"
    return f"  {icon} {BOLD}{name}{RESET}  {DIM}{status}{RESET}"


def _accuracy_tips(accuracy: dict[str, str]) -> list[str]:
    """Generate actionable tips based on accuracy feature status."""
    tips = []
    hist = accuracy.get("History", "")
    corr = accuracy.get("Corrections", "")
    vocab = accuracy.get("Vocabulary", "")
    auto = accuracy.get("Autocorrect", "")

    if corr == "none" and "entries" in hist:
        tips.append("voiceio correct --auto  — scan history for Whisper mistakes")
    elif corr == "none":
        tips.append("voiceio correct \"wrong\" \"right\"  — add a correction rule")
    if vocab == "none":
        tips.append("Add names/jargon to ~/.config/voiceio/vocabulary.txt")
    if "no API key" in auto:
        tips.append("Set OPENROUTER_API_KEY for cloud-powered autocorrect")
    return tips


def format_report(report: HealthReport) -> str:
    """Format a health report as a human-readable string."""
    from voiceio.wizard import LOGO_DOCTOR

    lines = [LOGO_DOCTOR]

    # Platform header
    p = report.platform
    lines.append(
        f"  {BOLD}Platform{RESET}  "
        f"{p.os} / {p.display_server} / {p.desktop}"
    )

    # Find which backends are active (first OK in each list)
    active_hotkey = next((b.name for b in report.hotkey_backends if b.ok), None)
    active_typer = next((b.name for b in report.typer_backends if b.ok), None)

    # Hotkey backends
    lines.append(_section("Hotkey backends"))
    for b in report.hotkey_backends:
        lines.extend(_backend_line(b, active=(b.name == active_hotkey)))

    # Typer backends
    lines.append(_section("Typer backends"))
    for b in report.typer_backends:
        lines.extend(_backend_line(b, active=(b.name == active_typer)))

    # System checks
    lines.append(_section("System"))
    lines.append(_check_line("Audio input", report.audio_ok,
                             report.audio_reason if not report.audio_ok else ""))
    lines.append(_check_line("CLI in PATH", report.cli_in_path,
                             "run 'voiceio doctor --fix'" if not report.cli_in_path else ""))
    lines.append(_check_line("Tray icon", report.tray_ok,
                             report.tray_reason if not report.tray_ok else "",
                             warn=not report.tray_ok))
    if report.tray_fix_hint and not report.tray_ok:
        lines.append(f"    {DIM}→ {report.tray_fix_hint}{RESET}")

    # IBus pipeline
    if report.ibus_checks is not None:
        ibus = report.ibus_checks
        lines.append(_section("IBus pipeline"))
        for label, ok, hint in [
            ("Component installed", ibus.component_installed, "voiceio setup"),
            ("Daemon running", ibus.daemon_running, "ibus-daemon -drxR"),
            ("Engine socket", ibus.socket_reachable, "start voiceio to activate"),
            ("GNOME input source", ibus.gnome_source_configured, "voiceio setup"),
            ("Env persisted (reboot-safe)", ibus.env_persisted, "voiceio setup"),
        ]:
            lines.append(_check_line(label, ok, hint if not ok else "",
                                     warn=not ok))

    # Processing features
    if report.features.processing:
        lines.append(_section("Processing"))
        for name, status in report.features.processing.items():
            lines.append(_feature_line(name, status))

    # Accuracy / learning stack
    if report.features.accuracy:
        lines.append(_section("Accuracy"))
        lines.append(f"  {DIM}Learns from your dictation history to fix recurring mistakes.{RESET}")
        for name, status in report.features.accuracy.items():
            lines.append(_feature_line(name, status))
        # Show actionable tips at the bottom
        tips = _accuracy_tips(report.features.accuracy)
        for tip in tips:
            lines.append(f"  {DIM}→ {tip}{RESET}")

    # Warnings
    if report.warnings:
        lines.append(_section("Warnings"))
        for w in report.warnings:
            lines.append(f"  {YELLOW}⚠{RESET} {w}")

    return "\n".join(lines)
