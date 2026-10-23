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
    except OSError:
        # sounddevice imports but PortAudio's shared library is missing.
        report.audio_reason = ("PortAudio library not found — install: "
                               + plat.system_deps_install_cmd(["portaudio"]))
    except ImportError:
        report.audio_reason = plat.desktop_dep_hint("sounddevice")
    else:
        try:
            _sd.query_devices(kind="input")
            report.audio_ok = True
        except Exception:
            # sounddevice raises an unhelpful "Error querying device -1" here.
            report.audio_reason = "no microphone / audio input device found"

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
        from voiceio.quiet_audio import describe
        quiet = describe(cfg.audio.pause_media, cfg.audio.duck_volume)
        if quiet:
            groups.processing["Quiet while recording"] = quiet
        from voiceio.idle_audio import describe as describe_idle
        groups.processing["Audio when idle"] = describe_idle(
            cfg.audio.keep_warm, cfg.audio.warm_minutes)

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

        if cfg.model.screen_vocabulary:
            from voiceio.context_terms import describe as describe_screen
            groups.accuracy["Screen vocabulary"] = f"on ({describe_screen()})"
        else:
            groups.accuracy["Screen vocabulary"] = "off"

        groups.accuracy.update(_decoder_status(cfg))
        groups.accuracy.update(_learning_status(cfg))

        if cfg.postcorrect.enabled:
            model = cfg.postcorrect.model or cfg.cloud.model
            groups.accuracy["Cloud postcorrect"] = f"on ({model}), adds up to {cfg.postcorrect.timeout_secs:g}s"
    except Exception:
        pass
    return groups


def _decoder_status(cfg) -> dict[str, str]:
    """For a sherpa-onnx model: whether it can be biased, and why not. For
    Whisper: whether Parakeet would do better here."""
    from voiceio.core.decoders.catalog import PARAKEET, runtime_of
    from voiceio.core.decoders.sherpa import features
    f = features()
    if runtime_of(cfg.model.name) == "sherpa-onnx":
        if not f.installed:
            return {"Decoder": "sherpa-onnx not installed — fix: voiceio models sherpa"}
        if not f.length_reward:
            return {"Decoder": "greedy, NO vocabulary biasing: this sherpa-onnx lacks the "
                               "NeMo TDT beam-search fix — fix: voiceio models sherpa"}
        how = "pre-tokenized" if f.pretokenized_hotwords else \
            "sherpa-tokenized, misses many names — fix: voiceio models sherpa"
        return {"Decoder": f"beam search, every proper noun as a hotword ({how})"}
    if f.length_reward and cfg.model.language == "en":
        return {"Decoder": f"Whisper; {PARAKEET} is ~2x faster here: "
                           f"voiceio models use {PARAKEET}"}
    return {}


def _learning_status(cfg) -> dict[str, str]:
    """Whether voiceio is learning your voice: the model in use, whether
    recordings are kept to learn from, and whether it retrains nightly."""
    from pathlib import Path

    from voiceio import config, service
    from voiceio.learn.store import learn_dir
    out: dict[str, str] = {}
    models = learn_dir() / "models"
    name = cfg.model.name
    from voiceio.core.decoders.catalog import is_whisper
    if models in Path(name).expanduser().parents:
        out["Voice model"] = f"fine-tuned on you ({Path(name).parent.name})"
    elif not is_whisper(name):
        out["Voice model"] = (f"{name} — `voiceio learn` fine-tunes Whisper, so its models "
                              "aren't used while this is the decoder")
    else:
        out["Voice model"] = f"stock ({name})"
    if cfg.data.retain_audio:
        n = sum(1 for _ in config.RECORDINGS_DIR.glob("*.wav")) \
            if config.RECORDINGS_DIR.exists() else 0
        out["Recordings"] = f"on ({n} kept to learn from)"
    else:
        out["Recordings"] = "off (nothing to learn from)"
    out["Scheduled learning"] = service.learn_schedule_summary()
    return out


def _check_ibus() -> IBusStatus:
    """Run detailed IBus health checks."""
    from pathlib import Path
    from voiceio.ibus.install import component_installed, daemon_running

    status = IBusStatus()
    status.component_installed = component_installed()
    status.daemon_running = daemon_running()
    # Ping, not exists(): an engine can hold a socket file yet be deaf.
    from voiceio.ibus import ping_engine
    status.socket_reachable = ping_engine(timeout=1.0)

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


# ── ANSI colors (shared with the setup wizard) ───────────────────────────────

from voiceio.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402


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

    if corr == "none" and "entries" in hist:
        tips.append("voiceio vocab suggest  — unusual words from your history")
    elif corr == "none":
        tips.append("voiceio corrections add WRONG RIGHT  — fix a word that keeps being misheard")
    if vocab == "none":
        tips.append("Add names/jargon to ~/.config/voiceio/vocabulary.txt")
    if accuracy.get("Recordings", "").startswith("off"):
        tips.append("voiceio config set data.retain_audio true  — keep recordings so voiceio can learn your voice")
    elif accuracy.get("Scheduled learning") == "off":
        tips.append("voiceio learn schedule on  — retrain on your voice every night")
    return tips


def format_report(report: HealthReport) -> str:
    """Format a health report as a human-readable string."""
    from voiceio.setup.ui import LOGO_DOCTOR

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
            ("Engine answers", ibus.socket_reachable,
             "voiceio service restart (or start voiceio)"),
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
        lines.append(f"  {DIM}What voiceio knows about your words, and whether it keeps learning.{RESET}")
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


# ── Usage checks: how it is actually performing, from the user's own data ──

def _latency_hint(cfg) -> str:
    """Only suggest settings that differ from what is already in effect."""
    hints = []
    if cfg.model.decode_lanes < 2:
        hints.append("voiceio config set model.decode_lanes 2")
    if cfg.output.streaming_freeze_secs > 8:
        hints.append("voiceio config set output.streaming_freeze_secs 8")
    if cfg.postcorrect.enabled:
        hints.append("voiceio config set postcorrect.enabled false")
    return "  ·  ".join(hints)


def usage_findings(cfg=None) -> list[tuple[bool, str, str]]:
    """(ok, finding, hint) from history/traces/logs. Cheap: reads local files.

    These catch what the probes can't: everything can be "installed" while
    stopping a dictation takes 20s or the worker crashes daily.
    """
    import json
    import statistics
    import time

    from voiceio import config
    out: list[tuple[bool, str, str]] = []

    from voiceio import __version__
    waits: list[tuple[float, str | None]] = []
    if config.HISTORY_PATH.exists():
        for line in config.HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-400:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lat = row.get("latency") or {}
            if "finalize_total" in lat:
                waits.append((lat["finalize_total"], row.get("v")))
    # Judge this release on its own dictations once there are enough of them.
    current = [w for w, v in waits if v == __version__]
    if len(current) >= 10:
        sample, scope = current, f"last {len(current)} on {__version__}"
    else:
        sample, scope = [w for w, _ in waits], f"last {len(waits)} dictations"
        if waits and not current:
            scope += f", all before {__version__}"
    if len(sample) >= 10:
        p50 = statistics.median(sample)
        p90 = statistics.quantiles(sample, n=10, method="inclusive")[-1]
        out.append((p90 <= 10, f"stop→text: median {p50:.1f}s, p90 {p90:.1f}s ({scope})",
                    _latency_hint(cfg or config.load())))

    # Rotated logs are voiceio.log.N with N=1 the newest: read oldest first so
    # each line is attributed to the release whose startup preceded it.
    log_files = sorted(config.LOG_DIR.glob("voiceio.log*"), reverse=True) \
        if config.LOG_DIR.exists() else []
    cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - 7 * 86400))
    startup = f"=== voiceio v{__version__} startup ==="
    # [older releases, this release]; a hang's kill also logs "Worker died",
    # so a death right after a timeout is the same incident.
    hangs, crashes = [0, 0], [0, 0]
    current = killed = False
    for path in log_files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if "startup ===" in line:
                current = startup in line
            if line[:10] < cutoff:        # log lines start with an ISO date
                continue
            if "Transcription timed out" in line:
                hangs[current] += 1
                killed = True
            elif "Worker died" in line:
                crashes[current] += not killed
                killed = False
    if log_files:
        older = hangs[0] + crashes[0]
        note = f" ({older} more before {__version__})" if older else ""
        out.append((hangs[1] + crashes[1] < 5,
                    f"decoder: {hangs[1]} hangs, {crashes[1]} crashes in the last 7 days "
                    f"on {__version__}{note}",
                    f"stack dumps of hung decodes: {config.LOG_DIR / 'worker.log'}"))

    try:
        from voiceio.tokens import HOTWORDS_TOKEN_BUDGET
        from voiceio.vocabulary import VocabularyLoader
        loader = VocabularyLoader(config.load().model)
        total = len(loader.get_all())
        if total:
            biasing = len(loader.get_selected(token_budget=HOTWORDS_TOKEN_BUDGET))
            out.append((True, f"vocabulary: {biasing} of {total} terms bias the decoder",
                        "see which: voiceio vocab show"))
    except Exception:  # noqa: BLE001 — a diagnostic must not crash doctor
        pass

    try:
        from voiceio.corrections import CorrectionDict, risky_rules
        risky = len(risky_rules(CorrectionDict().list_all()))
        if risky:
            out.append((False, f"corrections: {risky} rule(s) may rewrite the wrong words",
                        "review them: voiceio corrections audit"))
    except Exception:  # noqa: BLE001
        pass

    try:
        from voiceio.learn.clean import clean
        junk = sum(r.removed for r in clean(apply=False))
        if junk:
            out.append((False, f"{junk} test-fixture rows in your history/traces",
                        "voiceio learn clean --apply"))
    except Exception:  # noqa: BLE001
        pass
    return out


def format_usage(findings: list[tuple[bool, str, str]]) -> str:
    if not findings:
        return ""
    lines = [f"\n{BOLD}Usage{RESET}"]
    for ok, finding, hint in findings:
        mark = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}!{RESET}"
        lines.append(f"  {mark} {finding}")
        if not ok and hint:
            lines.append(f"      {DIM}{hint}{RESET}")
    return "\n".join(lines)
