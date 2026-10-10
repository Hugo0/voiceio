"""Non-interactive setup for agents and scripts: answers dict in, exit code out."""
from __future__ import annotations

import os
import subprocess

from voiceio.config import CONFIG_PATH
from voiceio.setup.configfile import _record_cloud_consent, _write_config
from voiceio.setup.hotkey import _setup_gnome_shortcut
from voiceio.setup.speech import MODELS, _download_model
from voiceio.setup.system import _check_system


# ── Non-interactive setup (agents & scripts) ─────────────────────────────

_ALLOWED_ANSWER_KEYS = {
    "model", "language", "hotkey", "backend", "streaming",
    "tts_enabled", "tts_engine", "tts_speed",
    "llm_enabled", "llm_model",
    "cloud_api_key", "cloud_base_url", "cloud_model",
    "sound_enabled", "notify_clipboard", "commands_enabled",
    "punctuation_cleanup", "number_conversion", "voice_input_prefix",
    "tray_enabled", "install_service", "download_model", "install_system_deps",
}


def _progress(step: str, **fields) -> None:
    """Emit one machine-readable progress line to stdout."""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[voiceio-setup] step={step}"
    if parts:
        line += " " + parts
    print(line, flush=True)


def _fail(step: str, reason: str, code: int) -> int:
    _progress(step, status="error", reason=repr(reason))
    return code


def run_setup_noninteractive(answers: dict | None = None, *, install_deps: bool = False) -> int:
    """Non-interactive setup for agents/scripts. Returns a process exit code.

    Accepts ``answers`` (any subset of _ALLOWED_ANSWER_KEYS); unspecified values
    fall back to recommended defaults. Emits machine-readable ``[voiceio-setup]``
    progress lines and never reads stdin. Exit code 0 = success; non-zero with a
    printed reason otherwise.
    """
    answers = dict(answers or {})
    # Pre-1.0 name of the cloud endpoint answers.
    for key in [k for k in answers if k.startswith("autocorrect_")]:
        answers.setdefault("cloud_" + key.removeprefix("autocorrect_"), answers.pop(key))

    # ── Validate at the boundary ────────────────────────────────────────
    unknown = set(answers) - _ALLOWED_ANSWER_KEYS
    if unknown:
        return _fail("validate", f"unknown answer keys: {sorted(unknown)}", 2)
    valid_models = {m[0] for m in MODELS}
    model = str(answers.get("model", "small"))
    if model not in valid_models:
        return _fail("validate", f"invalid model '{model}'; choose from {sorted(valid_models)}", 2)
    hotkey = str(answers.get("hotkey", "ctrl+alt+v")).strip()
    if not hotkey:
        return _fail("validate", "hotkey must not be empty", 2)
    language = str(answers.get("language", "en")).strip() or "en"

    _progress("start", mode="noninteractive", model=model, language=language, hotkey=hotkey)

    from voiceio import platform as plat

    # ── Preflight ───────────────────────────────────────────────────────
    missing = plat.check_system_deps()
    if missing:
        cmd = plat.system_deps_install_cmd(missing)
        _progress("preflight", status="missing", deps=",".join(missing), cmd=repr(cmd))
        if install_deps or answers.get("install_system_deps"):
            _progress("preflight", status="installing")
            try:
                rc = subprocess.run(cmd.split(), timeout=600).returncode
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                return _fail("preflight", f"installer failed: {e}", 4)
            if rc != 0:
                return _fail("preflight", f"installer exit {rc}", 4)
            missing = plat.check_system_deps()
    if not missing:
        _progress("preflight", status="ok")

    # ── System capability gate ──────────────────────────────────────────
    # Failures here don't abort immediately: setup still writes a valid config
    # (so a later fix — plugging in a mic, installing IBus — needs no re-setup)
    # and then exits with the documented code. Heavy/irreversible steps (model
    # download, service install) are skipped while the system is unusable.
    checks = _check_system()
    system_error: tuple[str, int] | None = None  # (reason, exit code)
    if not checks["audio"]:
        _progress("system", check="audio", status="missing",
                  reason=repr("no microphone/audio input device found"))
        system_error = ("no microphone/audio input device found", 5)
    if checks["is_windows"] or checks["is_mac"]:
        if not checks["pynput"]:
            _progress("system", check="typer", status="missing",
                      reason=repr("no text injection backend (pip install pynput)"))
            system_error = system_error or ("no text injection backend (pip install pynput)", 6)
    else:
        has_typer = (
            (checks["ibus"] and checks["ibus_gi"])
            or checks["xdotool"] or checks["ydotool"] or checks["wtype"]
        )
        if not has_typer:
            reason = f"no text injection backend; install: {plat.system_deps_install_cmd(['ibus'])}"
            _progress("system", check="typer", status="missing", reason=repr(reason))
            system_error = system_error or (reason, 6)

    # ── Backend selection (mirrors interactive path) ────────────────────
    backend = answers.get("backend")
    if not backend:
        if checks["is_windows"] or checks["is_mac"]:
            backend = "auto"
        elif checks["evdev_access"]:
            backend = "evdev"
        elif checks["pynput"] and checks["display"] != "wayland":
            backend = "auto"
        else:
            backend = "socket"
    _progress("backend", value=backend, display=checks["display"])

    # ── Model download ──────────────────────────────────────────────────
    if system_error is not None:
        _progress("model", name=model, status="skipped", reason="'system unusable'")
    elif answers.get("download_model", True):
        _progress("model", name=model, status="downloading")
        if not _download_model(model):
            return _fail("model", f"could not download model '{model}'", 7)
        _progress("model", name=model, status="ready")
    else:
        _progress("model", name=model, status="ready")

    from voiceio.tray import probe_availability
    tray_ok, _, _ = probe_availability()
    if "tray_enabled" in answers:
        tray_ok = bool(answers["tray_enabled"])

    cloud_key = str(answers.get("cloud_api_key", ""))
    if cloud_key:
        _record_cloud_consent()

    # ── Write config (merge-preserving) ─────────────────────────────────
    _write_config(
        model=model, language=language, hotkey=hotkey, method="auto",
        streaming=bool(answers.get("streaming", True)), backend=backend,
        sound_enabled=bool(answers.get("sound_enabled", True)),
        notify_clipboard=bool(answers.get("notify_clipboard", False)),
        tray_enabled=tray_ok,
        commands_enabled=bool(answers.get("commands_enabled", True)),
        punctuation_cleanup=bool(answers.get("punctuation_cleanup", True)),
        number_conversion=bool(answers.get("number_conversion", True)),
        voice_input_prefix=str(answers.get("voice_input_prefix", "")),
        llm_enabled=bool(answers.get("llm_enabled", False)),
        llm_model=str(answers.get("llm_model", "")),
        cloud_api_key=cloud_key,
        cloud_base_url=str(answers.get("cloud_base_url", "")),
        cloud_model=str(answers.get("cloud_model", "")),
        tts_enabled=bool(answers.get("tts_enabled", True)),
        tts_engine=str(answers.get("tts_engine", "auto")),
        tts_speed=float(answers.get("tts_speed", 1.0)),
        quiet=True,
    )
    _progress("config", status="written", path=str(CONFIG_PATH))

    # ── IBus engine component (best-effort) ─────────────────────────────
    if checks["is_linux"] and checks["ibus"] and checks["ibus_gi"]:
        try:
            from voiceio.typers.ibus import _ensure_gnome_input_source, install_component
            if install_component():
                _ensure_gnome_input_source()
                _progress("ibus", status="installed")
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            _progress("ibus", status="skipped", reason=repr(str(e)))

    # ── GNOME shortcut for socket backend ───────────────────────────────
    if checks["is_linux"] and backend == "socket":
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
        if "GNOME" in desktop:
            ok = _setup_gnome_shortcut(hotkey)
            _progress("shortcut", status="ok" if ok else "manual", hotkey=hotkey)
        else:
            _progress("shortcut", status="manual", hint="bind 'voiceio toggle' in your WM")

    # ── Autostart service ───────────────────────────────────────────────
    autostart = False
    if answers.get("install_service", True) and system_error is None:
        from voiceio.service import has_systemd, install_service
        if checks["is_windows"]:
            autostart = install_service()
        elif checks["is_linux"] and has_systemd():
            autostart = install_service()
        _progress("service", status="installed" if autostart else "skipped")
        if autostart and checks["is_linux"]:
            try:
                subprocess.run(["systemctl", "--user", "restart", "voiceio.service"],
                               capture_output=True, timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    if system_error is not None:
        reason, code = system_error
        return _fail("system", reason, code)

    _progress("done", status="ok", hotkey=hotkey, config=str(CONFIG_PATH))
    return 0
