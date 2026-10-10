"""The interactive setup wizard: one function per step over a shared SetupState."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from voiceio.config import CONFIG_DIR, CONFIG_PATH
from voiceio.setup import speech
from voiceio.setup.configfile import _record_cloud_consent, _write_config
from voiceio.setup.extras import _run_advanced_options, _setup_ollama, _tts_demo
from voiceio.setup.hotkey import _setup_gnome_shortcut, _test_hotkey, _wm_bind_lines
from voiceio.setup.speech import LANGUAGES, MODELS, _download_model, _get_or_load_model, _streaming_test
from voiceio.setup.system import _check_system, _preflight_deps, _wait_for_service_ready
from voiceio.setup.ui import (
    LOGO, Spinner, _ask, _ask_choice, _ask_yn, _current_note, _index_of,
    _print_check, _print_step, _satisfaction_loop,
)
from voiceio.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW

if TYPE_CHECKING:
    from voiceio.config import Config

_TOTAL_STEPS = 8

_HOTKEY_OPTIONS = [
    ("ctrl+alt+v", "Ctrl + Alt + V (recommended)"),
    ("alt+v", "Alt + V"),
    ("ctrl+shift+v", "Ctrl + Shift + V"),
    ("super+v", "Super + V (may not work on Wayland/GNOME)"),
    ("Custom",),
]


@dataclass
class SetupState:
    """Everything the wizard steps hand to each other, plus the final answers."""

    checks: dict = field(default_factory=dict)
    cur: Config | None = None  # existing config when re-running (supplies defaults)
    tray_ok: bool = False
    language: str = ""
    model_name: str = ""
    hotkey: str = ""
    method: str = "auto"
    backend: str = "auto"
    desktop: str = ""
    autostart: bool = False
    # Optional settings (defaults unless a step changes them)
    sound_enabled: bool = True
    notify_clipboard: bool = False
    commands_enabled: bool = True
    punctuation_cleanup: bool = True
    number_conversion: bool = True
    voice_input_prefix: str = ""
    llm_enabled: bool = False
    llm_model: str = ""
    autocorrect_api_key: str = ""
    autocorrect_base_url: str = ""
    autocorrect_model: str = ""
    tts_enabled: bool = True
    tts_engine: str = "auto"
    tts_speed: float = 1.0

    @property
    def is_rerun(self) -> bool:
        return self.cur is not None

    @property
    def is_win_or_mac(self) -> bool:
        """Windows or macOS (pynput path) rather than Linux."""
        return self.checks["is_windows"] or self.checks["is_mac"]

    def config_kwargs(self) -> dict:
        return dict(
            model=self.model_name, language=self.language, hotkey=self.hotkey,
            method=self.method, streaming=True, backend=self.backend,
            sound_enabled=self.sound_enabled, notify_clipboard=self.notify_clipboard,
            tray_enabled=self.tray_ok,
            commands_enabled=self.commands_enabled,
            punctuation_cleanup=self.punctuation_cleanup,
            number_conversion=self.number_conversion,
            voice_input_prefix=self.voice_input_prefix,
            llm_enabled=self.llm_enabled, llm_model=self.llm_model,
            autocorrect_api_key=self.autocorrect_api_key,
            autocorrect_base_url=self.autocorrect_base_url,
            autocorrect_model=self.autocorrect_model,
            tts_enabled=self.tts_enabled, tts_engine=self.tts_engine,
            tts_speed=self.tts_speed,
        )


def run_wizard() -> None:
    state = _welcome()
    _preflight_deps(interactive=True)
    _step_system_check(state)    # 1
    _step_language(state)        # 2
    _step_model(state)           # 3
    _step_hotkey(state)          # 4
    _step_tts(state)             # 5
    _step_llm(state)             # 6
    _step_advanced(state)        # 7
    _step_finalize(state)        # 8
    _print_ready_panel(state.hotkey, autostart=state.autostart)


def _welcome() -> SetupState:
    print(LOGO)
    is_rerun = CONFIG_PATH.exists()
    from voiceio.config import load as _load_cfg
    state = SetupState(cur=_load_cfg() if is_rerun else None)
    if is_rerun:
        print(f"  {DIM}Re-running setup — current values are shown as defaults.{RESET}")
    return state


# ── Step 1: System check ─────────────────────────────────────────────────

def _step_system_check(state: SetupState) -> None:
    _print_step(1, _TOTAL_STEPS, "System check")
    state.checks = checks = _check_system()

    if state.is_win_or_mac:
        _plat = "Windows" if checks["is_windows"] else "macOS"
        print(f"\n  {YELLOW}⚠  {_plat} is experimental and untested.{RESET}")
        print(f"  {DIM}voiceio is developed on Linux; this platform may be broken. "
              f"Proceed at your own risk.{RESET}\n")

    _print_capability_checks(checks)
    state.tray_ok = _print_tray_check()
    _ensure_cli_commands()
    _require_mic_and_typer(state)


def _print_capability_checks(checks: dict) -> None:
    _print_check("Platform", True, checks["display"])
    _print_check("Audio input", checks["audio"],
                 f"{len(checks['audio_devices'])} device(s)" if checks["audio"] else "no devices found")

    if checks["is_windows"] or checks["is_mac"]:
        _print_check("pynput", checks["pynput"],
                     "hotkeys + text injection" if checks["pynput"] else "pip install pynput")
    else:
        from voiceio.platform import pkg_install
        if checks["ibus"] and checks["ibus_gi"]:
            _print_check("IBus", True, "recommended, atomic text insertion")
        elif checks["ibus"]:
            _print_check("IBus", False, f"install bindings: {pkg_install('gir1.2-ibus-1.0')}")
        else:
            _print_check("IBus", False, f"install: {pkg_install('ibus', 'gir1.2-ibus-1.0')}")

        if checks["display"] == "wayland":
            _print_check("ydotool", checks["ydotool"],
                         "fallback" if checks["ydotool"] else f"optional: {pkg_install('ydotool')}",
                         optional=True)
            _print_check("wtype", checks["wtype"],
                         "fallback" if checks["wtype"] else f"optional: {pkg_install('wtype')}",
                         optional=True)
        else:
            _print_check("xdotool", checks["xdotool"],
                         "fallback" if checks["xdotool"] else f"optional: {pkg_install('xdotool')}",
                         optional=True)

    _print_check("CUDA GPU", checks["cuda"],
                 "will use GPU" if checks["cuda"] else "will use CPU (still fast)",
                 optional=True)

    if checks["is_linux"]:
        from voiceio.hotkeys.evdev import access_hint as evdev_access_hint
        _print_check("Keyboard access (evdev)", checks["evdev_access"],
                     "evdev hotkeys available" if checks["evdev_access"]
                     else evdev_access_hint(),
                     optional=checks["evdev_access"])


def _print_tray_check() -> bool:
    from voiceio.tray import probe_availability
    tray_ok, tray_reason, tray_fix_hint = probe_availability()
    if tray_ok:
        _print_check("Tray icon", True, "system tray indicator available")
    else:
        _print_check("Tray icon", False,
                     tray_fix_hint if tray_fix_hint else tray_reason,
                     optional=True)
    return tray_ok


def _ensure_cli_commands() -> None:
    """Install CLI symlinks to ~/.local/bin/ unless already on PATH."""
    from voiceio.service import install_symlinks, symlinks_installed, path_hint_needed, _is_pipx_install
    if _is_pipx_install():
        _print_check("CLI commands", True, "installed via pipx (already on PATH)")
    elif not symlinks_installed():
        linked = install_symlinks()
        if linked:
            _print_check("CLI commands", True, f"linked {len(linked)} commands to ~/.local/bin/")
            if path_hint_needed():
                print(f"  {YELLOW}ℹ{RESET}  {DIM}Restart your terminal so 'voiceio' is on PATH{RESET}")
            else:
                print(f"  {DIM}  You can now run 'voiceio' from any terminal{RESET}")
        else:
            _print_check("CLI commands", False, "could not create symlinks in ~/.local/bin/")
    else:
        _print_check("CLI commands", True, "voiceio in PATH")


def _require_mic_and_typer(state: SetupState) -> None:
    """Exit unless there is a microphone and some text injection backend."""
    checks = state.checks
    if not checks["audio"]:
        print(f"\n  {RED}No microphone found. Connect one and try again.{RESET}")
        sys.exit(1)

    if state.is_win_or_mac:
        has_typer = checks["pynput"]
        if not has_typer:
            print(f"\n  {RED}No text injection backend available.{RESET}")
            print(f"  {DIM}Install pynput: pip install pynput{RESET}")
            sys.exit(1)
    else:
        has_typer = checks["ibus"] and checks["ibus_gi"]
        has_typer = has_typer or checks["xdotool"] or checks["ydotool"] or checks["wtype"]
        if not has_typer:
            from voiceio.platform import pkg_install
            print(f"\n  {RED}No text injection backend available.{RESET}")
            print(f"  {DIM}Install one: {pkg_install('ibus', 'gir1.2-ibus-1.0')}{RESET}")
            sys.exit(1)


# ── Step 2: Language ─────────────────────────────────────────────────────

def _step_language(state: SetupState) -> None:
    cur = state.cur
    _print_step(2, _TOTAL_STEPS, "Language")
    print(f"  {DIM}Pick your primary language, or auto-detect.{RESET}")
    _current_note(state.is_rerun, cur.model.language if cur else "")
    print()
    lang_default = _index_of(LANGUAGES, cur.model.language, 0) if cur else 0
    lang_idx = _ask_choice(LANGUAGES, default=lang_default)
    state.language = LANGUAGES[lang_idx][0]


# ── Step 3: Speech recognition model ─────────────────────────────────────

def _step_model(state: SetupState) -> None:
    cur = state.cur
    _print_step(3, _TOTAL_STEPS, "Speech recognition model")
    print(f"  {DIM}Larger models are more accurate but slower and use more RAM.{RESET}")
    _current_note(state.is_rerun, cur.model.name if cur else "")
    print()
    model_default = _index_of(MODELS, cur.model.name, 2) if cur else 2
    model_idx = _ask_choice(MODELS, default=model_default)
    state.model_name = MODELS[model_idx][0]

    # Download immediately
    if not _download_model(state.model_name):
        sys.exit(1)

    # Quick mic test + satisfaction loop to try different models
    if _ask_yn("Test it with your microphone?"):
        def _stt_demo_fn(idx: int) -> bool:
            name = MODELS[idx][0]
            # Already-downloaded models are cached; only download new ones
            if speech._cached_model_name != name:
                if not _download_model(name):
                    return False
            result = _streaming_test(model=_get_or_load_model(name), language=state.language)
            return bool(result)

        selected = _satisfaction_loop(
            options=MODELS,
            demo_fn=_stt_demo_fn,
            prompt="Happy with the transcription quality?",
            default=model_idx,
        )
        if selected is not None:
            state.model_name = MODELS[selected][0]


# ── Step 4: Keyboard shortcut + output method ────────────────────────────

def _step_hotkey(state: SetupState) -> None:
    cur = state.cur
    _print_step(4, _TOTAL_STEPS, "Keyboard shortcut")
    print(f"  {DIM}This combo toggles recording on/off.{RESET}")
    _current_note(state.is_rerun, cur.hotkey.key if cur else "")
    print()
    hk_default = _index_of(_HOTKEY_OPTIONS, cur.hotkey.key, 0) if cur else 0
    state.hotkey = _choose_hotkey(default=hk_default, custom_default="super+v")
    _select_output_backend(state)


def _choose_hotkey(default: int, custom_default: str) -> str:
    hk_idx = _ask_choice(_HOTKEY_OPTIONS, default=default)
    if hk_idx == len(_HOTKEY_OPTIONS) - 1:
        return _ask("Enter combo (e.g. ctrl+shift+r)", custom_default)
    return _HOTKEY_OPTIONS[hk_idx][0]


def _select_output_backend(state: SetupState) -> None:
    """Output method is always auto; pick the hotkey backend and set up IBus."""
    checks = state.checks
    state.method = "auto"
    if state.is_win_or_mac:
        state.backend = "auto"
        label = "pynput" if checks["is_windows"] else "pynput (requires Accessibility permission)"
        print(f"\n  {GREEN}✓{RESET} {DIM}Text injection: {label}{RESET}")
        return

    if checks["ibus"] and checks["ibus_gi"]:
        print(f"\n  {GREEN}✓{RESET} {DIM}Text injection: IBus (best quality, auto-selected){RESET}")
        from voiceio.typers.ibus import install_component, _ensure_gnome_input_source
        if install_component():
            print(f"  {GREEN}✓{RESET} {DIM}IBus engine component installed{RESET}")
            _ensure_gnome_input_source()
            print(f"  {GREEN}✓{RESET} {DIM}Added VoiceIO to GNOME input sources{RESET}")
        else:
            print(f"  {YELLOW}⚠{RESET}  {DIM}Could not install IBus component, will use fallback{RESET}")

    if checks["evdev_access"]:
        state.backend = "evdev"
    elif checks["pynput"] and checks["display"] != "wayland":
        state.backend = "auto"  # pynput works on X11
    else:
        state.backend = "socket"


# ── Step 5: Text-to-speech ───────────────────────────────────────────────

def _step_tts(state: SetupState) -> None:
    _print_step(5, _TOTAL_STEPS, "Text-to-speech")
    print(f"  {DIM}Select text and press ctrl+alt+s to hear it spoken.{RESET}")
    print(f"  {DIM}Engines: piper (offline), edge-tts (cloud), espeak-ng (system).{RESET}\n")
    tts_top_options = [
        ("Enabled", "auto-select best available engine"),
        ("Choose engine", "hear a demo and pick"),
        ("Disabled",),
    ]
    tts_choice = _ask_choice(tts_top_options, default=0)
    if tts_choice == 2:
        state.tts_enabled = False
    elif tts_choice == 1:
        _choose_tts_engine(state)


def _choose_tts_engine(state: SetupState) -> None:
    # Let user pick an engine first, then demo it
    engine_options = [
        ("piper", "offline, best quality"),
        ("edge-tts", "cloud, free Microsoft TTS"),
        ("espeak", "system package, lightweight"),
    ]
    print(f"\n  {DIM}Pick an engine to hear a demo:{RESET}\n")
    first_pick = _ask_choice(engine_options, default=0)

    selected = _satisfaction_loop(
        options=engine_options,
        demo_fn=lambda idx: _tts_demo(engine_options[idx][0]),
        prompt="Happy with this voice?",
        default=first_pick,
    )
    state.tts_engine = engine_options[selected][0] if selected is not None else "auto"

    print(f"\n  {DIM}Speech speed:{RESET}\n")
    speed_options = [
        ("Normal", "1.0x"),
        ("Slow", "0.8x"),
        ("Fast", "1.3x"),
    ]
    sp_idx = _ask_choice(speed_options, default=0)
    state.tts_speed = [1.0, 0.8, 1.3][sp_idx]


# ── Step 6: LLM & autocorrect ────────────────────────────────────────────

def _step_llm(state: SetupState) -> None:
    _print_step(6, _TOTAL_STEPS, "LLM & autocorrect")
    print(f"  {DIM}Use a local LLM to fix grammar and spelling in your dictation.{RESET}")
    print(f"  {DIM}Runs only on the final pass — streaming preview is unaffected.{RESET}")
    _choose_local_llm(state)
    _choose_cloud_autocorrect(state)


def _choose_local_llm(state: SetupState) -> None:
    from voiceio.config import LLMConfig
    from voiceio.llm import OllamaStatus, diagnose_ollama, _has_gpu
    _llm_status, _llm_models = diagnose_ollama(LLMConfig(enabled=True))
    has_gpu = _has_gpu()

    if not has_gpu:
        print(f"  {YELLOW}⚠{RESET}  {DIM}No GPU detected. LLM adds 5-15s latency on CPU — not recommended.{RESET}\n")

    if _llm_status == OllamaStatus.OK:
        print(f"  {GREEN}✓{RESET} {DIM}Ollama detected with {len(_llm_models)} model(s){RESET}\n")
        if has_gpu:
            print(f"  {DIM}Adds ~0.5-2s latency per dictation with GPU.{RESET}\n")
            llm_options = [
                ("Enabled", "use your existing Ollama install"),
                ("Disabled", "skip LLM"),
            ]
        else:
            llm_options = [
                ("Disabled", "recommended without GPU"),
                ("Enabled", "adds 5-15s latency on CPU"),
            ]
    elif _llm_status == OllamaStatus.NOT_INSTALLED:
        if has_gpu:
            size_note = "~800 MB Ollama + ~400 MB model"
            print(f"  {DIM}Adds ~0.5-2s latency per dictation with GPU.{RESET}\n")
            llm_options = [
                ("Install Ollama + model", f"downloads {size_note}"),
                ("Skip for now", "you can enable later with 'voiceio setup'"),
            ]
        else:
            size_note = "~60 MB Ollama + ~400 MB model"
            print(f"  {DIM}Requires Ollama ({size_note}).{RESET}\n")
            llm_options = [
                ("Skip for now", "recommended without GPU"),
                ("Install anyway", f"downloads {size_note}, adds 5-15s latency"),
            ]
    else:
        # Installed but not running, or missing model
        print(f"  {DIM}Ollama is installed. Just needs a model (~400 MB).{RESET}\n")
        if has_gpu:
            llm_options = [
                ("Enabled", "pull a small model"),
                ("Disabled", "skip LLM"),
            ]
        else:
            llm_options = [
                ("Disabled", "recommended without GPU"),
                ("Enabled", "pull a model, adds 5-15s latency on CPU"),
            ]

    llm_choice = _ask_choice(llm_options, default=0)
    # With GPU: enable option is always first (index 0)
    # Without GPU: enable option is always second (index 1)
    if llm_choice == (0 if has_gpu else 1):
        state.llm_model = _setup_ollama(_llm_status, _llm_models)
        state.llm_enabled = bool(state.llm_model)
    else:
        state.llm_enabled = False
        state.llm_model = ""


def _choose_cloud_autocorrect(state: SetupState) -> None:
    print(f"\n  {BOLD}Autocorrect{RESET}")
    print(f"  {DIM}The optional cloud pass (postcorrect) uses an LLM to fix misheard words at the end of each dictation.{RESET}\n")
    ac_options = [
        ("Skip", "no cloud API key"),
        ("Enable", "paste an API key (OpenRouter, OpenAI, or Anthropic)"),
    ]
    if _ask_choice(ac_options, default=0) != 1:
        return
    api_input = _ask("API key", "")
    if not api_input:
        return

    from voiceio.config import AutocorrectConfig
    from voiceio.llm_api import check_api_key, detect_provider
    det_url, det_model = detect_provider(api_input)
    provider_name = "OpenRouter"
    if "openai.com" in det_url:
        provider_name = "OpenAI"
    elif "anthropic.com" in det_url:
        provider_name = "Anthropic"
    print(f"  {DIM}Detected: {provider_name}{RESET}")
    cfg_check = AutocorrectConfig(
        api_key=api_input, base_url=det_url, model=det_model,
    )
    if check_api_key(cfg_check):
        print(f"  {GREEN}✓{RESET} API key validated ({provider_name})")
        state.autocorrect_api_key = api_input
        state.autocorrect_base_url = det_url
        state.autocorrect_model = det_model
        _record_cloud_consent()
    else:
        print(f"  {YELLOW}⚠{RESET}  Validation failed. You can set it later in config.toml.")


# ── Step 7: Advanced options (optional) ──────────────────────────────────

def _step_advanced(state: SetupState) -> None:
    _print_step(7, _TOTAL_STEPS, "Advanced options")
    print(f"  {DIM}Sensible defaults are already set. Customize if you want.{RESET}\n")
    advanced_options = [
        ("Skip", "use defaults (recommended)"),
        ("Customize", "feedback, voice commands, punctuation, numbers"),
    ]
    if _ask_choice(advanced_options, default=0) == 1:
        for key, value in _run_advanced_options(state.checks, state.tray_ok).items():
            setattr(state, key, value)


# ── Step 8: Finalize ─────────────────────────────────────────────────────

def _step_finalize(state: SetupState) -> None:
    _print_step(8, _TOTAL_STEPS, "Finalize")
    _write_config(**state.config_kwargs())
    _setup_desktop_shortcut(state)
    _install_autostart(state)
    _verify_hotkey(state)
    if state.autostart:
        _start_service()


def _setup_desktop_shortcut(state: SetupState) -> None:
    """Set up DE shortcut if on GNOME + socket backend (Linux only)."""
    if not state.checks["is_linux"]:
        return
    hotkey, backend = state.hotkey, state.backend
    state.desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    if "GNOME" in state.desktop and backend == "socket":
        print(f"\n  {CYAN}Setting up GNOME keyboard shortcut...{RESET}")
        if _setup_gnome_shortcut(hotkey):
            print(f"  {GREEN}✓{RESET} Shortcut {BOLD}{hotkey}{RESET} → voiceio toggle configured!")
        else:
            print(f"  {YELLOW}⚠{RESET}  Auto-setup failed. Add manually in Settings → Keyboard → Shortcuts:")
            print("    Command: voiceio toggle")
    elif backend == "socket":
        print(f"\n  {YELLOW}ℹ{RESET}  Add a keyboard shortcut in your window manager config:")
        for wm, line in _wm_bind_lines(hotkey):
            print(f"    {DIM}{wm + ':':<10}{RESET} {line}")
        print(f"    {DIM}{'other:':<10}{RESET} bind {BOLD}{hotkey}{RESET} → {BOLD}voiceio toggle{RESET}")
        from voiceio.hotkeys.evdev import access_hint as evdev_access_hint
        print(f"\n    {DIM}Or enable evdev (no WM config needed):{RESET}")
        print(f"    {evdev_access_hint()}")


def _install_autostart(state: SetupState) -> None:
    from voiceio.service import install_service

    if state.checks["is_windows"]:
        with Spinner("Setting up Windows Startup...") as sp:
            if install_service():
                sp.ok("Added to Windows Startup")
                state.autostart = True
            else:
                sp.warn("Could not add to Startup folder")
    elif state.checks["is_mac"]:
        pass  # macOS autostart not yet supported
    else:
        from voiceio.service import has_systemd
        if has_systemd():
            with Spinner("Installing systemd service...") as sp:
                if install_service():
                    sp.ok("Systemd service installed and enabled")
                    state.autostart = True
                else:
                    sp.warn("Could not install service")


def _verify_hotkey(state: SetupState) -> None:
    """Quick hotkey test; on failure, offer one retry with a different combo."""
    print(f"\n  {DIM}Testing hotkey...{RESET}")
    if _test_hotkey(state.hotkey, state.backend):
        return

    gnome_socket = "GNOME" in state.desktop and state.backend == "socket"
    print(f"\n  {YELLOW}Troubleshooting:{RESET}")
    if gnome_socket:
        print(f"  {DIM}• The GNOME shortcut may need a moment to register.{RESET}")
        print(f"  {DIM}• Try: Settings → Keyboard → Custom Shortcuts to verify.{RESET}")
        print(f"  {DIM}• Shortcut command should be: {BOLD}voiceio toggle{RESET}")
    if _ask_yn("Try a different shortcut?"):
        print()
        state.hotkey = _choose_hotkey(default=1, custom_default="ctrl+shift+v")
        _write_config(**state.config_kwargs())
        if gnome_socket:
            _setup_gnome_shortcut(state.hotkey)
        _test_hotkey(state.hotkey, state.backend)


def _start_service() -> None:
    """Start (or restart) the service so it's immediately usable."""
    from voiceio.hotkeys.socket_backend import SOCKET_PATH
    from voiceio.service import is_running
    action = "restart" if is_running() else "start"
    SOCKET_PATH.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["systemctl", "--user", action, "voiceio.service"],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    with Spinner("Waiting for service to start...") as sp:
        if _wait_for_service_ready(timeout=30):
            sp.ok("Service is running!")
        else:
            sp.warn("Timed out — service may still be loading. Check: voiceio service logs")


# ── Done ─────────────────────────────────────────────────────────────────

_FIRST_SENTENCE = "Hello world, this is my first dictation with voiceio."


def _print_ready_panel(hotkey: str, autostart: bool) -> None:
    """The 'you're ready' panel shown after a successful setup."""
    from voiceio.config import LOG_DIR, LOG_PATH
    from voiceio.service import _is_pipx_install

    start_hint = (
        "  voiceio is running and will start automatically on login."
        if autostart
        else f"  Start voiceio:\n    {CYAN}voiceio run{RESET}"
    )
    path_note = ""
    if not _is_pipx_install() and not shutil.which("voiceio"):
        path_note = f"  {YELLOW}ℹ{RESET}  Restart your terminal so 'voiceio' is on PATH.\n\n"

    print(f"""
{GREEN}{'━' * 54}{RESET}
{BOLD}  You're ready.{RESET}  {DIM}Speak → text, locally and instantly.{RESET}

{path_note}{start_hint}

  {BOLD}Try it now:{RESET} press {BOLD}{hotkey}{RESET}, then say
    {CYAN}"{_FIRST_SENTENCE}"{RESET}
  {DIM}Press {hotkey} again to commit. Text streams at your cursor.{RESET}

  {DIM}Want the full tour?{RESET}  {CYAN}voiceio demo{RESET}
  {DIM}Other commands:{RESET}     {CYAN}voiceio doctor{RESET} · {CYAN}test{RESET} · {CYAN}correct{RESET} · {CYAN}history{RESET}

  {DIM}Your data lives in {CONFIG_DIR} and {LOG_DIR} —
  plain local files you own. Zero telemetry.{RESET}
  {DIM}Config: {CONFIG_PATH}   Logs: {LOG_PATH}{RESET}
{GREEN}{'━' * 54}{RESET}
""")
