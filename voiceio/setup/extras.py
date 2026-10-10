"""Optional wizard extras: Ollama setup, TTS engine demo, advanced options."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from voiceio.setup.system import _install_pip_package
from voiceio.setup.ui import Spinner, _ask, _ask_choice, _ask_yn
from voiceio.ui import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW

log = logging.getLogger(__name__)


# ── Ollama setup ─────────────────────────────────────────────────────────

_LLM_RECOMMENDED_MODELS = [
    ("qwen2.5:0.5b", "~400 MB", "fastest, recommended for CPU"),
    ("llama3.2:1b", "~700 MB", "fast, good quality"),
    ("phi3:mini", "~2.3 GB", "balanced speed/quality"),
    ("mistral:7b", "~4.1 GB", "best quality, GPU recommended"),
]


def _setup_ollama(status=None, models=None) -> str:
    """Guide user through Ollama installation, daemon start, and model pull.

    Accepts cached status/models from the caller to avoid redundant diagnosis.
    Returns the selected model name.
    """
    from voiceio.config import LLMConfig
    from voiceio.llm import (
        OllamaStatus, diagnose_ollama, install_ollama, start_ollama, pull_model,
    )

    cfg = LLMConfig(enabled=True)
    if status is None:
        status, models = diagnose_ollama(cfg)
    if models is None:
        models = []

    # Step 1: Install if missing
    if status == OllamaStatus.NOT_INSTALLED:
        print(f"\n  {YELLOW}⚠{RESET}  Ollama is not installed.")
        if sys.platform == "linux":
            if _ask_yn("Install Ollama now?"):
                from voiceio.llm import _has_gpu
                if _has_gpu():
                    print(f"\n  {CYAN}Installing Ollama (with GPU support)...{RESET}\n")
                else:
                    print(f"\n  {CYAN}Installing Ollama (CPU-only, ~60 MB)...{RESET}\n")
                if install_ollama():
                    print(f"\n  {GREEN}✓{RESET} Ollama installed!")
                    status, models = diagnose_ollama(cfg)
                else:
                    print(f"\n  {RED}✗{RESET} Installation failed. Visit https://ollama.com")
                    return ""
            else:
                print(f"  {DIM}Install later from https://ollama.com{RESET}")
                return ""
        else:
            print(f"  {DIM}Install from https://ollama.com and re-run setup.{RESET}")
            return ""

    # Step 2: Start daemon if not running
    if status == OllamaStatus.NOT_RUNNING:
        print(f"\n  {YELLOW}⚠{RESET}  Ollama is installed but not responding.")
        if _ask_yn("Start Ollama now?"):
            with Spinner("Starting Ollama...") as sp:
                if start_ollama():
                    sp.ok("Ollama is running!")
                    status, models = diagnose_ollama(cfg)
                else:
                    sp.fail("Could not start Ollama. Try 'ollama serve' manually.")
                    return ""

    # Step 3: Pick from installed models or pull a new one
    if models:
        print(f"\n  {GREEN}✓{RESET} Ollama is running with {len(models)} model(s)")
        print(f"\n  {DIM}Select a model for grammar/spelling cleanup:{RESET}\n")
        model_options = [(m,) for m in models[:6]]
        model_options.append(("Pull a new model...",))
        idx = _ask_choice(model_options, default=0)
        if idx < len(models[:6]):
            selected = models[idx]
            print(f"  {GREEN}✓{RESET} Using {BOLD}{selected}{RESET}")
            return selected
        # Fall through to pull flow

    if not models:
        print(f"\n  {YELLOW}⚠{RESET}  No models installed yet. Let's pull one.")

    # Step 4: Pull a recommended model
    print(f"\n  {DIM}Choose a model to download:{RESET}\n")
    idx = _ask_choice(_LLM_RECOMMENDED_MODELS, default=0)
    model_name = _LLM_RECOMMENDED_MODELS[idx][0]

    print(f"\n  {CYAN}Pulling {model_name}...{RESET}")
    print(f"  {DIM}This downloads the model. May take a few minutes.{RESET}\n")
    if pull_model(model_name):
        print(f"\n  {GREEN}✓{RESET} Model {BOLD}{model_name}{RESET} ready!")
    else:
        print(f"\n  {RED}✗{RESET} Pull failed. You can pull manually later: ollama pull {model_name}")
    return model_name


# ── Advanced options (shown only when user opts in) ──────────────────────

_TTS_DEMO_PHRASE = "Hello! This is a preview of the voiceio text to speech engine."

_TTS_PIP_PACKAGES = {
    "piper": ["piper-tts"],
    "edge-tts": ["edge-tts", "soundfile"],
}


def _tts_demo(engine_name: str) -> bool:
    """Install engine if needed, synthesize a phrase, play it. Returns True on success."""
    import importlib

    from voiceio.config import TTSConfig
    from voiceio.tts.chain import _create

    cfg = TTSConfig(enabled=True, engine=engine_name)

    # Install if needed
    pip_pkgs = _TTS_PIP_PACKAGES.get(engine_name)
    if pip_pkgs:
        try:
            engine = _create(engine_name, cfg)
            probe = engine.probe()
        except Exception as exc:
            log.debug("TTS probe failed for %s: %s", engine_name, exc)
            probe = None

        if not probe or not probe.ok:
            for pkg in pip_pkgs:
                if not _install_pip_package(pkg):
                    print(f"  {YELLOW}⚠{RESET}  Skipped — {pkg} not installed.")
                    return False
            importlib.invalidate_caches()
            engine = _create(engine_name, cfg)
            probe = engine.probe()
            if not probe.ok:
                print(f"  {RED}✗{RESET} {engine_name} still unavailable after install: {probe.reason}")
                return False
    elif engine_name == "espeak":
        if not shutil.which("espeak-ng"):
            from voiceio.platform import pkg_install
            install_cmd = pkg_install("espeak-ng")
            print(f"  {DIM}espeak-ng is a lightweight system package (~2 MB).{RESET}")
            if not _ask_yn(f"Install espeak-ng? ({install_cmd})"):
                return False
            parts = install_cmd.split()
            try:
                result = subprocess.run(parts, timeout=120)
                if result.returncode != 0:
                    print(f"  {RED}✗{RESET} Installation failed")
                    return False
                print(f"  {GREEN}✓{RESET} espeak-ng installed!")
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"  {RED}✗{RESET} Installation failed: {e}")
                return False
        engine = _create(engine_name, cfg)
    else:
        engine = _create(engine_name, cfg)

    # Synthesize and play
    try:
        with Spinner(f"Synthesizing with {engine_name}...") as sp:
            audio, sr = engine.synthesize(_TTS_DEMO_PHRASE, voice="", speed=1.0)
            sp.ok(f"Playing {engine_name} demo")

        from voiceio.tts.player import TTSPlayer
        player = TTSPlayer()
        player.play(audio, sr)
        engine.shutdown()
        return True
    except Exception as e:
        print(f"  {RED}✗{RESET} Demo failed: {e}")
        return False


def _run_advanced_options(checks: dict, tray_ok: bool) -> dict:
    """Interactive advanced options. Returns dict of settings."""
    result: dict = {}

    # Feedback
    print(f"\n  {BOLD}Feedback{RESET}")
    print(f"  {DIM}Sound plays when recording starts/stops. Notifications show transcribed text.{RESET}\n")
    feedback_options = [
        ("Sound only", "short chime on start/stop (default)"),
        ("Sound + notification", "also shows a desktop notification"),
        ("None", "silent"),
    ]
    fb_idx = _ask_choice(feedback_options, default=0)
    result["sound_enabled"] = fb_idx in (0, 1)
    result["notify_clipboard"] = fb_idx == 1

    # Voice commands
    print(f"\n  {BOLD}Voice commands{RESET}")
    print(f"  {DIM}Recognize spoken commands: \"new line\", \"period\", \"scratch that\", etc.{RESET}\n")
    cmd_options = [
        ("Enabled", "default"),
        ("Disabled", "pass through raw text"),
    ]
    result["commands_enabled"] = _ask_choice(cmd_options, default=0) == 0

    # Smart punctuation
    print(f"\n  {BOLD}Smart punctuation{RESET}")
    print(f"  {DIM}Auto-capitalize sentences, fix spacing around punctuation.{RESET}\n")
    punct_options = [
        ("Enabled", "default"),
        ("Disabled", "raw Whisper output"),
    ]
    result["punctuation_cleanup"] = _ask_choice(punct_options, default=0) == 0

    # Number conversion
    print(f"\n  {BOLD}Number conversion{RESET}")
    print(f"  {DIM}Convert spoken numbers to digits: \"twenty five\" → \"25\".{RESET}\n")
    num_options = [
        ("Enabled", "default"),
        ("Disabled", "keep number words as text"),
    ]
    result["number_conversion"] = _ask_choice(num_options, default=0) == 0

    # Voice input prefix
    print(f"\n  {BOLD}Voice input prefix{RESET}")
    print(f"  {DIM}Prepend a tag (e.g. \"[voice]\") to every transcription. Helps{RESET}")
    print(f"  {DIM}readers/agents spot Whisper mishearings and infer intent.{RESET}\n")
    prefix_options = [
        ("Off", "default — no prefix"),
        ("[voice]", "tag every transcription with [voice]"),
        ("Custom", "type your own prefix"),
    ]
    prefix_idx = _ask_choice(prefix_options, default=0)
    if prefix_idx == 0:
        result["voice_input_prefix"] = ""
    elif prefix_idx == 1:
        result["voice_input_prefix"] = "[voice]"
    else:
        custom = _ask("Prefix", default="[voice]").strip()
        result["voice_input_prefix"] = custom

    return result
