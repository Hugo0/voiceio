"""Interactive setup wizard for voiceio."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from voiceio.config import CONFIG_DIR, CONFIG_PATH

# ── Colors ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

LOGO = f"""{CYAN}{BOLD}
 ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██╗ ██████╗
 ██║   ██║██╔═══██╗██║██╔════╝██╔════╝██║██╔═══██╗
 ██║   ██║██║   ██║██║██║     █████╗  ██║██║   ██║
 ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██║██║   ██║
  ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██║╚██████╔╝
   ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═╝ ╚═════╝
{RESET}{DIM}  speak → text, locally, instantly{RESET}
"""

MODELS = [
    ("tiny", "75 MB", "Fastest, basic accuracy"),
    ("base", "150 MB", "Fast, good accuracy (recommended)"),
    ("small", "500 MB", "Moderate speed, better accuracy"),
    ("medium", "1.5 GB", "Slower, great accuracy"),
    ("large-v3", "3 GB", "Slowest, best accuracy"),
]

LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("pt", "Portuguese"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("auto", "Auto-detect (slower)"),
]


def _print_step(n: int, total: int, title: str) -> None:
    bar = f"{MAGENTA}[{n}/{total}]{RESET}"
    print(f"\n{bar} {BOLD}{title}{RESET}")
    print(f"{DIM}{'─' * 50}{RESET}")


def _ask(prompt: str, default: str = "") -> str:
    default_hint = f" {DIM}[{default}]{RESET}" if default else ""
    try:
        answer = input(f"  {CYAN}›{RESET} {prompt}{default_hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer or default


def _ask_choice(options: list[tuple[str, ...]], default: int = 0) -> int:
    """Interactive choice with arrow keys, w/s, j/k navigation. Enter to confirm."""
    import atexit
    import re
    import select
    import termios
    import tty

    # Ensure cursor is restored if process crashes/exits unexpectedly
    atexit.register(lambda: sys.stdout.write("\033[?25h"))

    selected = default
    n = len(options)

    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    _ansi_re = re.compile(r"\033\[[0-9;]*m")

    def _visible_len(s: str) -> int:
        return len(_ansi_re.sub("", s))

    def _truncate(s: str, max_width: int) -> str:
        vis = 0
        result = []
        for part in _ansi_re.split(s):
            if _ansi_re.match(part):
                result.append(part)
            else:
                remaining = max_width - vis
                result.append(part[:remaining])
                vis += min(len(part), remaining)
                if vis >= max_width:
                    break
        return "".join(result) + RESET

    def _format_line(i: int) -> str:
        marker = f"{GREEN}●{RESET}" if i == selected else f"{DIM}○{RESET}"
        label = options[i][0]
        detail = f"  {DIM}({', '.join(options[i][1:])}){RESET}" if len(options[i]) > 1 else ""
        line = f"  {marker} {BOLD}{i + 1}{RESET}. {label}{detail}"
        if _visible_len(line) >= cols:
            line = _truncate(line, cols - 2)
        return line

    # Hint text shown below options
    hint = f"  {DIM}\u2191\u2193 navigate, enter to confirm{RESET}"

    def _draw() -> None:
        """Draw menu from current cursor position.

        After this call, cursor is at column 0 of the hint line.
        """
        sys.stdout.write("\033[J")  # clear from cursor to end of screen
        for i in range(n):
            sys.stdout.write(f"{_format_line(i)}\r\n")
        sys.stdout.write(f"{hint}\r")
        sys.stdout.flush()

    def _redraw() -> None:
        """Move cursor from hint line back to first option, then redraw."""
        sys.stdout.write(f"\033[{n}A")
        _draw()

    def _read_key(fd: int) -> str:
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            if select.select([fd], [], [], 0.05)[0]:
                ch2 = os.read(fd, 1)
                if ch2 == b"[" and select.select([fd], [], [], 0.05)[0]:
                    ch3 = os.read(fd, 1)
                    if ch3 == b"A":
                        return "up"
                    if ch3 == b"B":
                        return "down"
            return "esc"
        if ch in (b"\r", b"\n"):
            return "enter"
        if ch == b"\x03":
            return "ctrl-c"
        return ch.decode("utf-8", errors="replace")

    sys.stdout.write("\033[?25l")  # hide cursor
    _draw()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = _read_key(fd)
            if key in ("up", "w", "k"):
                selected = (selected - 1) % n
            elif key in ("down", "s", "j"):
                selected = (selected + 1) % n
            elif key == "enter":
                break
            elif key == "ctrl-c":
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\033[?25h\r\n")
                sys.exit(0)
            elif key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < n:
                    selected = idx
            else:
                continue
            _redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        termios.tcflush(fd, termios.TCIFLUSH)  # flush stale input from raw mode

    # Final redraw showing confirmed selection, then move past menu
    _redraw()
    sys.stdout.write("\033[?25h\r\n\n")  # show cursor, past hint, blank line
    sys.stdout.flush()
    return selected


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _check_system() -> dict:
    """Check system dependencies and capabilities."""
    checks = {}

    # Display server
    session = os.environ.get("XDG_SESSION_TYPE", "unknown")
    checks["display"] = session

    # Typer binaries
    checks["xdotool"] = _check_binary("xdotool")
    checks["xclip"] = _check_binary("xclip")
    checks["ydotool"] = _check_binary("ydotool")
    checks["wtype"] = _check_binary("wtype")
    checks["ibus"] = _check_binary("ibus")

    # IBus Python bindings (check system Python, not venv)
    checks["ibus_gi"] = False
    if checks["ibus"]:
        from voiceio.typers.ibus import _has_ibus_gi
        checks["ibus_gi"] = _has_ibus_gi()

    # Audio
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        checks["audio"] = len(input_devs) > 0
        checks["audio_devices"] = input_devs
    except Exception:
        checks["audio"] = False
        checks["audio_devices"] = []

    # GPU
    try:
        import ctranslate2
        checks["cuda"] = "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        checks["cuda"] = False

    # Input group (for evdev)
    groups = os.getgroups()
    try:
        import grp
        input_gid = grp.getgrnam("input").gr_gid
        checks["input_group"] = input_gid in groups
    except (KeyError, ImportError):
        checks["input_group"] = False

    return checks


def _print_check(label: str, ok: bool, detail: str = "", optional: bool = False) -> None:
    if ok:
        icon = f"{GREEN}✓{RESET}"
    elif optional:
        icon = f"{YELLOW}○{RESET}"
    else:
        icon = f"{RED}✗{RESET}"
    extra = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {icon} {label}{extra}")


_cached_model = None
_cached_model_name: str | None = None


def _get_or_load_model(model_name: str | None = None):
    """Get or load a whisper model. Caches in module global."""
    global _cached_model, _cached_model_name

    if model_name is None:
        from voiceio.config import load
        cfg = load()
        model_name = cfg.model.name

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    from faster_whisper import WhisperModel
    _cached_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _cached_model_name = model_name
    return _cached_model


def _download_model(model_name: str) -> bool:
    """Download the whisper model with a progress display."""
    print(f"\n  {CYAN}Downloading model '{model_name}'...{RESET}")
    print(f"  {DIM}This only happens once. The model is cached locally.{RESET}\n")

    try:
        # Suppress HuggingFace "unauthenticated requests" warning during download
        import logging
        hf_logger = logging.getLogger("huggingface_hub")
        prev_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)
        try:
            _get_or_load_model(model_name)
        finally:
            hf_logger.setLevel(prev_level)
        print(f"\n  {GREEN}✓{RESET} Model '{model_name}' ready!")
        return True
    except Exception as e:
        print(f"\n  {RED}✗{RESET} Download failed: {e}")
        return False


def _write_config(
    model: str, language: str, hotkey: str, method: str, streaming: bool, backend: str,
    sound_enabled: bool = True, notify_clipboard: bool = False,
) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_text = f"""# voiceio configuration, generated by setup wizard

[hotkey]
key = "{hotkey}"
backend = "{backend}"

[model]
name = "{model}"
language = "{language}"
device = "auto"
compute_type = "int8"

[audio]
sample_rate = 16000
device = "default"

[output]
method = "{method}"
streaming = {'true' if streaming else 'false'}

[feedback]
sound_enabled = {'true' if sound_enabled else 'false'}
notify_clipboard = {'true' if notify_clipboard else 'false'}

[tray]
enabled = false

[daemon]
log_level = "INFO"
"""
    CONFIG_PATH.write_text(config_text)
    print(f"\n  {GREEN}✓{RESET} Config saved to {DIM}{CONFIG_PATH}{RESET}")


def _setup_gnome_shortcut(hotkey: str) -> bool:
    # Always use absolute path because GNOME doesn't know about venvs
    toggle_path = None
    # Check venv first (most common case)
    venv_path = Path(sys.prefix) / "bin" / "voiceio-toggle"
    if venv_path.exists():
        toggle_path = str(venv_path.resolve())
    else:
        found = shutil.which("voiceio-toggle")
        if found:
            toggle_path = str(Path(found).resolve())

    if not toggle_path:
        print(f"  {RED}✗{RESET} voiceio-toggle not found")
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


def _streaming_test(model=None) -> None:
    """Record audio and stream transcription results in real-time."""
    import numpy as np
    import sounddevice as sd

    sample_rate = 16000
    chunk_secs = 0.5
    chunk_size = int(sample_rate * chunk_secs)
    silence_threshold = 0.01
    max_duration = 10

    if model is None:
        print(f"\n  {CYAN}Loading model...{RESET}", end="", flush=True)
        model = _get_or_load_model()
        print(f"\r  {GREEN}✓{RESET} Model loaded     ")

    from voiceio.config import load
    cfg = load()
    lang = cfg.model.language if cfg.model.language != "auto" else None

    print(f"\n  {YELLOW}Speak now!{RESET} (up to {max_duration}s, stops on 1.5s silence)")
    print(f"  {DIM}{'─' * 40}{RESET}")
    sys.stdout.write(f"  {BOLD}")
    sys.stdout.flush()

    audio_chunks: list[np.ndarray] = []
    silent_time = 0.0
    last_text = ""
    last_text_len = 0  # characters printed on screen
    recording = True

    def callback(indata, frames, time_info, status):
        nonlocal silent_time, recording
        if not recording:
            return
        audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        if rms < silence_threshold:
            silent_time += chunk_secs
        else:
            silent_time = 0.0

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        blocksize=chunk_size, callback=callback,
    )
    stream.start()
    start_time = time.time()

    try:
        while recording:
            time.sleep(0.8)
            elapsed = time.time() - start_time

            if elapsed > max_duration:
                recording = False
                break

            # Stop on sustained silence (but only after we've heard something)
            if silent_time >= 1.5 and len(audio_chunks) > 3:
                recording = False
                break

            if not audio_chunks:
                continue

            audio = np.concatenate(audio_chunks, axis=0).flatten()
            if len(audio) < sample_rate * 0.5:
                continue

            # Transcribe everything so far
            segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and text != last_text:
                # Clear previous text and rewrite
                if last_text_len > 0:
                    sys.stdout.write("\b" * last_text_len + " " * last_text_len + "\b" * last_text_len)
                sys.stdout.write(text)
                sys.stdout.flush()
                last_text_len = len(text)
                last_text = text
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

    # Final transcription of complete audio
    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        if len(audio) >= sample_rate * 0.3:
            segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
            final_text = " ".join(seg.text.strip() for seg in segments).strip()
            if final_text and final_text != last_text:
                if last_text_len > 0:
                    sys.stdout.write("\b" * last_text_len + " " * last_text_len + "\b" * last_text_len)
                sys.stdout.write(final_text)
                last_text = final_text

    sys.stdout.write(f"{RESET}\n")
    print(f"  {DIM}{'─' * 40}{RESET}")

    if last_text:
        print(f"  {GREEN}✓{RESET} Transcribed successfully!")
    else:
        print(f"  {YELLOW}⚠{RESET}  No speech detected. Check your microphone.")


def _test_hotkey(hotkey: str, backend: str) -> bool:
    """Start voiceio daemon briefly and test that the hotkey triggers."""
    from voiceio.hotkeys.socket_backend import SOCKET_PATH

    print(f"\n  {CYAN}Testing hotkey: {BOLD}{hotkey}{RESET}")
    print(f"  {DIM}Backend: {backend}{RESET}")

    if backend == "socket":
        # For socket backend, we test if the GNOME shortcut triggers voiceio-toggle
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

            # Diagnostic: test if voiceio-toggle itself works
            print(f"\n  {DIM}Diagnosing...{RESET}")
            venv_toggle = Path(sys.prefix) / "bin" / "voiceio-toggle"
            print(f"  {DIM}Running voiceio-toggle directly...{RESET}", end="", flush=True)

            # Re-bind socket for the direct test
            s2 = sock.socket(sock.AF_UNIX, sock.SOCK_DGRAM)
            s2.bind(str(SOCKET_PATH))
            s2.settimeout(3.0)

            try:
                subprocess.run([str(venv_toggle)], capture_output=True, timeout=3)
                data = s2.recv(64)
                if data == b"toggle":
                    print(f" {GREEN}OK{RESET}")
                    print(f"\n  {YELLOW}ℹ{RESET}  voiceio-toggle works, but the DE shortcut didn't fire.")
                    print("    The GNOME shortcut may need a moment to register, or")
                    print(f"    {BOLD}{hotkey}{RESET} may conflict with an existing shortcut.")
            except Exception:
                print(f" {RED}FAIL{RESET}")
                print(f"\n  {YELLOW}ℹ{RESET}  voiceio-toggle itself failed. This is unexpected.")

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


def run_test() -> None:
    """Standalone test command: voiceio-test."""
    print(f"{CYAN}{BOLD}voiceio test{RESET}\n")

    options = [
        ("Mic + streaming transcription", "Test your microphone and see real-time transcription"),
        ("Hotkey test", "Verify your keyboard shortcut works"),
        ("Full test", "Both of the above"),
    ]

    idx = _ask_choice(options, default=0)

    if idx in (0, 2):
        print(f"\n{BOLD}Mic test{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        _streaming_test()

    if idx in (1, 2):
        print(f"\n{BOLD}Hotkey test{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        from voiceio.config import load
        cfg = load()
        ok = _test_hotkey(cfg.hotkey.key, cfg.hotkey.backend)
        if not ok:
            print(f"\n  {YELLOW}Troubleshooting:{RESET}")
            print(f"  {DIM}• On Wayland/GNOME: run {BOLD}voiceio --setup-shortcut{RESET}")
            print(f"  {DIM}• Or add shortcut manually: Settings → Keyboard → Custom Shortcuts{RESET}")
            print(f"  {DIM}  Command: voiceio-toggle{RESET}")

    print()


def run_wizard() -> None:
    print(LOGO)

    total_steps = 9

    # ── Step 1: System check ────────────────────────────────────────────
    _print_step(1, total_steps, "System check")
    checks = _check_system()

    _print_check("Display server", True, checks["display"])
    _print_check("Audio input", checks["audio"],
                 f"{len(checks['audio_devices'])} device(s)" if checks["audio"] else "no devices found")

    # IBus (preferred typer on Linux)
    if checks["ibus"] and checks["ibus_gi"]:
        _print_check("IBus", True, "recommended, atomic text insertion")
    elif checks["ibus"]:
        _print_check("IBus", False, "install bindings: sudo apt install gir1.2-ibus-1.0")
    else:
        _print_check("IBus", False, "install: sudo apt install ibus gir1.2-ibus-1.0")

    # Fallback typers (optional)
    if checks["display"] == "wayland":
        _print_check("ydotool", checks["ydotool"],
                     "fallback" if checks["ydotool"] else "optional: sudo apt install ydotool",
                     optional=True)
        _print_check("wtype", checks["wtype"],
                     "fallback" if checks["wtype"] else "optional: sudo apt install wtype",
                     optional=True)
    else:
        _print_check("xdotool", checks["xdotool"],
                     "fallback" if checks["xdotool"] else "optional: sudo apt install xdotool",
                     optional=True)

    _print_check("CUDA GPU", checks["cuda"],
                 "will use GPU" if checks["cuda"] else "will use CPU (still fast)",
                 optional=True)

    if checks["display"] == "wayland":
        _print_check("Input group (evdev)", checks["input_group"],
                     "" if checks["input_group"] else "optional: sudo usermod -aG input $USER",
                     optional=True)

    # Install CLI symlinks to ~/.local/bin/
    from voiceio.service import install_symlinks, symlinks_installed, path_hint_needed, _is_pipx_install
    if _is_pipx_install():
        _print_check("CLI commands", True, "installed via pipx (already on PATH)")
    elif not symlinks_installed():
        linked = install_symlinks()
        if linked:
            _print_check("CLI commands", True, f"linked {len(linked)} commands to ~/.local/bin/")
            if path_hint_needed():
                print(f"  {YELLOW}ℹ{RESET}  {DIM}Restart your terminal for 'voiceio' to be on PATH{RESET}")
        else:
            _print_check("CLI commands", False, "could not create symlinks in ~/.local/bin/")
    else:
        _print_check("CLI commands", True, "voiceio in PATH")

    if not checks["audio"]:
        print(f"\n  {RED}No microphone found. Connect one and try again.{RESET}")
        sys.exit(1)

    # Need at least one typer
    has_typer = checks["ibus"] and checks["ibus_gi"]
    has_typer = has_typer or checks["xdotool"] or checks["ydotool"] or checks["wtype"]
    if not has_typer:
        print(f"\n  {RED}No text injection backend available.{RESET}")
        print(f"  {DIM}Install one: sudo apt install ibus gir1.2-ibus-1.0{RESET}")
        sys.exit(1)

    # ── Step 2: Choose model ────────────────────────────────────────────
    _print_step(2, total_steps, "Choose a Whisper model")
    print(f"  {DIM}Larger models are more accurate but slower and use more RAM.{RESET}\n")
    model_idx = _ask_choice(MODELS, default=1)
    model_name = MODELS[model_idx][0]

    # ── Step 3: Language ────────────────────────────────────────────────
    _print_step(3, total_steps, "Language")
    print(f"  {DIM}Pick your primary language, or auto-detect.{RESET}\n")
    lang_idx = _ask_choice(LANGUAGES, default=0)
    language = LANGUAGES[lang_idx][0]

    # ── Step 4: Hotkey ──────────────────────────────────────────────────
    _print_step(4, total_steps, "Keyboard shortcut")
    print(f"  {DIM}This combo toggles recording on/off.{RESET}\n")
    hotkey_options = [
        ("ctrl+alt+v", "Ctrl + Alt + V (recommended)"),
        ("alt+v", "Alt + V"),
        ("ctrl+shift+v", "Ctrl + Shift + V"),
        ("super+v", "Super + V (may not work on Wayland/GNOME)"),
        ("Custom",),
    ]
    hk_idx = _ask_choice(hotkey_options, default=0)
    if hk_idx == len(hotkey_options) - 1:
        hotkey = _ask("Enter combo (e.g. ctrl+shift+r)", "super+v")
    else:
        hotkey = hotkey_options[hk_idx][0]

    # Output method: auto selects best available (IBus preferred)
    method = "auto"
    if checks["ibus"] and checks["ibus_gi"]:
        print(f"\n  {GREEN}✓{RESET} {DIM}Text injection: IBus (best quality, auto-selected){RESET}")
        # Install IBus component and add GNOME input source
        from voiceio.typers.ibus import install_component, _ensure_gnome_input_source
        if install_component():
            print(f"  {GREEN}✓{RESET} {DIM}IBus engine component installed{RESET}")
            _ensure_gnome_input_source()
            print(f"  {GREEN}✓{RESET} {DIM}Added VoiceIO to GNOME input sources{RESET}")
        else:
            print(f"  {YELLOW}⚠{RESET}  {DIM}Could not install IBus component, will use fallback{RESET}")

    # Backend
    if checks["display"] == "wayland":
        if checks["input_group"]:
            backend = "evdev"
        else:
            backend = "socket"
    else:
        backend = "auto"

    # ── Step 5: Feedback ───────────────────────────────────────────────
    _print_step(5, total_steps, "Feedback")
    print(f"  {DIM}Sound plays when text is committed. Notifications show clipboard status.{RESET}\n")
    feedback_options = [
        ("Sound only", "short chime on commit"),
        ("Sound + notification", "also shows a desktop notification"),
        ("None", "silent"),
    ]
    fb_idx = _ask_choice(feedback_options, default=0)
    sound_enabled = fb_idx in (0, 1)
    notify_clipboard = fb_idx == 1

    # ── Step 6: Download model ──────────────────────────────────────────
    _print_step(6, total_steps, "Download model")
    if not _download_model(model_name):
        sys.exit(1)

    # ── Step 7: Save config & set up shortcut ───────────────────────────
    _print_step(7, total_steps, "Save config & shortcut")

    _write_config(model_name, language, hotkey, method, streaming=True, backend=backend,
                  sound_enabled=sound_enabled, notify_clipboard=notify_clipboard)

    # Set up DE shortcut if on GNOME + socket backend
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    if "GNOME" in desktop and backend == "socket":
        print(f"\n  {CYAN}Setting up GNOME keyboard shortcut...{RESET}")
        if _setup_gnome_shortcut(hotkey):
            print(f"  {GREEN}✓{RESET} Shortcut {BOLD}{hotkey}{RESET} → voiceio-toggle configured!")
        else:
            print(f"  {YELLOW}⚠{RESET}  Auto-setup failed. Add manually in Settings → Keyboard → Shortcuts:")
            print("    Command: voiceio-toggle")
    elif backend == "socket":
        print(f"\n  {YELLOW}ℹ{RESET}  Add a keyboard shortcut manually in your DE settings:")
        print(f"    Shortcut: {BOLD}{hotkey}{RESET}")
        print(f"    Command:  {BOLD}voiceio-toggle{RESET}")

    # ── Step 8: Autostart ─────────────────────────────────────────────────
    _print_step(8, total_steps, "Autostart")
    from voiceio.service import has_systemd
    autostart_idx = 1  # default: no autostart
    if has_systemd():
        print(f"  {DIM}Install a systemd user service so voiceio starts on login{RESET}")
        print(f"  {DIM}and restarts automatically if it crashes.{RESET}\n")
        autostart_options = [
            ("Yes", "install & enable systemd service"),
            ("No", "I'll start it manually"),
        ]
        autostart_idx = _ask_choice(autostart_options, default=0)
        if autostart_idx == 0:
            from voiceio.service import install_service
            if install_service():
                print(f"  {GREEN}✓{RESET} Systemd service installed and enabled")
                print(f"  {DIM}voiceio will start automatically on next login{RESET}")
            else:
                print(f"  {YELLOW}⚠{RESET}  Could not install systemd service")
                print(f"  {DIM}Start manually with: voiceio{RESET}")
    else:
        print(f"  {DIM}systemd not available, skipping autostart setup{RESET}")
        print(f"  {DIM}Start manually with: voiceio{RESET}")

    # ── Step 9: Test ────────────────────────────────────────────────────
    _print_step(9, total_steps, "Test")

    # Hotkey test
    print(f"  {DIM}Let's verify your shortcut works.{RESET}")
    hotkey_ok = _test_hotkey(hotkey, backend)

    if not hotkey_ok:
        print(f"\n  {YELLOW}Troubleshooting:{RESET}")
        if backend == "socket" and "GNOME" in desktop:
            print(f"  {DIM}• The GNOME shortcut may need a moment to register.{RESET}")
            print(f"  {DIM}• Try: Settings → Keyboard → Custom Shortcuts to verify.{RESET}")
            print(f"  {DIM}• Shortcut command should be: {BOLD}voiceio-toggle{RESET}")
        retry = _ask("Try a different shortcut? (y/n)", "y")
        if retry.lower() in ("y", "yes"):
            print()
            hk_idx = _ask_choice(hotkey_options, default=1)
            if hk_idx == len(hotkey_options) - 1:
                hotkey = _ask("Enter combo (e.g. ctrl+shift+r)", "ctrl+shift+v")
            else:
                hotkey = hotkey_options[hk_idx][0]

            # Re-save config and shortcut with new hotkey
            _write_config(model_name, language, hotkey, method, streaming=True, backend=backend,
                  sound_enabled=sound_enabled, notify_clipboard=notify_clipboard)
            if "GNOME" in desktop and backend == "socket":
                _setup_gnome_shortcut(hotkey)

            hotkey_ok = _test_hotkey(hotkey, backend)

    # Mic + streaming test
    print(f"\n{'─' * 50}")
    test = _ask("Run a streaming mic test? (y/n)", "y")
    if test.lower() in ("y", "yes", ""):
        _streaming_test(model=_get_or_load_model())

    # ── Done ────────────────────────────────────────────────────────────
    # Start (or restart) the service so it's immediately usable
    if autostart_idx == 0:
        from voiceio.service import is_running
        action = "restart" if is_running() else "start"
        try:
            subprocess.run(
                ["systemctl", "--user", action, "voiceio.service"],
                capture_output=True, timeout=5,
            )
            print(f"  {GREEN}✓{RESET} {DIM}voiceio service started{RESET}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    from voiceio.config import LOG_PATH
    log_path = LOG_PATH
    start_hint = (
        "  voiceio is running and will start automatically on login."
        if autostart_idx == 0
        else f"  Start voiceio:\n    {CYAN}voiceio{RESET}"
    )
    print(f"""
{GREEN}{'━' * 50}{RESET}
{BOLD}  Setup complete!{RESET}

{start_hint}

  Press {BOLD}{hotkey}{RESET} to toggle recording.
  Speak naturally, and text streams at your cursor.

  Useful commands:
    {CYAN}voiceio doctor{RESET}   check system health
    {CYAN}voiceio test{RESET}     test mic + hotkey

  Config: {DIM}{CONFIG_PATH}{RESET}
  Logs:   {DIM}{log_path}{RESET}
{GREEN}{'━' * 50}{RESET}
""")
