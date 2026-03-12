"""Interactive setup wizard for voiceio."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from voiceio.config import CONFIG_DIR, CONFIG_PATH

log = logging.getLogger(__name__)

# ── Colors ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# ── Spinner ─────────────────────────────────────────────────────────────────

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Braille spinner for indeterminate waits. Use as context manager."""

    def __init__(self, message: str):
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._final: str | None = None

    def __enter__(self) -> Spinner:
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Clear spinner line and show final status
        sys.stdout.write("\r\033[K")
        if self._final:
            sys.stdout.write(f"{self._final}\n")
        sys.stdout.flush()

    def ok(self, message: str | None = None) -> None:
        self._final = f"  {GREEN}✓{RESET} {message or self._message}"

    def fail(self, message: str | None = None) -> None:
        self._final = f"  {RED}✗{RESET} {message or self._message}"

    def warn(self, message: str | None = None) -> None:
        self._final = f"  {YELLOW}⚠{RESET}  {message or self._message}"

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = _BRAILLE[i % len(_BRAILLE)]
            sys.stdout.write(f"\r  {CYAN}{frame}{RESET} {self._message}")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)


LOGO = f"""{CYAN}{BOLD}
 ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██╗ ██████╗
 ██║   ██║██╔═══██╗██║██╔════╝██╔════╝██║██╔═══██╗
 ██║   ██║██║   ██║██║██║     █████╗  ██║██║   ██║
 ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██║██║   ██║
  ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██║╚██████╔╝
   ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═╝ ╚═════╝
{RESET}{DIM}  speak → text, locally, instantly{RESET}
"""

LOGO_CORRECT = f"""{CYAN}{BOLD}\
  ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ ██████╗████████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝
 ██║     ██║   ██║██████╔╝██████╔╝█████╗  ██║        ██║
 ██║     ██║   ██║██╔══██╗██╔══██╗██╔══╝  ██║        ██║
 ╚██████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗   ██║
  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝   ╚═╝
{RESET}{DIM}  find and fix transcription mistakes{RESET}
"""

LOGO_DOCTOR = f"""{CYAN}{BOLD}\
 ██████╗  ██████╗  ██████╗████████╗ ██████╗ ██████╗
 ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
 ██║  ██║██║   ██║██║        ██║   ██║   ██║██████╔╝
 ██║  ██║██║   ██║██║        ██║   ██║   ██║██╔══██╗
 ██████╔╝╚██████╔╝╚██████╗   ██║   ╚██████╔╝██║  ██║
 ╚═════╝  ╚═════╝  ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
{RESET}{DIM}  system health check{RESET}
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


def _rl_prompt(prompt: str) -> str:
    """Wrap ANSI escapes in \\001/\\002 so readline calculates width correctly."""
    import re
    return re.sub(r"(\033\[[0-9;]*m)", r"\001\1\002", prompt)


def _ask(prompt: str, default: str = "") -> str:
    default_hint = f" {DIM}[{default}]{RESET}" if default else ""
    try:
        answer = input(_rl_prompt(f"  {CYAN}›{RESET} {prompt}{default_hint}: ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer or default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    """Yes/no prompt with arrow key toggle."""
    options = [("Yes",), ("No",)]
    default_idx = 0 if default else 1
    print(f"  {CYAN}›{RESET} {prompt}\n")
    idx = _ask_choice(options, default=default_idx)
    return idx == 0


class _MenuRenderer:
    """Shared ANSI menu rendering for interactive choice menus."""

    _ansi_re = __import__("re").compile(r"\033\[[0-9;]*m")

    def __init__(self, options: list[tuple[str, ...]], selected: int = 0):
        self.options = options
        self.selected = selected
        self.n = len(options)
        try:
            self.cols = os.get_terminal_size().columns
        except OSError:
            self.cols = 80
        self.hint = f"  {DIM}\u2191\u2193 navigate, enter to confirm{RESET}"

    def _visible_len(self, s: str) -> int:
        return len(self._ansi_re.sub("", s))

    def _truncate(self, s: str, max_width: int) -> str:
        vis = 0
        result = []
        for part in self._ansi_re.split(s):
            if self._ansi_re.match(part):
                result.append(part)
            else:
                remaining = max_width - vis
                result.append(part[:remaining])
                vis += min(len(part), remaining)
                if vis >= max_width:
                    break
        return "".join(result) + RESET

    def format_line(self, i: int) -> str:
        marker = f"{GREEN}\u25cf{RESET}" if i == self.selected else f"{DIM}\u25cb{RESET}"
        label = self.options[i][0]
        detail = f"  {DIM}({', '.join(self.options[i][1:])}){RESET}" if len(self.options[i]) > 1 else ""
        line = f"  {marker} {BOLD}{i + 1}{RESET}. {label}{detail}"
        if self._visible_len(line) >= self.cols:
            line = self._truncate(line, self.cols - 2)
        return line

    def draw(self) -> None:
        sys.stdout.write("\033[J")
        for i in range(self.n):
            sys.stdout.write(f"{self.format_line(i)}\r\n")
        sys.stdout.write(f"{self.hint}\r")
        sys.stdout.flush()

    def redraw(self) -> None:
        sys.stdout.write(f"\033[{self.n}A")
        self.draw()

    def handle_key(self, key: str) -> str | None:
        """Process a key name. Returns 'done' on enter, None otherwise."""
        if key in ("up", "w", "k"):
            self.selected = (self.selected - 1) % self.n
        elif key in ("down", "s", "j"):
            self.selected = (self.selected + 1) % self.n
        elif key == "enter":
            return "done"
        elif key == "ctrl-c":
            sys.stdout.write("\033[?25h\r\n")
            sys.exit(0)
        elif key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < self.n:
                self.selected = idx
        else:
            return None
        self.redraw()
        return None

    def finish(self) -> int:
        self.redraw()
        sys.stdout.write("\033[?25h\r\n\n")
        sys.stdout.flush()
        return self.selected


def _ask_choice(options: list[tuple[str, ...]], default: int = 0) -> int:
    """Interactive choice with arrow keys, w/s, j/k navigation. Enter to confirm."""
    import atexit
    atexit.register(lambda: sys.stdout.write("\033[?25h"))

    menu = _MenuRenderer(options, default)
    sys.stdout.write("\033[?25l")
    menu.draw()

    if sys.platform == "win32":
        return _menu_loop_win(menu)
    return _menu_loop_unix(menu)


def _menu_loop_win(menu: _MenuRenderer) -> int:
    """Read keys using msvcrt (Windows)."""
    import msvcrt

    # Enable ANSI escape codes on legacy cmd.exe
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

    try:
        while True:
            ch = msvcrt.getwch()
            if ch in ("\xe0", "\x00"):
                ch2 = msvcrt.getwch()
                key = {"H": "up", "P": "down"}.get(ch2)
                if key and menu.handle_key(key) == "done":
                    break
            elif ch == "\r":
                if menu.handle_key("enter") == "done":
                    break
            elif ch == "\x03":
                menu.handle_key("ctrl-c")
            else:
                menu.handle_key(ch)
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\r\n")
        sys.exit(0)

    return menu.finish()


def _menu_loop_unix(menu: _MenuRenderer) -> int:
    """Read keys using termios/tty/select (Unix)."""
    import select
    import termios
    import tty

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

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = _read_key(fd)
            if menu.handle_key(key) == "done":
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        termios.tcflush(fd, termios.TCIFLUSH)

    return menu.finish()


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _install_pip_package(package: str) -> bool:
    """Offer to pip-install a package. Returns True if installed successfully."""
    if not _ask_yn(f"Install {package}? (pip install {package})"):
        return False
    with Spinner(f"Installing {package}...") as sp:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True, timeout=120,
            )
            if result.returncode == 0:
                sp.ok(f"{package} installed!")
                return True
            sp.fail(f"pip install failed (exit {result.returncode})")
            stderr = result.stderr.decode()[:200]
            if stderr:
                print(f"  {DIM}{stderr}{RESET}")
            return False
        except subprocess.TimeoutExpired:
            sp.fail("Installation timed out")
            return False


def _satisfaction_loop(
    options: list[tuple[str, ...]],
    demo_fn: Callable[[int], bool],
    prompt: str = "Are you satisfied?",
    default: int = 0,
) -> int | None:
    """Reusable demo-then-ask loop.

    Calls demo_fn(index) for the selected option. If the demo succeeds,
    asks the user if they're satisfied. If not, offers to try a different
    option. Returns the final selected index, or None if user skips all.
    """
    tried: set[int] = set()
    last_ok: int | None = None
    current = default

    while True:
        ok = demo_fn(current)
        tried.add(current)
        if ok:
            last_ok = current
            if _ask_yn(prompt):
                return current

        # Build list of remaining options
        remaining = [(i, opt) for i, opt in enumerate(options) if i not in tried]
        if not remaining:
            print(f"  {DIM}All options tried.{RESET}")
            return last_ok if last_ok is not None else current

        print(f"\n  {DIM}Other options:{RESET}\n")
        menu_options = [opt for _, opt in remaining]
        # Only show "Keep current" if we have a working option
        if last_ok is not None:
            keep_label = options[last_ok][0]
            menu_options.append((f"Keep {keep_label}",))
        choice = _ask_choice(menu_options, default=0)

        if last_ok is not None and choice == len(menu_options) - 1:
            return last_ok
        current = remaining[choice][0]


def _wait_for_service_ready(timeout: float = 30) -> bool:
    """Wait for the voiceio service to finish loading by checking for its socket/PID file."""
    from voiceio.config import PID_PATH
    from voiceio.hotkeys.socket_backend import SOCKET_PATH, _IS_WINDOWS

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _IS_WINDOWS:
            # On Windows there's no Unix socket; check PID file as readiness signal
            if PID_PATH.exists():
                return True
        else:
            if SOCKET_PATH.exists():
                return True
        time.sleep(0.5)
    return False


def _check_system() -> dict:
    """Check system dependencies and capabilities."""
    checks = {}

    _is_win = sys.platform == "win32"
    _is_mac = sys.platform == "darwin"
    checks["is_windows"] = _is_win
    checks["is_mac"] = _is_mac
    checks["is_linux"] = not _is_win and not _is_mac

    # Display server
    if _is_win:
        checks["display"] = "win32"
    elif _is_mac:
        checks["display"] = "darwin"
    else:
        checks["display"] = os.environ.get("XDG_SESSION_TYPE", "unknown")

    # pynput (Windows/macOS primary backend)
    checks["pynput"] = False
    if _is_win or _is_mac:
        try:
            import pynput  # noqa: F401
            checks["pynput"] = True
        except ImportError:
            pass

    # Linux-only typer binaries
    checks["xdotool"] = _check_binary("xdotool") if checks["is_linux"] else False
    checks["xclip"] = _check_binary("xclip") if checks["is_linux"] else False
    checks["ydotool"] = _check_binary("ydotool") if checks["is_linux"] else False
    checks["wtype"] = _check_binary("wtype") if checks["is_linux"] else False
    checks["ibus"] = _check_binary("ibus") if checks["is_linux"] else False

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

    # Input group (for evdev, Linux only)
    checks["input_group"] = False
    if checks["is_linux"]:
        groups = os.getgroups()
        try:
            import grp
            input_gid = grp.getgrnam("input").gr_gid
            checks["input_group"] = input_gid in groups
        except (KeyError, ImportError):
            pass

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
    """Download the whisper model with a spinner."""
    print(f"  {DIM}This only happens once. The model is cached locally.{RESET}")

    with Spinner(f"Downloading model '{model_name}'...") as sp:
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
            sp.ok(f"Model '{model_name}' ready!")
            return True
        except Exception as e:
            sp.fail(f"Download failed: {e}")
            return False


def _write_config(
    model: str, language: str, hotkey: str, method: str, streaming: bool, backend: str,
    sound_enabled: bool = True, notify_clipboard: bool = False,
    tray_enabled: bool = False,
    commands_enabled: bool = True,
    punctuation_cleanup: bool = True,
    number_conversion: bool = True,
    llm_enabled: bool = False,
    llm_model: str = "",
    autocorrect_api_key: str = "",
    autocorrect_base_url: str = "",
    autocorrect_model: str = "",
    tts_enabled: bool = True,
    tts_engine: str = "auto",
    tts_speed: float = 1.0,
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
auto_stop_silence_secs = 5.0

[output]
method = "{method}"
streaming = {'true' if streaming else 'false'}
punctuation_cleanup = {'true' if punctuation_cleanup else 'false'}
number_conversion = {'true' if number_conversion else 'false'}

[commands]
enabled = {'true' if commands_enabled else 'false'}

[feedback]
sound_enabled = {'true' if sound_enabled else 'false'}
notify_clipboard = {'true' if notify_clipboard else 'false'}

[tray]
enabled = {'true' if tray_enabled else 'false'}

[llm]
enabled = {'true' if llm_enabled else 'false'}
model = "{llm_model}"

[autocorrect]
api_key = "{autocorrect_api_key}"
base_url = "{autocorrect_base_url or 'https://openrouter.ai/api/v1'}"
model = "{autocorrect_model or 'anthropic/claude-sonnet-4'}"

[tts]
enabled = {'true' if tts_enabled else 'false'}
engine = "{tts_engine}"
speed = {tts_speed}

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


def _streaming_test(model=None, language: str | None = None) -> str:
    """Record audio and stream transcription results in real-time. Returns transcribed text."""
    import numpy as np
    import sounddevice as sd

    from voiceio.postprocess import cleanup

    sample_rate = 16000
    chunk_secs = 0.5
    chunk_size = int(sample_rate * chunk_secs)
    silence_threshold = 0.01
    max_duration = 10

    if model is None:
        with Spinner("Loading model...") as sp:
            model = _get_or_load_model()
            sp.ok("Model loaded")

    if language is None:
        from voiceio.config import load
        cfg = load()
        language = cfg.model.language

    lang = language if language != "auto" else None
    lang_str = lang or "en"

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
            # Light cleanup for streaming preview (capitalization, punctuation spacing)
            text = cleanup(text, lang_str)

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

    # Final transcription with cleanup (capitalization, punctuation spacing)
    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        if len(audio) >= sample_rate * 0.3:
            segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
            final_text = " ".join(seg.text.strip() for seg in segments).strip()
            final_text = cleanup(final_text, lang_str)
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

    return last_text


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

    # ── Step 1: System check ────────────────────────────────────────────
    _print_step(1, 8, "System check")
    checks = _check_system()

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

    if checks["display"] == "wayland":
        _print_check("Input group (evdev)", checks["input_group"],
                     "" if checks["input_group"] else "optional: sudo usermod -aG input $USER",
                     optional=True)

    # Tray icon
    from voiceio.tray import probe_availability
    tray_ok, tray_reason, tray_fix_hint = probe_availability()
    if tray_ok:
        _print_check("Tray icon", True, "system tray indicator available")
    else:
        _print_check("Tray icon", False,
                     tray_fix_hint if tray_fix_hint else tray_reason,
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
                print(f"  {YELLOW}ℹ{RESET}  {DIM}Restart your terminal so 'voiceio' is on PATH{RESET}")
            else:
                print(f"  {DIM}  You can now run 'voiceio' from any terminal{RESET}")
        else:
            _print_check("CLI commands", False, "could not create symlinks in ~/.local/bin/")
    else:
        _print_check("CLI commands", True, "voiceio in PATH")

    if not checks["audio"]:
        print(f"\n  {RED}No microphone found. Connect one and try again.{RESET}")
        sys.exit(1)

    if checks["is_windows"] or checks["is_mac"]:
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

    # ── Step 2: Language ────────────────────────────────────────────────
    _print_step(2, 8, "Language")
    print(f"  {DIM}Pick your primary language, or auto-detect.{RESET}\n")
    lang_idx = _ask_choice(LANGUAGES, default=0)
    language = LANGUAGES[lang_idx][0]

    # ── Step 3: Speech recognition model ────────────────────────────────
    _print_step(3, 8, "Speech recognition model")
    print(f"  {DIM}Larger models are more accurate but slower and use more RAM.{RESET}\n")
    model_idx = _ask_choice(MODELS, default=1)
    model_name = MODELS[model_idx][0]

    # Download immediately
    if not _download_model(model_name):
        sys.exit(1)

    # Quick mic test + satisfaction loop to try different models
    if _ask_yn("Test it with your microphone?"):
        current_idx = model_idx

        def _stt_demo_fn(idx: int) -> bool:
            name = MODELS[idx][0]
            # Already-downloaded models are cached; only download new ones
            if _cached_model_name != name:
                if not _download_model(name):
                    return False
            result = _streaming_test(model=_get_or_load_model(name), language=language)
            return bool(result)

        selected = _satisfaction_loop(
            options=MODELS,
            demo_fn=_stt_demo_fn,
            prompt="Happy with the transcription quality?",
            default=current_idx,
        )
        if selected is not None:
            model_name = MODELS[selected][0]

    # ── Step 4: Hotkey ──────────────────────────────────────────────────
    _print_step(4, 8, "Keyboard shortcut")
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

    # Output method: auto selects best available
    method = "auto"
    if checks["is_windows"] or checks["is_mac"]:
        backend = "auto"
        label = "pynput" if checks["is_windows"] else "pynput (requires Accessibility permission)"
        print(f"\n  {GREEN}✓{RESET} {DIM}Text injection: {label}{RESET}")
    else:
        if checks["ibus"] and checks["ibus_gi"]:
            print(f"\n  {GREEN}✓{RESET} {DIM}Text injection: IBus (best quality, auto-selected){RESET}")
            from voiceio.typers.ibus import install_component, _ensure_gnome_input_source
            if install_component():
                print(f"  {GREEN}✓{RESET} {DIM}IBus engine component installed{RESET}")
                _ensure_gnome_input_source()
                print(f"  {GREEN}✓{RESET} {DIM}Added VoiceIO to GNOME input sources{RESET}")
            else:
                print(f"  {YELLOW}⚠{RESET}  {DIM}Could not install IBus component, will use fallback{RESET}")

        if checks["display"] == "wayland":
            if checks["input_group"]:
                backend = "evdev"
            else:
                backend = "socket"
        else:
            backend = "auto"

    # ── Defaults for optional settings ──────────────────────────────────
    sound_enabled = True
    notify_clipboard = False
    commands_enabled = True
    punctuation_cleanup = True
    number_conversion = True
    llm_enabled = False
    llm_model = ""
    tts_enabled = True
    tts_engine = "auto"
    tts_speed = 1.0

    # ── Step 5: Text-to-speech ──────────────────────────────────────────
    _print_step(5, 8, "Text-to-speech")
    print(f"  {DIM}Select text and press ctrl+alt+s to hear it spoken.{RESET}")
    print(f"  {DIM}Engines: piper (offline), edge-tts (cloud), espeak-ng (system).{RESET}\n")
    tts_top_options = [
        ("Enabled", "auto-select best available engine"),
        ("Choose engine", "hear a demo and pick"),
        ("Disabled",),
    ]
    tts_choice = _ask_choice(tts_top_options, default=0)
    if tts_choice == 2:
        tts_enabled = False
    elif tts_choice == 1:
        # Let user pick an engine first, then demo it
        engine_options = [
            ("piper", "offline, best quality"),
            ("edge-tts", "cloud, free Microsoft TTS"),
            ("espeak", "system package, lightweight"),
        ]
        print(f"\n  {DIM}Pick an engine to hear a demo:{RESET}\n")
        first_pick = _ask_choice(engine_options, default=0)

        def _tts_demo_fn(idx: int) -> bool:
            return _tts_demo(engine_options[idx][0])

        selected = _satisfaction_loop(
            options=engine_options,
            demo_fn=_tts_demo_fn,
            prompt="Happy with this voice?",
            default=first_pick,
        )
        tts_engine = engine_options[selected][0] if selected is not None else "auto"

        print(f"\n  {DIM}Speech speed:{RESET}\n")
        speed_options = [
            ("Normal", "1.0x"),
            ("Slow", "0.8x"),
            ("Fast", "1.3x"),
        ]
        sp_idx = _ask_choice(speed_options, default=0)
        tts_speed = [1.0, 0.8, 1.3][sp_idx]

    # ── Step 6: LLM & autocorrect ───────────────────────────────────────
    _print_step(6, 8, "LLM & autocorrect")
    print(f"  {DIM}Use a local LLM to fix grammar and spelling in your dictation.{RESET}")
    print(f"  {DIM}Runs only on the final pass — streaming preview is unaffected.{RESET}")

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
            default_llm = 0
        else:
            llm_options = [
                ("Disabled", "recommended without GPU"),
                ("Enabled", "adds 5-15s latency on CPU"),
            ]
            default_llm = 0
    elif _llm_status == OllamaStatus.NOT_INSTALLED:
        if has_gpu:
            size_note = "~800 MB Ollama + ~400 MB model"
            print(f"  {DIM}Adds ~0.5-2s latency per dictation with GPU.{RESET}\n")
            llm_options = [
                ("Install Ollama + model", f"downloads {size_note}"),
                ("Skip for now", "you can enable later with 'voiceio setup'"),
            ]
            default_llm = 0
        else:
            size_note = "~60 MB Ollama + ~400 MB model"
            print(f"  {DIM}Requires Ollama ({size_note}).{RESET}\n")
            llm_options = [
                ("Skip for now", "recommended without GPU"),
                ("Install anyway", f"downloads {size_note}, adds 5-15s latency"),
            ]
            default_llm = 0
    else:
        # Installed but not running, or missing model
        print(f"  {DIM}Ollama is installed. Just needs a model (~400 MB).{RESET}\n")
        if has_gpu:
            llm_options = [
                ("Enabled", "pull a small model"),
                ("Disabled", "skip LLM"),
            ]
            default_llm = 0
        else:
            llm_options = [
                ("Disabled", "recommended without GPU"),
                ("Enabled", "pull a model, adds 5-15s latency on CPU"),
            ]
            default_llm = 0

    llm_choice = _ask_choice(llm_options, default=default_llm)
    # With GPU: enable option is always first (index 0)
    # Without GPU: enable option is always second (index 1)
    want_llm = llm_choice == (0 if has_gpu else 1)

    if want_llm:
        llm_enabled = True
        llm_model = _setup_ollama(_llm_status, _llm_models)
        if not llm_model:
            llm_enabled = False
    else:
        llm_enabled = False
        llm_model = ""

    # Autocorrect (cloud LLM)
    autocorrect_api_key = ""
    autocorrect_base_url = ""
    autocorrect_model = ""
    print(f"\n  {BOLD}Autocorrect{RESET}")
    print(f"  {DIM}'voiceio correct --auto' uses a cloud LLM to find and fix Whisper mistakes.{RESET}\n")
    ac_options = [
        ("Skip", "no cloud API key"),
        ("Enable", "paste an API key (OpenRouter, OpenAI, or Anthropic)"),
    ]
    if _ask_choice(ac_options, default=0) == 1:
        api_input = _ask("API key", "")
        if api_input:
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
                autocorrect_api_key = api_input
                autocorrect_base_url = det_url
                autocorrect_model = det_model
            else:
                print(f"  {YELLOW}⚠{RESET}  Validation failed. You can set it later in config.toml.")

    # ── Step 7: Advanced options (optional) ─────────────────────────────
    _print_step(7, 8, "Advanced options")
    print(f"  {DIM}Sensible defaults are already set. Customize if you want.{RESET}\n")
    advanced_options = [
        ("Skip", "use defaults (recommended)"),
        ("Customize", "feedback, voice commands, punctuation, numbers"),
    ]
    if _ask_choice(advanced_options, default=0) == 1:
        adv = _run_advanced_options(checks, tray_ok)
        sound_enabled = adv.get("sound_enabled", True)
        notify_clipboard = adv.get("notify_clipboard", False)
        commands_enabled = adv.get("commands_enabled", True)
        punctuation_cleanup = adv.get("punctuation_cleanup", True)
        number_conversion = adv.get("number_conversion", True)

    # ── Step 8: Finalize ────────────────────────────────────────────────
    _print_step(8, 8, "Finalize")

    # Build config kwargs once — shared by initial save and later updates
    config_kwargs = dict(
        model=model_name, language=language, hotkey=hotkey, method=method,
        streaming=True, backend=backend,
        sound_enabled=sound_enabled, notify_clipboard=notify_clipboard,
        tray_enabled=tray_ok,
        commands_enabled=commands_enabled,
        punctuation_cleanup=punctuation_cleanup,
        number_conversion=number_conversion,
        llm_enabled=llm_enabled, llm_model=llm_model,
        autocorrect_api_key=autocorrect_api_key,
        autocorrect_base_url=autocorrect_base_url,
        autocorrect_model=autocorrect_model,
        tts_enabled=tts_enabled, tts_engine=tts_engine, tts_speed=tts_speed,
    )

    # Save config
    _write_config(**config_kwargs)

    # Set up DE shortcut if on GNOME + socket backend (Linux only)
    desktop = ""
    if checks["is_linux"]:
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

    # Autostart
    from voiceio.service import install_service
    autostart_idx = 1

    if checks["is_windows"]:
        with Spinner("Setting up Windows Startup...") as sp:
            if install_service():
                sp.ok("Added to Windows Startup")
                autostart_idx = 0
            else:
                sp.warn("Could not add to Startup folder")
    elif checks["is_mac"]:
        pass  # macOS autostart not yet supported
    else:
        from voiceio.service import has_systemd
        if has_systemd():
            with Spinner("Installing systemd service...") as sp:
                if install_service():
                    sp.ok("Systemd service installed and enabled")
                    autostart_idx = 0
                else:
                    sp.warn("Could not install service")

    # Quick hotkey test
    print(f"\n  {DIM}Testing hotkey...{RESET}")
    hotkey_ok = _test_hotkey(hotkey, backend)

    if not hotkey_ok:
        print(f"\n  {YELLOW}Troubleshooting:{RESET}")
        if backend == "socket" and "GNOME" in desktop:
            print(f"  {DIM}• The GNOME shortcut may need a moment to register.{RESET}")
            print(f"  {DIM}• Try: Settings → Keyboard → Custom Shortcuts to verify.{RESET}")
            print(f"  {DIM}• Shortcut command should be: {BOLD}voiceio-toggle{RESET}")
        if _ask_yn("Try a different shortcut?"):
            print()
            hk_idx = _ask_choice(hotkey_options, default=1)
            if hk_idx == len(hotkey_options) - 1:
                hotkey = _ask("Enter combo (e.g. ctrl+shift+r)", "ctrl+shift+v")
            else:
                hotkey = hotkey_options[hk_idx][0]

            config_kwargs["hotkey"] = hotkey
            _write_config(**config_kwargs)
            if "GNOME" in desktop and backend == "socket":
                _setup_gnome_shortcut(hotkey)

            hotkey_ok = _test_hotkey(hotkey, backend)

    # Start (or restart) the service so it's immediately usable
    if autostart_idx == 0:
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
                sp.warn("Timed out — service may still be loading. Check: voiceio logs")

    # ── Done ────────────────────────────────────────────────────────────
    from voiceio.config import LOG_PATH
    from voiceio.service import _is_pipx_install
    log_path = LOG_PATH
    start_hint = (
        "  voiceio is running and will start automatically on login."
        if autostart_idx == 0
        else f"  Start voiceio:\n    {CYAN}voiceio{RESET}"
    )
    path_note = ""
    if not _is_pipx_install() and not shutil.which("voiceio"):
        path_note = f"\n  {YELLOW}ℹ{RESET}  Restart your terminal so 'voiceio' is on PATH.\n"
    print(f"""
{GREEN}{'━' * 50}{RESET}
{BOLD}  Setup complete!{RESET}
{path_note}
{start_hint}

  Press {BOLD}{hotkey}{RESET} to toggle recording.
  Speak naturally, and text streams at your cursor.

  Useful commands:
    {CYAN}voiceio doctor{RESET}     check system health
    {CYAN}voiceio test{RESET}       test mic + hotkey
    {CYAN}voiceio correct{RESET}    manage corrections dictionary
    {CYAN}voiceio history{RESET}    view transcription history

  Config: {DIM}{CONFIG_PATH}{RESET}
  Logs:   {DIM}{log_path}{RESET}
{GREEN}{'━' * 50}{RESET}
""")


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

    return result
