"""Terminal UI primitives for the setup wizard: spinner, logos, prompts, menus."""
from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable

from voiceio.ui import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW


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

    def set_message(self, message: str) -> None:
        """Update the spinner message live (for showing progress)."""
        self._message = message

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


def _print_check(label: str, ok: bool, detail: str = "", optional: bool = False) -> None:
    if ok:
        icon = f"{GREEN}✓{RESET}"
    elif optional:
        icon = f"{YELLOW}○{RESET}"
    else:
        icon = f"{RED}✗{RESET}"
    extra = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {icon} {label}{extra}")


def _index_of(options: list[tuple[str, ...]], value: str, fallback: int = 0) -> int:
    """Index of the option whose first field == value (for current-value defaults)."""
    for i, opt in enumerate(options):
        if opt[0] == value:
            return i
    return fallback


def _current_note(is_rerun: bool, value: str) -> None:
    """Print a '(current: X)' line when re-running setup over an existing config."""
    if is_rerun and value != "":
        print(f"  {DIM}(current: {value}){RESET}")
