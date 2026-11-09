"""Where this voiceio install lives, and whether it will still be there tomorrow.

`uvx 'python-voiceio[desktop]'` runs voiceio from a throwaway environment in
uv's cache (``~/.cache/uv/archive-v0/<hash>/``). uv deletes it on
``uv cache clean`` / ``uv cache prune`` or when a newer version is resolved, so
a systemd unit, IBus launcher or ~/.local/bin link that points there breaks
some day without warning. Setup refuses to write such paths, and doctor flags
the ones already written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

UV_TOOL_INSTALL = "uv tool install 'python-voiceio[desktop]'"
PIPX_INSTALL = "pipx install 'python-voiceio[desktop]'"

# The Cache Directory Tagging Specification (bford.info/cachedir): a directory
# holding this file, with this first line, is a cache whose contents may be
# deleted at any time. uv writes it at the root of every cache it uses,
# including the temporary one of `uvx --no-cache`.
CACHEDIR_TAG = "CACHEDIR.TAG"
_CACHEDIR_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"


def uv_cache_dir() -> Path:
    """uv's default cache directory, resolved the way uv resolves it:
    $UV_CACHE_DIR, else the platform cache dir ($XDG_CACHE_HOME or ~/.cache
    on Linux and macOS, %LOCALAPPDATA% on Windows) + ``uv``."""
    if env := os.environ.get("UV_CACHE_DIR"):
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "uv" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    # XDG: a relative value is invalid and must be ignored.
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".cache"
    return base / "uv"


def _is_cachedir(d: Path) -> bool:
    try:
        with open(d / CACHEDIR_TAG, encoding="utf-8", errors="replace") as f:
            return f.read(len(_CACHEDIR_SIGNATURE)) == _CACHEDIR_SIGNATURE
    except OSError:
        return False


def in_uv_cache(path: str | os.PathLike) -> bool:
    """True when ``path`` lies inside uv's cache (or any tagged cache dir).

    Two checks, either suffices: the path is under uv_cache_dir(), or one of
    its parent directories carries a CACHEDIR.TAG (catches `--cache-dir`,
    `--no-cache` and a UV_CACHE_DIR that differed when the file was written).
    uv sets no variable that marks an ephemeral run: `$UV` is also set by
    `uv run` in a project and inherited by every child, and pyvenv.cfg's
    `uv = ...` line is also in `uv tool install` and `uv venv` environments.
    """
    p = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        p = p.resolve()
    except OSError:
        pass
    cache = uv_cache_dir()
    for c in {cache, Path(os.path.abspath(cache))}:
        try:
            c = c.resolve()
        except OSError:
            pass
        if p == c or c in p.parents:
            return True
    return any(_is_cachedir(d) for d in p.parents if d != d.parent)


def running_ephemeral() -> bool:
    """This interpreter runs from a uvx environment that uv may delete."""
    return in_uv_cache(sys.prefix)


def persistent_install_hint() -> str:
    """What to run instead of uvx, for messages."""
    return (f"Install voiceio permanently, then run setup again:\n"
            f"  {UV_TOOL_INSTALL}    (or: {PIPX_INSTALL})\n"
            f"  voiceio setup")


def ephemeral_message() -> str:
    """Why setup refuses to run from a uvx environment, and what to do."""
    return (f"voiceio is running from a temporary uvx environment ({sys.prefix}).\n"
            "uv deletes it on `uv cache clean`, `uv cache prune` or the next version, so\n"
            "the autostart service and IBus launcher setup writes would stop working.\n"
            + persistent_install_hint())


def _exec_start(unit: Path) -> str | None:
    """The program a systemd unit runs (ExecStart's first word)."""
    import shlex
    try:
        text = unit.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            try:
                words = shlex.split(line.removeprefix("ExecStart="))
            except ValueError:
                return None
            # Strip systemd's special-executable prefixes (-, @, :, +, !).
            return words[0].lstrip("-@:+!") if words else None
    return None


def _launcher_paths(launcher: Path) -> list[str]:
    """Paths the IBus launcher bakes in: VOICEIO_PKG_ROOT, SYSTEM_PY, INSTALL_PY."""
    try:
        text = launcher.read_text()
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        for var in ("VOICEIO_PKG_ROOT", "SYSTEM_PY", "INSTALL_PY"):
            if line.startswith(f'{var}="') and line.endswith('"'):
                found.append(line[len(var) + 2:-1])
    return found


def stale_install_paths() -> list[tuple[str, str, str]]:
    """(what, path, problem) for each file voiceio wrote whose program lives
    in uv's cache or no longer exists. Empty when everything is durable."""
    from voiceio import service
    from voiceio.ibus import install as ibus_install

    refs: list[tuple[str, str]] = []
    for label, unit in (("autostart service", service.SERVICE_PATH),
                        ("learning service", service.LEARN_SERVICE_PATH)):
        if prog := _exec_start(unit):
            refs.append((f"{label} {unit}", prog))
    refs += [(f"IBus launcher {ibus_install.LAUNCHER_PATH}", p)
             for p in _launcher_paths(ibus_install.LAUNCHER_PATH)]
    for name in service.SCRIPT_NAMES:
        link = service.LOCAL_BIN / name
        if link.is_symlink():
            refs.append((f"command link {link}", os.path.join(link.parent, os.readlink(link))))

    stale = []
    for what, path in refs:
        if not os.path.isabs(path):
            continue  # a bare name (`voiceio`) is looked up on PATH at run time
        cached = in_uv_cache(path)
        exists = os.path.exists(path)
        if cached:
            stale.append((what, path, "in uv's cache" if exists
                          else "in uv's cache, already deleted"))
        elif not exists:
            stale.append((what, path, "no longer exists"))
    return stale


def stale_paths_report(stale: list[tuple[str, str, str]]) -> str:
    """Doctor's text for stale_install_paths() findings."""
    lines = ["\nInstall paths that will break (or already did):"]
    lines += [f"  ✗ {what}\n      → {path} ({problem})" for what, path, problem in stale]
    if running_ephemeral():
        lines.append("\n" + ephemeral_message())
    else:
        lines.append("\nRun 'voiceio doctor --fix' or 'voiceio setup' to rewrite them for this "
                     "install.\nIf voiceio came from uvx: " + persistent_install_hint())
    return "\n".join(lines)
