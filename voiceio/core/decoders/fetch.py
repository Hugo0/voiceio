"""Download the models voiceio fetches itself (sherpa-onnx release tarballs).

From the sherpa-onnx GitHub release, never Hugging Face: on some networks
IPv6 to huggingface.co blackholes and a connect hangs forever. Every byte is
checked against the catalog's sha256 before anything is unpacked, and a model
only appears under its final name once it is complete.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voiceio.core.decoders.catalog import ModelInfo, info, pins

log = logging.getLogger(__name__)

_REQUIRED = ("encoder", "decoder", "joiner")
_TIMEOUT = 30  # per socket operation, not for the whole download
_BLOCK = 1 << 20


def models_dir() -> Path:
    from voiceio.config import MODELS_DIR
    return MODELS_DIR


def model_path(name: str) -> Path:
    return models_dir() / name


def is_complete(path: Path) -> bool:
    """A sherpa-onnx transducer directory: encoder, decoder, joiner, tokens."""
    return (path / "tokens.txt").is_file() and all(
        any(path.glob(f"{stem}*.onnx")) for stem in _REQUIRED)


def is_downloaded(name: str) -> bool:
    return is_complete(model_path(name))


def _unpack(archive: Path, dest: Path) -> None:
    """Extract the tarball's single top directory as `dest`, without its
    sample WAVs. Members must stay inside it (no links, no absolute paths)."""
    with tarfile.open(archive, "r:*") as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) < 2 or "test_wavs" in parts:
                continue
            if m.issym() or m.islnk() or Path(m.name).is_absolute() or ".." in parts:
                raise OSError(f"unsafe path in {archive.name}: {m.name}")
            m.name = str(Path(*parts[1:]))
            members.append(m)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, members=members, **kwargs)


def download(name: str, *, progress: Callable[[int, int], None] | None = None) -> Path:
    """Fetch, verify and unpack catalog model `name`; returns its directory.
    Already downloaded: returns at once. Raises OSError on any failure,
    leaving nothing half-written under the model's name."""
    m: ModelInfo | None = info(name)
    if m is None or not m.url:
        raise OSError(f"{name} is not a model voiceio downloads itself")
    return download_archive(name, m.url, m.sha256, m.size_mb, is_complete,
                            what="encoder/decoder/joiner/tokens.txt", progress=progress)


ISSUES = "https://github.com/Hugo0/voiceio/issues"


def download_archive(name: str, url: str, sha256: str | tuple[str, ...], size_mb: int,
                     complete: Callable[[Path], bool], *, what: str,
                     progress: Callable[[int, int], None] | None = None) -> Path:
    """Fetch, verify and unpack a sherpa-onnx release tarball as
    `models_dir()/name`. `sha256` is one known-good digest or a tuple of
    them; the archive must match one. `complete` says whether a directory
    holds the whole model (`what` names its files for the error)."""
    dest = model_path(name)
    if complete(dest):
        return dest
    known = pins(sha256)
    if not known:
        raise OSError(f"{name}: no pinned sha256 — not downloading")
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    part = root / f".{name}.{os.getpid()}.part"
    staging = root / f".{name}.{os.getpid()}.unpack"
    log.info("Downloading %s (%d MB) from %s", name, size_mb, url)
    try:
        digest = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": "voiceio"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or size_mb * 1_000_000)
            done = 0
            while block := resp.read(_BLOCK):
                out.write(block)
                digest.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
        if digest.hexdigest() not in known:
            want = " or ".join(f"{h[:12]}…" for h in known)
            raise OSError(f"{name}: checksum mismatch (got {digest.hexdigest()[:12]}…, "
                          f"want {want}) — download again; if it keeps failing, upstream "
                          f"may have re-uploaded the archive: report it at {ISSUES}")
        _unpack(part, staging)
        if not complete(staging):
            raise OSError(f"{name}: the archive has no {what}")
        shutil.rmtree(dest, ignore_errors=True)   # an incomplete earlier copy
        os.replace(staging, dest)
    except OSError:
        raise
    except Exception as e:  # noqa: BLE001 — tarfile/http errors: one failure type
        raise OSError(f"{name}: download failed: {e}") from e
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    log.info("Model %s ready in %s", name, dest)
    return dest


def print_progress(label: str) -> Callable[[int, int], None]:
    """A progress callback that redraws one terminal line."""
    import sys
    last = [-1]

    def show(done: int, total: int) -> None:
        pct = min(100, done * 100 // max(1, total))
        if pct != last[0]:
            last[0] = pct
            sys.stderr.write(f"\r  {label}: {done / 1e6:5.0f} / {total / 1e6:.0f} MB ({pct}%)")
            if done >= total:
                sys.stderr.write("\n")
            sys.stderr.flush()
    return show


# ── sherpa-onnx ──────────────────────────────────────────────────────────
# Parakeet's biasing needs the NeMo TDT beam-search fix (k2-fsa/sherpa-onnx
# #4084) and pre-tokenized hotwords (#4088), both released upstream in 2.0.2.
SHERPA_REQUIREMENT = "sherpa-onnx>=2.0.2"
# Before that release: Hugo0/sherpa-onnx branch `voiceio-patched`, built by the
# voiceio repo's "sherpa-wheels" workflow into this release, with a SHA256SUMS
# file (`voiceio models sherpa --patched`).
SHERPA_WHEELS = "https://github.com/Hugo0/voiceio/releases/download/sherpa-onnx-voiceio"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "voiceio"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def patched_sherpa_wheel(sums: str, version: tuple[int, int] | None = None,
                         machine: str | None = None) -> tuple[str, str] | None:
    """(file name, sha256) of the wheel for this Python and CPU in a
    SHA256SUMS listing, or None."""
    import platform
    import sys
    major, minor = version or sys.version_info[:2]
    machine = machine or platform.machine()
    tag = f"-cp{major}{minor}-cp{major}{minor}-"
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith(".whl") and tag in parts[1] \
                and parts[1].endswith(f"_{machine}.whl"):
            return parts[1].lstrip("*"), parts[0]
    return None


def install_patched_sherpa(base_url: str = SHERPA_WHEELS) -> str:
    """Download, verify and install the patched sherpa-onnx wheel into this
    Python. Returns the wheel's name; raises OSError with what went wrong."""
    import sys
    import tempfile
    try:
        sums = _get(f"{base_url}/SHA256SUMS").decode()
    except Exception as e:  # noqa: BLE001
        raise OSError(f"could not reach {base_url}: {e}") from e
    found = patched_sherpa_wheel(sums)
    if found is None:
        import platform
        raise OSError(f"no patched sherpa-onnx wheel for Python "
                      f"{sys.version_info.major}.{sys.version_info.minor} on {platform.machine()}"
                      f" — build it from Hugo0/sherpa-onnx@voiceio-patched (python setup.py "
                      f"bdist_wheel) and pip install it")
    wheel, sha = found
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / wheel
        try:
            data = _get(f"{base_url}/{wheel}")
        except Exception as e:  # noqa: BLE001
            raise OSError(f"download of {wheel} failed: {e}") from e
        if hashlib.sha256(data).hexdigest() != sha:
            raise OSError(f"{wheel}: checksum mismatch — not installing")
        path.write_bytes(data)
        # Same version number as upstream's release: force the swap.
        _pip_install([str(path)], reinstall=True)
    _refresh_sherpa()
    return wheel


def install_sherpa(*, quiet: bool = False) -> None:
    """Install or upgrade sherpa-onnx from PyPI to a release with the
    Parakeet fixes. Raises OSError with what went wrong. `quiet`: keep pip's
    output off the terminal (the setup wizard shows a spinner instead)."""
    _pip_install(["--upgrade", SHERPA_REQUIREMENT], quiet=quiet)
    _refresh_sherpa()


def _refresh_sherpa() -> None:
    """Let this process see a sherpa-onnx installed after it started: the
    import system's directory caches and the cached feature detection."""
    import importlib

    from voiceio.core.decoders.sherpa import features
    importlib.invalidate_caches()
    features.cache_clear()


# Platforms PyPI has sherpa-onnx wheels for that voiceio offers to install on.
_SHERPA_MACHINES = frozenset({"x86_64", "amd64", "aarch64", "arm64"})


def can_install_sherpa() -> bool:
    """Whether `install_sherpa` can work here: a pip or uv to run, a wheel for
    this CPU, and an environment this user may write to that will still be
    there tomorrow. Not a distro package (/usr/lib/voiceio is root's), not the
    Nix store (read-only), not a uvx environment (uv may delete it)."""
    import platform
    import sys
    import sysconfig

    from voiceio.install_env import running_ephemeral
    from voiceio.service import packaged_by
    if packaged_by() or sys.prefix.startswith("/nix/store/") or running_ephemeral():
        return False
    if platform.machine().lower() not in _SHERPA_MACHINES or _pip_command() is None:
        return False
    paths = sysconfig.get_paths()
    return all(os.access(paths[k], os.W_OK) for k in ("purelib", "platlib"))


def _pip_command(reinstall: bool = False) -> list[str] | None:
    """The pip (or uv) command that installs into this Python, or None."""
    import importlib.util
    import shutil as _shutil
    import sys
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"] + \
            (["--force-reinstall", "--no-deps"] if reinstall else [])
    if uv := _shutil.which("uv"):
        return [uv, "pip", "install", "--python", sys.executable] + \
            (["--reinstall", "--no-deps"] if reinstall else [])
    return None


def _pip_install(args: list[str], *, reinstall: bool = False, quiet: bool = False) -> None:
    """pip (or uv) install into this Python; OSError on failure."""
    import subprocess
    base = _pip_command(reinstall)
    if base is None:
        raise OSError(f"neither pip nor uv is available to install {args[-1]}")
    cmd = base + args
    res = subprocess.run(cmd, capture_output=quiet, text=quiet)
    if res.returncode != 0:
        tail = (res.stderr or "").strip().splitlines()[-1:] if quiet else []
        raise OSError(f"installing {args[-1]} failed (exit {res.returncode})"
                      + (f": {tail[0]}" if tail else f": {' '.join(cmd)}"))
