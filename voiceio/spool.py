"""Keep the audio of an open dictation where a crash can't take it.

A recording lives in the daemon's memory until it is decoded and typed. If
the daemon is killed in between (OOM, a frozen desktop, a SIGKILL), those
words are gone. So while recording, the samples also go to a file in
$XDG_RUNTIME_DIR (RAM, private to the user, cleared at logout), written by a
thread of its own so the audio callback never touches the disk. The file is
deleted once the dictation is done. One left behind by a dead daemon is an
orphan: the next start decodes it into history and says so.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_NAME = re.compile(r"^recording-(\d+)-(\d+)-(\d+)\.f32$")   # pid, rid, sample rate
MIN_SECS = 1.0   # shorter orphans aren't worth a notification


def spool_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/voiceio-{os.getuid()}"
    return Path(base) / "voiceio-spool"


class Spool:
    """The current process's recordings, one file each."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or spool_dir()
        self._queue: queue.Queue = queue.Queue()
        self._files: dict[int, object] = {}
        threading.Thread(target=self._run, name="spool", daemon=True).start()

    def path(self, rid: int, sample_rate: int) -> Path:
        return self._dir / f"recording-{os.getpid()}-{rid}-{sample_rate}.f32"

    def begin(self, rid: int, sample_rate: int, prebuffer: np.ndarray | None = None) -> None:
        self._queue.put(("begin", rid, sample_rate))
        if prebuffer is not None and len(prebuffer):
            self.write(rid, prebuffer)

    def write(self, rid: int, chunk: np.ndarray) -> None:
        """From the audio callback: hand the chunk over, never block."""
        self._queue.put(("write", rid, chunk))

    def finish(self, rid: int) -> None:
        """The dictation is decoded and delivered (or cancelled): forget it."""
        self._queue.put(("finish", rid, None))

    def flush(self, timeout: float = 2.0) -> None:
        done = threading.Event()
        self._queue.put(("flush", 0, done))
        done.wait(timeout)

    def _run(self) -> None:
        paths: dict[int, Path] = {}
        while True:
            op, rid, arg = self._queue.get()
            try:
                if op == "begin":
                    self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                    p = self.path(rid, arg)
                    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    self._files[rid], paths[rid] = os.fdopen(fd, "wb", buffering=0), p
                elif op == "write" and rid in self._files:
                    self._files[rid].write(np.ascontiguousarray(arg, dtype=np.float32).tobytes())
                elif op == "finish":
                    f = self._files.pop(rid, None)
                    if f is not None:
                        f.close()
                    p = paths.pop(rid, None)
                    if p is not None:
                        p.unlink(missing_ok=True)
                elif op == "flush":
                    arg.set()
            except OSError as e:   # a full or missing runtime dir: recording goes on
                log.warning("spool: %s failed: %s", op, e)
                self._files.pop(rid, None)


def orphans(directory: Path | None = None) -> list[tuple[Path, np.ndarray, int]]:
    """Recordings a dead daemon left behind: (path, audio, sample_rate).
    Files of a process still alive are not orphans."""
    d = directory or spool_dir()
    out = []
    for p in sorted(d.glob("recording-*.f32")) if d.is_dir() else []:
        m = _NAME.match(p.name)
        if m is None:
            continue
        pid, rate = int(m.group(1)), int(m.group(3))
        if pid == os.getpid() or _alive(pid):
            continue
        audio = np.fromfile(p, dtype=np.float32)
        if len(audio) < MIN_SECS * rate:
            p.unlink(missing_ok=True)
            continue
        out.append((p, audio, rate))
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
