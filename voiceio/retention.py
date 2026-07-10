"""Local data retention: per-utterance audio + context for later analysis.

Everything stays under ~/.local/state/voiceio/recordings/ — nothing leaves
the machine. Retained (audio, final text) pairs are what make correction
mining measurable and personal fine-tuning possible later.
"""
from __future__ import annotations

import functools
import json
import logging
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from voiceio import config
from voiceio.config import RECORDINGS_DIR, TRACES_PATH

if TYPE_CHECKING:
    from voiceio.config import DataConfig

log = logging.getLogger(__name__)

_which = functools.lru_cache(maxsize=8)(shutil.which)


def _free_gb(path: Path) -> float:
    """Free space (GB) on the filesystem holding `path`. Inf on failure."""
    try:
        return shutil.disk_usage(path).free / 1024**3
    except OSError:
        return float("inf")


def save_audio(audio: np.ndarray, ts: float, cfg: DataConfig) -> str | None:
    """Persist one utterance as 16kHz mono int16 WAV. Returns the filename
    (relative to the recordings dir) or None if disabled/failed."""
    if not cfg.retain_audio or audio is None or len(audio) == 0:
        return None
    if _free_gb(RECORDINGS_DIR.parent) < cfg.min_free_gb:
        log.warning(
            "Disk has <%.0fGB free — skipping audio retention for this utterance",
            cfg.min_free_gb,
        )
        return None
    name = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts)) + f"-{int(ts * 1000) % 1000:03d}.wav"
    path = RECORDINGS_DIR / name
    try:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        config._chmod(RECORDINGS_DIR, config._SECURE_DIR)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        config._chmod(path, config._SECURE_FILE)
        return name
    except OSError as e:
        log.warning("Failed to save recording: %s", e)
        return None


def prune(cfg: DataConfig) -> None:
    """Delete oldest recordings when total size exceeds the configured cap.

    Also runs the one-shot, idempotent permission migration — this is the
    startup housekeeping call site, so tightening existing files here means
    upgrades from older versions get 0600/0700 without extra wiring.
    """
    try:
        config.harden_permissions()
    except Exception:
        log.debug("Permission hardening failed", exc_info=True)
    if not RECORDINGS_DIR.exists():
        return
    try:
        files = sorted(
            (p for p in RECORDINGS_DIR.glob("*.wav")),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        budget = cfg.max_audio_mb * 1024 * 1024
        # Below the free-disk floor, shrink the budget so retention yields
        # space back instead of holding its full cap on a squeezed disk.
        free = _free_gb(RECORDINGS_DIR.parent)
        if free < cfg.min_free_gb:
            budget = min(budget, total // 2)
            log.warning(
                "Disk has %.1fGB free (<%.0fGB floor) — pruning recordings to %.0fMB",
                free, cfg.min_free_gb, budget / 1024 / 1024,
            )
        while total > budget and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            log.info("Pruned old recording %s (over %dMB cap)", oldest.name, cfg.max_audio_mb)
    except OSError:
        log.debug("Recording prune failed", exc_info=True)


_JSONL_MAX_BYTES = 64 * 1024 * 1024  # per capture file; oldest half dropped


def append_jsonl(path: Path, entry: dict) -> None:
    """Append one JSON line to a 0600 data file. Best-effort, never raises.

    Capture files are bounded: past _JSONL_MAX_BYTES the oldest half of the
    lines is dropped, so intermediate-data capture can never grow unbounded
    on a disk the min_free_gb floor is trying to protect.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        config._chmod(path.parent, config._SECURE_DIR)
        newly_created = not path.exists()
        if not newly_created and path.stat().st_size > _JSONL_MAX_BYTES:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text("".join(lines[len(lines) // 2:]), encoding="utf-8")
            log.info("Trimmed %s to newest %d entries", path.name, len(lines) - len(lines) // 2)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if newly_created:
            config._chmod(path, config._SECURE_FILE)
    except (OSError, TypeError, ValueError) as e:
        log.warning("Failed to write %s: %s", path.name, e)


def save_trace(cfg: DataConfig, entry: dict) -> None:
    """Persist one utterance's per-pass streaming decode trace.

    One line per utterance in streaming_trace.jsonl: pass timings, kinds
    (interim/freeze/final), tail lengths, and raw tail texts — the data
    needed to debug/profile streaming behaviour and to train on interim
    hypotheses later. Linked to history via ts + audio filename.
    """
    if not cfg.capture_intermediates:
        return
    append_jsonl(TRACES_PATH, entry)


def active_window_title() -> str | None:
    """Best-effort title of the focused window (dictation target context).

    Works on X11/XWayland via xdotool; returns None elsewhere. Never raises.
    """
    if not _which("xdotool"):
        return None
    try:
        out = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=1,
        )
        title = out.stdout.strip()
        return title or None
    except (subprocess.TimeoutExpired, OSError):
        return None
