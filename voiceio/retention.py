"""Local data retention: per-utterance audio + context for later analysis.

Everything stays under ~/.local/state/voiceio/recordings/ — nothing leaves
the machine. Retained (audio, final text) pairs are what make correction
mining measurable and personal fine-tuning possible later.
"""
from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import time
import wave
from typing import TYPE_CHECKING

import numpy as np

from voiceio import config
from voiceio.config import RECORDINGS_DIR

if TYPE_CHECKING:
    from voiceio.config import DataConfig

log = logging.getLogger(__name__)

_which = functools.lru_cache(maxsize=8)(shutil.which)


def save_audio(audio: np.ndarray, ts: float, cfg: DataConfig) -> str | None:
    """Persist one utterance as 16kHz mono int16 WAV. Returns the filename
    (relative to the recordings dir) or None if disabled/failed."""
    if not cfg.retain_audio or audio is None or len(audio) == 0:
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
        while total > budget and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            log.info("Pruned old recording %s (over %dMB cap)", oldest.name, cfg.max_audio_mb)
    except OSError:
        log.debug("Recording prune failed", exc_info=True)


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
