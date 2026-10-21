"""Mic input gain that follows your recordings ([audio] auto_gain).

Clipping can't be repaired after the ADC, so the fix is the input volume.
After each recording:

  clipped (flat-top runs in >= 0.5% of samples)
      → lower the default source's volume by a fifth, and remember that the
        old level clipped: it is never raised back to it
  three spoken recordings in a row that peaked below 0.1
      → raise it by a fifth, staying under the level that clipped

Only the default PipeWire/Pulse source is touched (wpctl, else pactl), and
never below 10% or above 100%. A device chosen in [audio] device is the
user's to set.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from voiceio import quiet_audio
from voiceio.quiet_audio import _Caller

log = logging.getLogger(__name__)

CLIP_RATIO = 0.005
QUIET_PEAK = 0.1
QUIET_STREAK = 3
MIN_SPEECH_SECS = 3.0
STEP = 0.8
FLOOR, CEILING = 0.10, 1.0


class _WpctlSource:
    def __init__(self, call: _Caller):
        self._call = call

    def get(self) -> float | None:
        m = re.match(r"Volume: ([\d.]+)", self._call("wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@") or "")
        return float(m[1]) if m else None

    def set(self, volume: float) -> bool:
        return self._call("wpctl", "set-volume", "@DEFAULT_AUDIO_SOURCE@", f"{volume:.2f}") is not None


class _PactlSource:
    def __init__(self, call: _Caller):
        self._call = call

    def get(self) -> float | None:
        m = re.search(r"(\d+)%", self._call("pactl", "get-source-volume", "@DEFAULT_SOURCE@") or "")
        return int(m[1]) / 100 if m else None

    def set(self, volume: float) -> bool:
        return self._call("pactl", "set-source-volume", "@DEFAULT_SOURCE@",
                          f"{round(volume * 100)}%") is not None


@dataclass
class Change:
    old: float
    new: float
    reason: str   # "clipping" | "quiet"


class MicGain:
    def __init__(self, enabled: bool = True, device: str = "default", run=None):
        call = _Caller(run)
        self._source = None
        if enabled and device in ("", "default"):
            # Through the module: tests swap quiet_audio._which/_run for guards.
            if quiet_audio._which("wpctl"):
                self._source = _WpctlSource(call)
            elif quiet_audio._which("pactl"):
                self._source = _PactlSource(call)
        self._clipped_at: float | None = None   # lowest level known to clip
        self._quiet = 0

    @property
    def active(self) -> bool:
        return self._source is not None

    def after_recording(self, peak: float, clip_ratio: float, secs: float) -> Change | None:
        """Adjust after one recording; the change made, if any. Never raises."""
        if self._source is None:
            return None
        try:
            return self._adjust(peak, clip_ratio, secs)
        except Exception:  # noqa: BLE001 — a gain tweak must never break dictation
            log.debug("mic gain adjustment failed", exc_info=True)
            return None

    def _adjust(self, peak: float, clip_ratio: float, secs: float) -> Change | None:
        if clip_ratio >= CLIP_RATIO:
            self._quiet = 0
            old = self._source.get()
            if old is None:
                return None
            self._clipped_at = old if self._clipped_at is None else min(self._clipped_at, old)
            new = max(FLOOR, round(old * STEP, 2))
            return self._set(old, new, "clipping")
        if secs < MIN_SPEECH_SECS or peak <= 0:
            return None   # too short to judge, or a dead/muted mic: not a gain problem
        if peak >= QUIET_PEAK:
            self._quiet = 0
            return None
        self._quiet += 1
        if self._quiet < QUIET_STREAK:
            return None
        self._quiet = 0
        old = self._source.get()
        if old is None:
            return None
        top = CEILING if self._clipped_at is None else min(CEILING, round(self._clipped_at * STEP, 2))
        new = min(top, round(old / STEP, 2))
        return self._set(old, new, "quiet")

    def _set(self, old: float, new: float, reason: str) -> Change | None:
        if abs(new - old) < 0.005 or not self._source.set(new):
            return None
        log.info("Mic gain %s: input volume %.0f%% → %.0f%%", reason, old * 100, new * 100)
        return Change(old, new, reason)
