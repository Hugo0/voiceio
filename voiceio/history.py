"""Transcription history: append-only JSONL log of all dictated text."""
from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path

from voiceio import config
from voiceio.config import HISTORY_PATH, HistoryConfig

log = logging.getLogger(__name__)

# Auto-prune roughly every N appends so retention caps are enforced during a
# long-running session, not only on daemon start.
_PRUNE_EVERY = 100
_append_count = 0


def _history_cfg(cfg: HistoryConfig | None) -> HistoryConfig:
    return cfg if cfg is not None else config.load().history


def append(
    text: str,
    path: Path | None = None,
    *,
    raw: str | None = None,
    segments: list[dict] | None = None,
    duration: float | None = None,
    extra: dict | None = None,
    cfg: HistoryConfig | None = None,
) -> None:
    """Append a transcription entry to the history file.

    `text` is the final post-processed output. `raw` is the unprocessed
    Whisper text and `segments` its per-segment confidence metadata — kept
    so correction efficacy can be measured against ground truth instead of
    the pipeline's own output.

    Honours `[history].enabled` (skips writing when False) and enforces the
    retention caps roughly every ``_PRUNE_EVERY`` appends.
    """
    global _append_count
    if not text or not text.strip():
        return
    hc = _history_cfg(cfg)
    if not hc.enabled:
        return
    p = path or HISTORY_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        config._chmod(p.parent, config._SECURE_DIR)
        entry: dict = {"ts": time.time(), "text": text.strip()}
        if raw is not None and raw.strip() != entry["text"]:
            entry["raw"] = raw.strip()
        if segments:
            entry["segments"] = segments
        if duration is not None:
            entry["duration"] = round(duration, 2)
        if extra:
            for k, v in extra.items():
                if v is not None and k not in entry:
                    entry[k] = v
        newly_created = not p.exists()
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if newly_created:
            config._chmod(p, config._SECURE_FILE)
    except OSError as e:
        log.warning("Failed to write history: %s", e)
        return

    _append_count += 1
    if (hc.max_entries or hc.max_age_days) and _append_count % _PRUNE_EVERY == 0:
        prune(hc, path=p)


def prune(cfg: HistoryConfig | None = None, path: Path | None = None) -> int:
    """Enforce `[history]` retention caps. Returns entries removed.

    Drops entries older than ``max_age_days`` (when > 0) and keeps only the
    newest ``max_entries`` (when > 0). Rewrites the file in place. Never
    raises; best-effort.
    """
    hc = _history_cfg(cfg)
    p = path or HISTORY_PATH
    if not p.exists():
        return 0
    if not (hc.max_entries or hc.max_age_days):
        return 0
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0

    kept: list[str] = []
    parsed: list[tuple[str, dict]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append((line, json.loads(line)))
        except json.JSONDecodeError:
            continue

    removed = 0
    if hc.max_age_days > 0:
        cutoff = time.time() - hc.max_age_days * 86400
        filtered = [(ln, e) for ln, e in parsed if e.get("ts", 0) >= cutoff]
        removed += len(parsed) - len(filtered)
        parsed = filtered

    if hc.max_entries > 0 and len(parsed) > hc.max_entries:
        removed += len(parsed) - hc.max_entries
        parsed = parsed[-hc.max_entries:]  # keep newest (file is chronological)

    if removed == 0:
        return 0

    kept = [ln for ln, _ in parsed]
    try:
        text = "\n".join(kept) + ("\n" if kept else "")
        p.write_text(text, encoding="utf-8")
        _chmod_secure(p)
        log.info("Pruned %d history entries (retention policy)", removed)
    except OSError as e:
        log.warning("History prune failed: %s", e)
        return 0
    return removed


def _chmod_secure(p: Path) -> None:
    if os.name == "posix":
        try:
            if stat.S_IMODE(p.stat().st_mode) != config._SECURE_FILE:
                p.chmod(config._SECURE_FILE)
        except OSError:
            pass


def read(path: Path | None = None, limit: int = 0) -> list[dict]:
    """Read history entries. Returns newest-first. limit=0 means all."""
    p = path or HISTORY_PATH
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        entries.reverse()  # newest first
        if limit > 0:
            entries = entries[:limit]
        return entries
    except OSError:
        return []


def search(query: str, path: Path | None = None) -> list[dict]:
    """Search history entries by substring (case-insensitive)."""
    query_lower = query.lower()
    return [e for e in read(path) if query_lower in e.get("text", "").lower()]


def clear(path: Path | None = None) -> None:
    """Clear all history."""
    p = path or HISTORY_PATH
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
