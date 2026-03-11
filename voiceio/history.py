"""Transcription history: append-only JSONL log of all dictated text."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from voiceio.config import HISTORY_PATH

log = logging.getLogger(__name__)


def append(text: str, path: Path | None = None) -> None:
    """Append a transcription entry to the history file."""
    if not text or not text.strip():
        return
    p = path or HISTORY_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.time(), "text": text.strip()}
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("Failed to write history: %s", e)


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
