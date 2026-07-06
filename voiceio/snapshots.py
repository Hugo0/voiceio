"""Snapshots + rollback for the self-correcting rule lifecycle.

Before the weekly mining run mutates corrections.json / vocabulary.txt, we
copy them into ~/.config/voiceio/snapshots/<ts>-<label>/ so a later drift
audit can restore the exact pre-mining state if the learned changes turn out
to hurt. Every learned rule is probationary; a snapshot is its undo.

Paths are read from `config` at call time so tests can isolate them.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from voiceio import config

log = logging.getLogger(__name__)

KEEP = 8  # retain the last N snapshots, prune older


def _vocab_path() -> Path:
    return config.CONFIG_DIR / "vocabulary.txt"


def _sources() -> list[Path]:
    """Files captured by a snapshot (those that exist are copied)."""
    return [config.CORRECTIONS_PATH, _vocab_path()]


def snapshot(label: str) -> Path:
    """Copy corrections.json + vocabulary.txt into a fresh snapshot dir.

    Returns the snapshot directory. Keeps only the most recent KEEP dirs,
    pruning older ones. Missing source files are simply skipped.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = config.SNAPSHOTS_DIR / f"{ts}-{label}"
    dest.mkdir(parents=True, exist_ok=True)
    for src in _sources():
        if src.exists():
            shutil.copy2(src, dest / src.name)
    _prune()
    return dest


def _prune() -> None:
    """Delete snapshot dirs beyond the KEEP most recent."""
    base = config.SNAPSHOTS_DIR
    if not base.exists():
        return
    dirs = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    for old in dirs[:-KEEP]:
        shutil.rmtree(old, ignore_errors=True)
        log.info("Pruned old snapshot %s", old.name)


def rollback(snapshot_dir: Path) -> bool:
    """Restore corrections.json + vocabulary.txt from a snapshot.

    Returns True if at least one file was restored.
    """
    snapshot_dir = Path(snapshot_dir)
    restored = False
    for dest in _sources():
        src = snapshot_dir / dest.name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            restored = True
    if restored:
        log.info("Rolled back rules from snapshot %s", snapshot_dir.name)
    return restored
