"""What may appear outside the user's own files.

A transcript belongs in history.jsonl and retained audio — files the user owns
and can delete. Logs are not that: the daemon writes DEBUG to
~/.local/state/voiceio/voiceio.log, and under systemd INFO is the journal. So
logs get sizes and outcomes, never the words (a test enforces it).
"""
from __future__ import annotations


def describe_transcript(text: str) -> str:
    """A transcript as it may appear in a log: its length, or that there was none."""
    return f"{len(text.split())} words" if text else "(silence)"
