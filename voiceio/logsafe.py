"""Log-safe summaries of things the user said.

voiceio's promise is that speech is transcribed on-device and the transcript
lives only in files the user owns and can delete — history.jsonl, retained
audio, the corrections dictionary. A log line is none of those: on a desktop it
goes to `~/.local/state/voiceio/`, and under systemd it goes to the journal, on
disk, indefinitely, for every utterance.

So logs record *lengths and outcomes, never the words*. Latency debugging —
which is what these lines are actually for — needs the seconds and the size,
not the sentence.
"""
from __future__ import annotations


def summary(text: str) -> str:
    """`text` as it may appear in a log: its size, or that there was none."""
    return f"{len(text)} chars" if text else "(silence)"
