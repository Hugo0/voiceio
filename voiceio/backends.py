"""Shared types for all backends (hotkey + typer)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProbeResult:
    """Result of probing whether a backend can work on this system."""
    ok: bool
    reason: str = ""
    fix_hint: str = ""
    fix_cmd: list[str] = field(default_factory=list)  # auto-fixable command
