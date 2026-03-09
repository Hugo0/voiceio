"""Base protocol and types for hotkey backends."""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from voiceio.backends import ProbeResult

__all__ = ["ProbeResult", "HotkeyBackend"]


@runtime_checkable
class HotkeyBackend(Protocol):
    """Protocol for hotkey detection backends."""

    name: str

    def probe(self) -> ProbeResult:
        """Check if this backend can work on the current system."""
        ...

    def start(self, combo: str, on_trigger: Callable[[], None]) -> None:
        """Start listening for the hotkey combo. Non-blocking."""
        ...

    def stop(self) -> None:
        """Stop listening. Idempotent."""
        ...
