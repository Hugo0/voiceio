"""Base protocol and types for text injection backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from voiceio.backends import ProbeResult

__all__ = ["ProbeResult", "TyperBackend", "StreamingTyper"]


@runtime_checkable
class TyperBackend(Protocol):
    """Protocol for text injection backends."""

    name: str

    def probe(self) -> ProbeResult:
        """Check if this backend can work on the current system."""
        ...

    def type_text(self, text: str) -> None:
        """Type text at the current cursor position."""
        ...

    def delete_chars(self, n: int) -> None:
        """Delete n characters before cursor."""
        ...


@runtime_checkable
class StreamingTyper(TyperBackend, Protocol):
    """Extended protocol for backends that support preedit-based streaming."""

    def update_preedit(self, text: str) -> None:
        """Show text as preedit preview (can be freely replaced)."""
        ...

    def commit_text(self, text: str) -> None:
        """Clear preedit and commit final text."""
        ...

    def clear_preedit(self) -> None:
        """Clear preedit without committing."""
        ...
