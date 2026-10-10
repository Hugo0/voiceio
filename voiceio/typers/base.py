"""Base protocol and types for text injection backends."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from voiceio.backends import ProbeResult
from voiceio.typers import modifier_guard

__all__ = ["ProbeResult", "TyperBackend", "StreamingTyper", "KeystrokeTyper"]


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


class KeystrokeTyper:
    """Base of the typers that synthesize key events, unlike IBus and fcitx
    which commit text. A modifier still held from the stop hotkey would turn
    those events into shortcuts, so every type and delete waits it out first.
    """

    def type_text(self, text: str) -> None:
        if text:
            modifier_guard.wait_for_release()
            self._type(text)

    def delete_chars(self, n: int) -> None:
        if n > 0:
            modifier_guard.wait_for_release()
            self._delete(n)

    def _type(self, text: str) -> None:
        raise NotImplementedError

    def _delete(self, n: int) -> None:
        raise NotImplementedError
