"""Size limits for text handed to IBus.

libwayland gives every client connection a fixed 4096-byte message buffer.
GNOME Shell relays our preedit and commit strings to the focused application as
single ``zwp_text_input_v3`` events, so a string that does not fit makes the
compositor tear down that application's connection — it dies mid-dictation,
whatever it happens to be. Shell can pack several events into one flush, so the
budget here leaves real headroom rather than sitting just under 4096.

Kept free of any ``gi`` dependency so it can be unit-tested without the IBus
GObject bindings installed.
"""
from __future__ import annotations

MAX_TEXT_BYTES = 2048
TRUNCATION_MARK = "… "


def _is_continuation(byte: int) -> bool:
    return byte & 0xC0 == 0x80


def split_utf8(text: str, limit: int = MAX_TEXT_BYTES) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` UTF-8 bytes.

    The chunks concatenate back to exactly ``text``, so committing them in order
    is indistinguishable to the target application from a single commit. A split
    lands on a space when one falls in the back half of the chunk, and on a
    character boundary otherwise, so no multi-byte character is ever cut in half.
    """
    data = text.encode("utf-8")
    if len(data) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(data):
        end = min(start + limit, len(data))
        if end < len(data):
            space = data.rfind(b" ", start + limit // 2, end)
            if space != -1:
                end = space + 1
            else:
                while end > start and _is_continuation(data[end]):
                    end -= 1
        chunks.append(data[start:end].decode("utf-8"))
        start = end
    return chunks


def tail_utf8(text: str, limit: int = MAX_TEXT_BYTES) -> str:
    """Return the trailing ``limit`` UTF-8 bytes of ``text``, marked if trimmed.

    The preedit preview is one replaceable string rather than an append, so it
    cannot be chunked the way a commit can — it has to be bounded instead. Only
    the preview is shortened; the full text still reaches the application when
    the dictation commits.
    """
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text

    start = len(data) - (limit - len(TRUNCATION_MARK.encode("utf-8")))
    while start < len(data) and _is_continuation(data[start]):
        start += 1
    return TRUNCATION_MARK + data[start:].decode("utf-8")
