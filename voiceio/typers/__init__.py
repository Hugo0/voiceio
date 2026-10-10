"""Text injection backends."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.config import Config
    from voiceio.platform import Platform
    from voiceio.typers.base import TyperBackend


def typer_options(cfg: Config) -> dict:
    """Config-driven backend options, passed through chain.select()/resolve()."""
    return {
        "paste_keys": cfg.output.paste_keys,
        # Never restore over a transcript put on the clipboard on purpose.
        "restore_clipboard": (cfg.output.restore_clipboard
                              and cfg.output.copy_to_clipboard == "off"
                              and not cfg.feedback.notify_clipboard),
    }


def create_typer_backend(name: str, platform: Platform, **kwargs) -> TyperBackend:
    """Create a typer backend by name."""
    if name == "xdotool":
        from voiceio.typers.xdotool import XdotoolTyper
        return XdotoolTyper()
    if name == "ydotool":
        from voiceio.typers.ydotool import YdotoolTyper
        return YdotoolTyper()
    if name == "wtype":
        from voiceio.typers.wtype import WtypeTyper
        return WtypeTyper()
    if name == "clipboard":
        from voiceio.typers.clipboard import ClipboardTyper
        return ClipboardTyper(platform, **kwargs)
    if name == "pynput":
        from voiceio.typers.pynput_type import PynputTyper
        return PynputTyper()
    if name == "fcitx-bridge":
        from voiceio.typers.fcitx_bridge import FcitxBridgeTyper
        return FcitxBridgeTyper()
    if name == "ibus":
        from voiceio.typers.ibus import IBusTyper
        return IBusTyper(platform, **kwargs)
    raise ValueError(f"Unknown typer backend: {name}")
