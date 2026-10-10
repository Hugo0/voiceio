"""TyperManager owns the per-recording input-source lease."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

from voiceio.platform import Platform
from voiceio.typers.base import TyperBackend
from voiceio.typers.manager import TyperManager


def _manager(name: str) -> tuple[TyperManager, MagicMock]:
    typer = MagicMock(spec=TyperBackend)
    typer.name = name
    ibus = MagicMock()
    mgr = TyperManager(Platform(os="linux", display_server="wayland", desktop="gnome"), typer,
                       ibus=ibus, method="auto", auto_fallback=True, lock=threading.Lock(),
                       is_idle=lambda: True, state=lambda: "IDLE",
                       shutdown=threading.Event())
    return mgr, ibus


def test_ibus_typer_claims_the_source_for_a_recording():
    mgr, ibus = _manager("ibus")
    mgr.begin(7)
    mgr.end(7)
    ibus.claim.assert_called_once_with(7)
    ibus.release.assert_called_once_with(7)


def test_other_typers_never_claim_ibus():
    mgr, ibus = _manager("ydotool")
    mgr.begin(1)
    ibus.claim.assert_not_called()
    assert not mgr.pastes_from_clipboard


def test_clipboard_typer_pastes_from_clipboard():
    assert _manager("clipboard")[0].pastes_from_clipboard
