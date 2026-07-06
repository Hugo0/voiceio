"""Shared test fixtures."""
from __future__ import annotations

import pytest

from voiceio.platform import Platform


@pytest.fixture(autouse=True)
def _isolate_user_state(tmp_path, monkeypatch):
    """Keep tests out of the user's real state dir (recordings, history)."""
    monkeypatch.setattr("voiceio.retention.RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr("voiceio.config.HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr("voiceio.history.HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(
        "voiceio.autocorrect_state.STATE_PATH", tmp_path / "autocorrect_state.json",
    )
    # Self-correcting rule lifecycle: keep fire log, metrics, snapshots, audit
    # state, corrections + vocabulary out of the user's real dirs.
    monkeypatch.setattr(
        "voiceio.config.CORRECTIONS_AUDIT_PATH", tmp_path / "corrections_audit.jsonl",
    )
    monkeypatch.setattr("voiceio.config.METRICS_PATH", tmp_path / "metrics.jsonl")
    monkeypatch.setattr("voiceio.config.SNAPSHOTS_DIR", tmp_path / "snapshots")
    monkeypatch.setattr("voiceio.config.AUDIT_STATE_PATH", tmp_path / "audit_state.json")
    monkeypatch.setattr("voiceio.config.CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr("voiceio.config.CORRECTIONS_PATH", tmp_path / "config" / "corrections.json")
    monkeypatch.setattr(
        "voiceio.corrections.CORRECTIONS_PATH", tmp_path / "config" / "corrections.json",
    )


@pytest.fixture
def linux_x11():
    return Platform(
        os="linux", display_server="x11", desktop="unknown",
        has_xdotool=True, has_xclip=True,
    )


@pytest.fixture
def linux_wayland_gnome():
    return Platform(
        os="linux", display_server="wayland", desktop="gnome",
        has_ydotool=True, has_wl_copy=True,
        has_input_group=True, has_uinput_access=True,
    )


@pytest.fixture
def linux_wayland_sway():
    return Platform(
        os="linux", display_server="wayland", desktop="sway",
        has_wtype=True, has_ydotool=True, has_wl_copy=True,
        has_input_group=True,
    )


@pytest.fixture
def macos():
    return Platform(os="darwin", display_server="quartz", desktop="macos")


@pytest.fixture
def windows():
    return Platform(os="windows", display_server="win32", desktop="windows")
