"""Shared test fixtures."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Sandbox HOME and the XDG dirs *before* voiceio is imported: config paths are
# resolved at import time, and a test that forgets to patch one (or a daemon
# thread outliving its monkeypatch) must land here, never in the user's real
# history/traces. Caches stay real so model loads don't re-download.
_REAL_HOME = Path.home()
os.environ.setdefault("XDG_CACHE_HOME", str(_REAL_HOME / ".cache"))
os.environ.setdefault("HF_HOME", str(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"))
_SANDBOX = Path(tempfile.mkdtemp(prefix="voiceio-tests-"))
os.environ["HOME"] = str(_SANDBOX)
# `voiceio learn` re-execs itself into an idle systemd scope; never from tests.
os.environ["VOICEIO_IDLE_SCOPE"] = "1"
for _var, _sub in (
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_STATE_HOME", ".local/state"),
    ("XDG_DATA_HOME", ".local/share"),
):
    os.environ[_var] = str(_SANDBOX / _sub)

import pytest  # noqa: E402

from voiceio import config  # noqa: E402
from voiceio.platform import Platform  # noqa: E402


def pytest_sessionstart(session):
    for path in (config.CONFIG_DIR, config.LOG_DIR):
        assert _SANDBOX in path.parents, f"test state escaped the sandbox: {path}"


@pytest.fixture(autouse=True)
def _spare_the_live_engine(monkeypatch):
    """Engine lookup sees only processes this test run started. The real one
    SIGTERMed the user's running IBus engine mid-dictation on every run."""
    from voiceio import ibus

    real = ibus.engine_pids

    def ours() -> list[int]:
        def parent(pid: int) -> int:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except OSError:
                return -1
            return int(stat.rsplit(")", 1)[1].split()[1])
        return [p for p in real() if parent(p) == os.getpid()]

    monkeypatch.setattr(ibus, "engine_pids", ours)


@pytest.fixture(autouse=True)
def _quiet_desktop(monkeypatch):
    """No real notifications or sounds: tests drive the daemon's failure paths
    ("microphone unavailable"), which popped up on the developer's desktop."""
    from voiceio import feedback
    monkeypatch.setattr(feedback, "_send_notification", lambda *a, **k: None)
    # Notifications with buttons go straight to notify-send: degrade them to
    # the plain path above (tests of notify_actions itself fake subprocess).
    monkeypatch.setattr(feedback, "_notify_send_has_actions", lambda: False)
    monkeypatch.setattr(feedback, "_play", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_keyboards(monkeypatch):
    """The modifier guard reads the developer's real keyboards: holding Ctrl
    while tests run would slow every typer test by its timeout."""
    from voiceio.typers import modifier_guard
    monkeypatch.setattr(modifier_guard, "_open_keyboards", lambda: [])


@pytest.fixture(autouse=True)
def _isolate_user_state(tmp_path, monkeypatch):
    """Keep tests out of the user's real state dir (recordings, history)."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))  # voiceio.learn
    monkeypatch.setattr("voiceio.retention.RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr("voiceio.config.RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr("voiceio.config.HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr("voiceio.history.HISTORY_PATH", tmp_path / "history.jsonl")
    # Corrections, vocabulary and config out of the user's real dirs.
    monkeypatch.setattr("voiceio.config.STATE_PATH", tmp_path / "voiceio.state")
    monkeypatch.setattr("voiceio.config.CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr("voiceio.config.CONFIG_PATH", tmp_path / "config" / "config.toml")
    monkeypatch.setattr("voiceio.config.CONSENT_PATH", tmp_path / "config" / "consent.json")
    monkeypatch.setattr("voiceio.config.CORRECTIONS_PATH", tmp_path / "config" / "corrections.json")
    monkeypatch.setattr("voiceio.config.FLAGGED_PATH", tmp_path / "config" / "flagged.txt")
    monkeypatch.setattr("voiceio.config.LOG_DIR", tmp_path / "state")
    monkeypatch.setattr(
        "voiceio.corrections.CORRECTIONS_PATH", tmp_path / "config" / "corrections.json",
    )
    monkeypatch.setattr("voiceio.corrections.FLAGGED_PATH", tmp_path / "config" / "flagged.txt")
    monkeypatch.setattr("voiceio.config.TRACES_PATH", tmp_path / "state" / "traces.jsonl")
    monkeypatch.setattr("voiceio.retention.TRACES_PATH", tmp_path / "state" / "traces.jsonl")
    monkeypatch.setattr(
        "voiceio.config.POSTCORRECT_PAIRS_PATH", tmp_path / "state" / "postcorrect_pairs.jsonl",
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


def pytest_sessionfinish(session, exitstatus):
    import shutil
    shutil.rmtree(_SANDBOX, ignore_errors=True)
