import sys
from pathlib import Path

from voiceio import service


def test_install_symlinks_relinks_and_drops_legacy(monkeypatch, tmp_path):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    for name in service.SCRIPT_NAMES:
        (venv_bin / name).write_text("#!/bin/sh\n")
    old = tmp_path / "old"
    old.mkdir()
    local_bin = tmp_path / "local-bin"
    local_bin.mkdir()
    (local_bin / "voiceio").symlink_to(old / "voiceio")        # stale install
    (local_bin / "voiceio-doctor").symlink_to(old / "voiceio-doctor")
    monkeypatch.setattr(service, "LOCAL_BIN", local_bin)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(service, "_is_pipx_install", lambda: False)
    monkeypatch.setattr(service, "_local_bin_on_path", lambda: True)

    assert service.install_symlinks() == service.SCRIPT_NAMES
    assert Path(local_bin / "voiceio").resolve() == (venv_bin / "voiceio").resolve()
    assert not (local_bin / "voiceio-doctor").is_symlink()
