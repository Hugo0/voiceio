"""Tests for config loading and migration."""
from __future__ import annotations

import textwrap
from pathlib import Path

from voiceio.config import load, _migrate_v1


def test_load_defaults(tmp_path):
    cfg = load(path=tmp_path / "nonexistent.toml")
    assert cfg.hotkey.key == "ctrl+alt+v"
    assert cfg.model.name == "base"
    assert cfg.audio.prebuffer_secs == 1.0
    assert cfg.health.auto_fallback is True


def test_load_user_config(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""\
        [model]
        name = "small"
        language = "fr"
    """))
    cfg = load(path=config_file)
    assert cfg.model.name == "small"
    assert cfg.model.language == "fr"
    assert cfg.model.device == "auto"  # default preserved


def test_load_ignores_unknown_keys(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""\
        [model]
        name = "small"
        unknown_key = "should be ignored"
    """))
    cfg = load(path=config_file)
    assert cfg.model.name == "small"


def test_migrate_removes_cpu_threads():
    raw = {"model": {"name": "base", "cpu_threads": 4}}
    result = _migrate_v1(raw)
    assert "cpu_threads" not in result["model"]


def test_migrate_xclip_to_clipboard():
    raw = {"output": {"method": "xclip"}}
    result = _migrate_v1(raw)
    assert result["output"]["method"] == "clipboard"


def test_migrate_wl_copy_to_clipboard():
    raw = {"output": {"method": "wl-copy"}}
    result = _migrate_v1(raw)
    assert result["output"]["method"] == "clipboard"


def test_migrate_x11_to_pynput():
    raw = {"hotkey": {"backend": "x11"}}
    result = _migrate_v1(raw)
    assert result["hotkey"]["backend"] == "pynput"
