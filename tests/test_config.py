"""Tests for config loading and migration."""
from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

from voiceio.config import load, _migrate_v1


def test_load_defaults(tmp_path):
    cfg = load(path=tmp_path / "nonexistent.toml")
    assert cfg.hotkey.key == "ctrl+alt+v"
    assert cfg.model.name == "small"
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


def test_migrate_keeps_cpu_threads():
    raw = {"model": {"name": "base", "cpu_threads": 4}}
    result = _migrate_v1(raw)
    assert result["model"]["cpu_threads"] == 4


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


def test_copy_to_clipboard_default_and_override(tmp_path):
    cfg = load(path=tmp_path / "nonexistent.toml")
    assert cfg.output.copy_to_clipboard == "final"

    config_file = tmp_path / "config.toml"
    config_file.write_text(textwrap.dedent("""\
        [output]
        copy_to_clipboard = "live"
    """))
    cfg = load(path=config_file)
    assert cfg.output.copy_to_clipboard == "live"


def test_set_value_preserves_comments_and_sections(tmp_path):
    from voiceio.config import set_value
    p = tmp_path / "config.toml"
    p.write_text('# my notes\n[model]\nname = "small"  # fast\n#language = "de"\n\n[tts]\nenabled = true\n')
    set_value("model", "name", "/home/u/models/run1/ct2", path=p)
    set_value("model", "language", "en", path=p)
    set_value("output", "streaming_freeze_secs", 8.0, path=p)
    set_value("tts", "enabled", False, path=p)
    text = p.read_text()
    assert text.startswith("# my notes\n")
    data = tomllib.loads(text)
    assert data["model"] == {"name": "/home/u/models/run1/ct2", "language": "en"}
    assert data["output"]["streaming_freeze_secs"] == 8.0
    assert data["tts"]["enabled"] is False


def test_example_config_matches_defaults():
    """config.example.toml documents every option with its real default."""
    import dataclasses
    from pathlib import Path

    from voiceio.config import Config, ModelConfig
    example = tomllib.loads(
        (Path(__file__).parent.parent / "config.example.toml").read_text())
    cfg = Config()
    for sec in dataclasses.fields(cfg):
        defaults = getattr(cfg, sec.name)
        assert sec.name in example, f"[{sec.name}] missing from example"
        for f in dataclasses.fields(defaults):
            assert f.name in example[sec.name], f"{sec.name}.{f.name} missing"
            if (sec.name, f.name) == ("model", "name"):
                # "auto" resolves to the default decoder on construction
                assert example["model"]["name"] == "auto"
                assert ModelConfig(name="auto").name == defaults.name
                continue
            assert example[sec.name][f.name] == getattr(defaults, f.name), \
                f"{sec.name}.{f.name}: example differs from default"
    for name, section in example.items():
        known = {f.name for f in dataclasses.fields(getattr(cfg, name))}
        assert set(section) <= known, f"[{name}] documents unknown {set(section) - known}"


def test_set_value_prefers_live_line_and_header_comments(tmp_path):
    """Regressions from review: a commented duplicate was replaced instead of
    the live key, and a header with a trailing comment wasn't recognized."""
    from voiceio.config import set_value
    p = tmp_path / "config.toml"
    p.write_text("[output]  # mine\n# streaming_freeze_secs = 15\nstreaming_freeze_secs = 10\n")
    set_value("output", "streaming_freeze_secs", 8.0, path=p)
    assert tomllib.loads(p.read_text()) == {"output": {"streaming_freeze_secs": 8.0}}
    assert p.read_text().count("[output]") == 1


def test_set_value_never_writes_unparseable_toml(tmp_path):
    import pytest as _pytest
    from voiceio.config import set_value
    p = tmp_path / "config.toml"
    p.write_text("[model]\nname = \"small\"\n")
    before = p.read_text()
    with _pytest.raises(tomllib.TOMLDecodeError):
        set_value("model", "bad key", "x", path=p)
    assert p.read_text() == before


def test_remove_legacy_state_deletes_pre_1_0_mining_files(tmp_path, monkeypatch):
    from voiceio import config
    files = (tmp_path / "corrections_audit.jsonl", tmp_path / "autocorrect_state.json")
    for f in files:
        f.write_text("{}")
    monkeypatch.setattr(config, "LEGACY_STATE_FILES", files + (tmp_path / "absent.json",))
    config.remove_legacy_state()
    assert not any(f.exists() for f in files)


def test_autocorrect_section_migrates_to_cloud_and_model(tmp_path):
    from voiceio import config
    path = tmp_path / "config.toml"
    path.write_text('[autocorrect]\napi_key = "k"\nmodel = "old/m"\nprotect_languages = ["es"]\n'
                    '[cloud]\nmodel = "new/m"\n')
    cfg = config.load(path)
    assert (cfg.cloud.api_key, cfg.cloud.model) == ("k", "new/m")   # [cloud] wins
    assert cfg.model.protect_languages == ["es"]
