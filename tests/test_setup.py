"""Tests for setup: config merge, preflight, and non-interactive setup."""
from __future__ import annotations

from unittest.mock import patch

import os

import pytest

import voiceio.config as config
from voiceio.setup import configfile, noninteractive
from voiceio.setup.hotkey import _wm_bind_lines
from voiceio import platform as plat


@pytest.fixture
def cfg_paths(tmp_path, monkeypatch):
    """Point config + setup at a throwaway config.toml."""
    cfg_dir = tmp_path / "config"
    cfg_path = cfg_dir / "config.toml"
    for mod, name, val in [
        (config, "CONFIG_DIR", cfg_dir), (config, "CONFIG_PATH", cfg_path),
        (configfile, "CONFIG_DIR", cfg_dir), (configfile, "CONFIG_PATH", cfg_path),
        (noninteractive, "CONFIG_PATH", cfg_path),
    ]:
        monkeypatch.setattr(mod, name, val)
    return cfg_path


# ── Config merge preserves keys / secrets ────────────────────────────────

def test_write_config_preserves_secret_when_no_new_key(cfg_paths):
    cfg_paths.parent.mkdir(parents=True)
    cfg_paths.write_text(
        '[autocorrect]\napi_key = "SECRET"\nmodel = "custom/model"\n'
        '[audio]\ndevice = "hw:1"\n'
        '[unknown]\nkept = 7\n'
    )
    configfile._write_config(
        model="base", language="es", hotkey="alt+v", method="auto",
        streaming=True, backend="evdev", quiet=True,
    )
    raw = configfile._load_raw_config()
    # Wizard-owned keys updated
    assert raw["model"]["name"] == "base"
    assert raw["model"]["language"] == "es"
    assert raw["hotkey"]["key"] == "alt+v"
    # Secret + hand-edited + unknown keys preserved verbatim
    assert raw["cloud"]["api_key"] == "SECRET"
    assert raw["cloud"]["model"] == "custom/model"
    assert raw["audio"]["device"] == "hw:1"
    assert raw["unknown"]["kept"] == 7


def test_write_config_sets_api_key_when_provided(cfg_paths):
    configfile._write_config(
        model="small", language="en", hotkey="ctrl+alt+v", method="auto",
        streaming=True, backend="evdev",
        cloud_api_key="NEWKEY", cloud_base_url="https://x/api",
        cloud_model="m/n", quiet=True,
    )
    raw = configfile._load_raw_config()
    assert raw["cloud"]["api_key"] == "NEWKEY"
    assert raw["cloud"]["base_url"] == "https://x/api"


def test_write_config_roundtrips_through_loader(cfg_paths):
    configfile._write_config(
        model="tiny", language="fr", hotkey="ctrl+shift+v", method="auto",
        streaming=False, backend="socket", voice_input_prefix="[voice]",
        quiet=True,
    )
    loaded = config.load(cfg_paths)
    assert loaded.model.name == "tiny"
    assert loaded.output.streaming is False
    assert loaded.output.voice_input_prefix == "[voice]"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_write_config_restricts_permissions(cfg_paths):
    configfile._write_config(
        model="small", language="en", hotkey="ctrl+alt+v", method="auto",
        streaming=True, backend="evdev", quiet=True,
    )
    assert oct(os.stat(cfg_paths).st_mode)[-3:] == "600"


# ── TOML serializer round-trips ──────────────────────────────────────────

def test_dump_toml_types():
    out = configfile._dump_toml({
        "s": {"str": "hi", "b": True, "i": 3, "f": 1.5, "lst": ["a", "b"]},
    })
    import tomllib
    parsed = tomllib.loads(out)
    assert parsed["s"] == {"str": "hi", "b": True, "i": 3, "f": 1.5, "lst": ["a", "b"]}


def test_dump_toml_escapes_quotes():
    out = configfile._dump_toml({"s": {"k": 'a"b\\c'}})
    import tomllib
    assert tomllib.loads(out)["s"]["k"] == 'a"b\\c'


# ── Preflight: package lists per distro + missing detection ──────────────

@pytest.mark.parametrize("mgr,expected", [
    ("apt", "sudo apt install build-essential python3-dev portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi"),
    ("dnf", "sudo dnf install gcc gcc-c++ make python3-devel portaudio-devel ibus ibus-libs python3-gobject"),
    ("pacman", "sudo pacman -S base-devel portaudio ibus python-gobject"),
])
def test_system_deps_install_cmd_per_distro(mgr, expected):
    with patch("voiceio.platform._detect_pkg_manager", return_value=mgr):
        assert plat.system_deps_install_cmd(["compiler", "portaudio", "ibus"]) == expected


def test_system_deps_install_cmd_subset():
    with patch("voiceio.platform._detect_pkg_manager", return_value="apt"):
        assert plat.system_deps_install_cmd(["portaudio"]) == "sudo apt install portaudio19-dev"


def test_check_system_deps_reports_missing():
    with patch("voiceio.platform._detect_os", return_value="linux"), \
         patch.dict(plat._DEP_PROBES, {
             "compiler": lambda: True,
             "portaudio": lambda: False,
             "ibus": lambda: False,
         }):
        assert plat.check_system_deps() == ["portaudio", "ibus"]


def test_check_system_deps_empty_on_non_linux():
    with patch("voiceio.platform._detect_os", return_value="darwin"):
        assert plat.check_system_deps() == []


# ── Non-interactive setup ────────────────────────────────────────────────

_LINUX_CHECKS = {
    "is_windows": False, "is_mac": False, "is_linux": True,
    "display": "wayland", "audio": True, "audio_devices": [object()],
    "pynput": False, "xdotool": True, "xclip": False, "ydotool": False,
    "wtype": False, "ibus": False, "ibus_gi": False, "cuda": False,
    "evdev_access": True,
}


def _patch_noninteractive(monkeypatch):
    """Mock the heavy/side-effecting steps of non-interactive setup."""
    monkeypatch.setattr(noninteractive, "_check_system", lambda: dict(_LINUX_CHECKS))
    monkeypatch.setattr(noninteractive, "_download_model", lambda name: True)
    monkeypatch.setattr("voiceio.tray.probe_availability", lambda: (False, "", ""))
    monkeypatch.setattr("voiceio.platform.check_system_deps", lambda: [])
    monkeypatch.setattr("voiceio.service.has_systemd", lambda: False)
    monkeypatch.setattr("voiceio.service.install_service", lambda: False)


def test_noninteractive_defaults_no_tty(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    code = noninteractive.run_setup_noninteractive({})
    assert code == 0
    out = capsys.readouterr().out
    assert "[voiceio-setup] step=done status=ok" in out
    loaded = config.load(cfg_paths)
    assert loaded.model.name == "small"       # default
    assert loaded.hotkey.key == "ctrl+alt+v"  # default
    assert loaded.hotkey.backend == "evdev"   # from evdev_access


def test_noninteractive_answers_applied(cfg_paths, monkeypatch):
    _patch_noninteractive(monkeypatch)
    code = noninteractive.run_setup_noninteractive({
        "model": "base", "language": "de", "hotkey": "alt+v",
        "tts_enabled": False, "voice_input_prefix": "[v]",
    })
    assert code == 0
    loaded = config.load(cfg_paths)
    assert loaded.model.name == "base"
    assert loaded.model.language == "de"
    assert loaded.hotkey.key == "alt+v"
    assert loaded.tts.enabled is False
    assert loaded.output.voice_input_prefix == "[v]"


def test_noninteractive_rejects_unknown_key(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    code = noninteractive.run_setup_noninteractive({"bogus": 1})
    assert code == 2
    assert "unknown answer keys" in capsys.readouterr().out


def test_noninteractive_rejects_invalid_model(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    code = noninteractive.run_setup_noninteractive({"model": "gigantic"})
    assert code == 2
    assert "invalid model" in capsys.readouterr().out


def test_noninteractive_fails_without_microphone(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    no_audio = dict(_LINUX_CHECKS, audio=False, audio_devices=[])
    monkeypatch.setattr(noninteractive, "_check_system", lambda: no_audio)
    code = noninteractive.run_setup_noninteractive({})
    assert code == 5
    assert "microphone" in capsys.readouterr().out


def test_noninteractive_headless_writes_config_and_skips_heavy_steps(
        cfg_paths, monkeypatch, capsys):
    """Unusable system (no mic, no typer) still gets a valid config, the
    documented exit code, and no model download / service install."""
    _patch_noninteractive(monkeypatch)
    headless = dict(_LINUX_CHECKS, audio=False, audio_devices=[], xdotool=False)
    monkeypatch.setattr(noninteractive, "_check_system", lambda: headless)
    monkeypatch.setattr(noninteractive, "_download_model",
                        lambda name: pytest.fail("model download must be skipped"))
    code = noninteractive.run_setup_noninteractive({})
    assert code == 5  # first failure (audio) wins
    out = capsys.readouterr().out
    assert "step=config status=written" in out
    assert "step=model" in out and "status=skipped" in out
    loaded = config.load(cfg_paths)
    assert loaded.model.name == "small"
    assert loaded.hotkey.key == "ctrl+alt+v"


def test_noninteractive_fails_on_model_download(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    monkeypatch.setattr(noninteractive, "_download_model", lambda name: False)
    code = noninteractive.run_setup_noninteractive({})
    assert code == 7
    assert "download" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("hotkey,hypr,sway", [
    ("ctrl+alt+v", "bind = CTRL ALT, V, exec, voiceio toggle",
     "bindsym Ctrl+Alt+v exec voiceio toggle"),
    ("super+shift+d", "bind = SUPER SHIFT, D, exec, voiceio toggle",
     "bindsym Mod4+Shift+d exec voiceio toggle"),
])
def test_wm_bind_lines_use_chosen_hotkey(hotkey, hypr, sway):
    assert dict(_wm_bind_lines(hotkey)) == {"hyprland": hypr, "i3/sway": sway}


def test_write_config_with_api_key_records_consent_and_is_private(tmp_path, monkeypatch):
    """Regression: wizard called record_consent() positionally → TypeError."""
    import stat
    import voiceio.consent as consent
    monkeypatch.setattr(configfile, "CONFIG_PATH", config.CONFIG_PATH)
    configfile._write_config(
        model="small", language="en", hotkey="ctrl+alt+v", method="auto",
        streaming=True, backend="auto", cloud_api_key="sk-test", quiet=True,
    )
    configfile._record_cloud_consent()
    assert consent.has_cloud_consent()
    assert stat.S_IMODE(config.CONFIG_PATH.stat().st_mode) == 0o600


def test_retention_is_written_only_when_asked(cfg_paths):
    from voiceio import config
    kw = dict(model="small", language="en", hotkey="ctrl+alt+v", method="auto",
              streaming=True, backend="auto", quiet=True)
    configfile._write_config(**kw)
    assert "retain_audio" not in configfile._load_raw_config().get("data", {})
    assert config.load(cfg_paths).data.retain_audio is False      # off unless chosen
    configfile._write_config(**kw, retain_audio=True)
    assert config.load(cfg_paths).data.retain_audio is True
