"""Tests for setup: config merge, preflight, and non-interactive setup."""
from __future__ import annotations

from unittest.mock import patch

import os
import sys

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
    ("emerge", "sudo emerge --ask sys-devel/gcc media-libs/portaudio app-i18n/ibus dev-python/pygobject"),
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


def test_gentoo_hints_use_atoms():
    with patch("voiceio.platform._detect_pkg_manager", return_value="emerge"):
        assert plat.pkg_install("gir1.2-ibus-1.0", "python3-gi") == \
            "sudo emerge --ask app-i18n/ibus dev-python/pygobject"
        assert plat.pkg_install("xdotool", "xclip") == "sudo emerge --ask x11-misc/xdotool x11-misc/xclip"
        assert plat.pkg_install("wtype") == "sudo emerge --ask gui-apps/wtype"
        assert plat.pkg_install("ydotool") == "sudo emerge --ask ydotool"  # emerge resolves it


def test_gentoo_is_detected_by_emerge(monkeypatch):
    plat._detect_pkg_manager.cache_clear()
    monkeypatch.setattr(plat, "is_nixos", lambda: False)
    monkeypatch.setattr(plat.shutil, "which", lambda b: "/usr/bin/emerge" if b == "emerge" else None)
    try:
        assert plat._detect_pkg_manager() == "emerge"
    finally:
        plat._detect_pkg_manager.cache_clear()


def test_missing_sounddevice_is_not_missing_portaudio(monkeypatch):
    """`uvx python-voiceio` (no [desktop]) has no sounddevice: that is the
    extra, not the system library, when the library is installed."""
    import builtins
    real_import = builtins.__import__

    def no_sounddevice(name, *a, **k):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_sounddevice)
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)

    monkeypatch.setattr(plat, "_portaudio_lib_present", lambda: True)
    assert plat._probe_portaudio()
    problem = plat.audio_problem()
    assert "sounddevice is not installed" in problem and "[desktop]" in problem
    assert "PortAudio" not in problem
    monkeypatch.setattr(plat, "_portaudio_lib_present", lambda: False)
    assert not plat._probe_portaudio()  # both missing: still list the system dep


def test_portaudio_library_missing_is_a_system_dep(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_lib(name, *a, **k):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_lib)
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(plat, "_detect_pkg_manager", lambda: "emerge")
    assert not plat._probe_portaudio()
    assert plat.audio_problem() == ("PortAudio library not found — install: "
                                    "sudo emerge --ask media-libs/portaudio")


def test_desktop_hint_under_uvx_names_the_extra(monkeypatch, tmp_path):
    cache = tmp_path / "uv"
    (cache / "archive-v0" / "h").mkdir(parents=True)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    monkeypatch.setattr(sys, "prefix", str(cache / "archive-v0" / "h"))
    hint = plat.desktop_dep_hint("sounddevice")
    assert "uvx 'python-voiceio[desktop]'" in hint and "uv tool install" in hint


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


def test_setup_offers_nightly_learning_once(monkeypatch):
    from voiceio import service
    from voiceio.setup import steps
    installed = []
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "learn_timer_enabled", lambda: bool(installed))
    monkeypatch.setattr(service, "install_learn_timer", lambda: installed.append(1) or True)
    asked = []
    monkeypatch.setattr(steps, "_ask_yn", lambda q, default=True: asked.append(q) or True)
    state = steps.SetupState(checks={"is_linux": True})
    steps._offer_nightly_learning(state)
    steps._offer_nightly_learning(state)      # already on: not asked again
    assert installed == [1] and len(asked) == 1


# ── root commands with no terminal (agent, launcher, GUI) ────────────

def _tools(monkeypatch, *names):
    """Only these privilege tools are on PATH; not running as root."""
    monkeypatch.setattr(plat, "_is_root", lambda: False)
    monkeypatch.setattr(plat.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name in names else None)


def _no_terminal(monkeypatch, *, sudo_cached=False, pkexec=True, display=True,
                 sudo=True, doas=False, doas_cached=False):
    monkeypatch.setattr(plat, "_has_tty", lambda: False)
    monkeypatch.setattr(plat, "_sudo_without_password", lambda: sudo_cached)
    monkeypatch.setattr(plat, "_doas_without_password", lambda: doas_cached)
    _tools(monkeypatch, *[n for n, on in (("sudo", sudo), ("doas", doas),
                                          ("pkexec", pkexec)) if on])
    monkeypatch.delenv("DISPLAY", raising=False)
    if display:
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    else:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def test_privileged_argv_unchanged_with_a_terminal(monkeypatch):
    monkeypatch.setattr(plat, "_has_tty", lambda: True)
    _tools(monkeypatch, "sudo", "doas", "pkexec")
    argv = ["sudo", "apt", "install", "ibus"]
    assert plat.privileged_argv(argv) == argv


def test_privileged_argv_leaves_unprivileged_commands_alone(monkeypatch):
    _no_terminal(monkeypatch, pkexec=False)
    assert plat.privileged_argv(["brew", "install", "x"]) == ["brew", "install", "x"]


def test_privileged_argv_uses_pkexec_without_a_terminal(monkeypatch):
    _no_terminal(monkeypatch)
    assert plat.privileged_argv(["sudo", "apt", "install", "ibus", "gcc"]) == [
        "pkexec", "apt", "install", "-y", "ibus", "gcc"]
    assert plat.privileged_argv(["sudo", "pacman", "-S", "ibus"]) == [
        "pkexec", "pacman", "-S", "--noconfirm", "ibus"]


def test_privileged_argv_runs_shell_scripts_with_inner_sudo_under_pkexec(monkeypatch):
    _no_terminal(monkeypatch)
    script = ["sh", "-c", "printf x | sudo tee /etc/udev/rules.d/r > /dev/null"]
    assert plat.privileged_argv(script) == ["pkexec", *script]


def test_privileged_argv_drops_emerge_ask_without_a_terminal(monkeypatch):
    _no_terminal(monkeypatch)
    assert plat.privileged_argv(["sudo", "emerge", "--ask", "media-libs/portaudio"]) == [
        "pkexec", "emerge", "media-libs/portaudio"]
    monkeypatch.setattr(plat, "_has_tty", lambda: True)
    assert plat.privileged_argv(["sudo", "emerge", "--ask", "x"]) == ["sudo", "emerge", "--ask", "x"]
    _tools(monkeypatch, "doas")
    assert plat.privileged_argv(["sudo", "emerge", "--ask", "x"]) == ["doas", "emerge", "--ask", "x"]


def test_privileged_argv_prefers_cached_sudo(monkeypatch):
    _no_terminal(monkeypatch, sudo_cached=True)
    assert plat.privileged_argv(["sudo", "apt", "install", "x"]) == [
        "sudo", "-n", "apt", "install", "-y", "x"]


def test_privileged_argv_uses_doas_with_a_terminal_when_no_sudo(monkeypatch):
    monkeypatch.setattr(plat, "_has_tty", lambda: True)
    _tools(monkeypatch, "doas", "pkexec")
    assert plat.privileged_argv(["sudo", "apt", "install", "x"]) == ["doas", "apt", "install", "x"]
    _tools(monkeypatch, "sudo", "doas")
    assert plat.privileged_argv(["sudo", "apt", "install", "x"]) == ["sudo", "apt", "install", "x"]


def test_privileged_argv_runs_the_udev_fix_under_doas(monkeypatch):
    from voiceio.hotkeys.evdev import uaccess_install_cmd
    monkeypatch.setattr(plat, "_has_tty", lambda: True)
    _tools(monkeypatch, "doas")
    cmd = uaccess_install_cmd()
    argv = plat.privileged_argv(cmd)
    assert argv == ["doas", *cmd[1:]]
    assert "sudo" not in " ".join(argv)   # the script itself never calls sudo


def test_privileged_argv_without_terminal_order_is_sudo_doas_pkexec(monkeypatch):
    argv = ["sudo", "apt", "install", "x"]
    _no_terminal(monkeypatch, sudo_cached=True, doas=True, doas_cached=True)
    assert plat.privileged_argv(argv) == ["sudo", "-n", "apt", "install", "-y", "x"]
    _no_terminal(monkeypatch, sudo_cached=False, doas=True, doas_cached=True)
    assert plat.privileged_argv(argv) == ["doas", "-n", "apt", "install", "-y", "x"]
    _no_terminal(monkeypatch, sudo=False, doas=True, doas_cached=True)
    assert plat.privileged_argv(argv) == ["doas", "-n", "apt", "install", "-y", "x"]
    _no_terminal(monkeypatch, doas=True, doas_cached=False)
    assert plat.privileged_argv(argv) == ["pkexec", "apt", "install", "-y", "x"]


def test_privileged_argv_as_root_drops_sudo(monkeypatch):
    _tools(monkeypatch)
    monkeypatch.setattr(plat, "_is_root", lambda: True)
    assert plat.privileged_argv(["sudo", "udevadm", "trigger"]) == ["udevadm", "trigger"]


def test_privileged_argv_none_without_any_tool(monkeypatch, capsys):
    from voiceio.cli import doctor
    from voiceio.hotkeys.evdev import uaccess_install_cmd
    monkeypatch.setattr(plat, "_has_tty", lambda: True)
    _tools(monkeypatch)
    cmd = uaccess_install_cmd()
    assert plat.privileged_argv(cmd) is None
    msg = plat.privilege_unavailable_message(cmd)
    assert "as root" in msg and "su" in msg
    assert "sudo" not in msg.split("\n", 1)[1]   # the command to paste in a root shell
    assert "mkdir -p /etc/udev/rules.d" in msg
    # doctor --fix prints that instead of running anything
    ran = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(doctor, "_confirm_privileged", lambda d, c: True)
    assert doctor._run_privileged("Fix evdev", cmd, 5) is False
    assert ran == [] and "as root" in capsys.readouterr().out


def test_as_root_command_strips_sudo():
    assert plat.as_root_command(["sudo", "emerge", "--ask", "x"]) == "emerge --ask x"
    assert plat.as_root_command(["sh", "-c", "a | sudo tee f && sudo -n b"]) == "sh -c 'a | tee f && b'"


@pytest.mark.parametrize("pkexec,display", [(False, True), (True, False)])
def test_privileged_argv_gives_up_without_pkexec_or_desktop(monkeypatch, pkexec, display):
    _no_terminal(monkeypatch, pkexec=pkexec, display=display)
    assert plat.privileged_argv(["sudo", "apt", "install", "x"]) is None


def test_noninteractive_install_deps_without_terminal_prints_command(cfg_paths, monkeypatch, capsys):
    _patch_noninteractive(monkeypatch)
    monkeypatch.setattr("voiceio.platform.check_system_deps", lambda: ["portaudio"])
    monkeypatch.setattr(plat, "system_deps_install_cmd", lambda keys: "sudo apt install libportaudio2")
    monkeypatch.setattr(plat, "privileged_argv", lambda argv: None)
    ran = []
    monkeypatch.setattr(noninteractive.subprocess, "run", lambda *a, **k: ran.append(a))
    code = noninteractive.run_setup_noninteractive({}, install_deps=True)
    assert code == 4
    assert ran == []
    assert "sudo apt install libportaudio2" in capsys.readouterr().out


def test_noninteractive_install_deps_runs_pkexec_argv(cfg_paths, monkeypatch):
    _patch_noninteractive(monkeypatch)
    state = {"missing": ["portaudio"]}
    monkeypatch.setattr("voiceio.platform.check_system_deps", lambda: state["missing"])
    monkeypatch.setattr(plat, "system_deps_install_cmd", lambda keys: "sudo apt install libportaudio2")
    _no_terminal(monkeypatch)
    ran = []

    class _Done:
        returncode = 0

    def fake_run(argv, **kw):
        ran.append(argv)
        state["missing"] = []
        return _Done()

    monkeypatch.setattr(noninteractive.subprocess, "run", fake_run)
    assert noninteractive.run_setup_noninteractive({}, install_deps=True) == 0
    assert ran == [["pkexec", "apt", "install", "-y", "libportaudio2"]]


def test_doctor_privileged_fix_asks_first_and_never_runs_when_declined(monkeypatch):
    from voiceio.cli import doctor
    ran = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(doctor, "_confirm_privileged", lambda d, c: False)
    assert doctor._run_privileged("Fix x", ["sudo", "true"], 5) is None
    assert ran == []


def test_doctor_privileged_fix_without_terminal_prints_command(monkeypatch, capsys):
    from voiceio.cli import doctor
    ran = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(doctor, "_confirm_privileged", lambda d, c: True)
    monkeypatch.setattr(plat, "privileged_argv", lambda argv: None)
    assert doctor._run_privileged("Fix x", ["sudo", "udevadm", "trigger"], 5) is False
    assert ran == []
    assert "sudo udevadm trigger" in capsys.readouterr().out


def test_wizard_names_the_missing_extra_not_a_missing_mic(capsys):
    from voiceio.setup import steps
    state = steps.SetupState()
    state.checks = {"audio": False, "audio_devices": [], "is_windows": False, "is_mac": False,
                    "audio_reason": plat.desktop_dep_hint("sounddevice")}
    with pytest.raises(SystemExit):
        steps._require_mic_and_typer(state)
    out = capsys.readouterr().out
    assert "sounddevice is not installed" in out and "No microphone" not in out
