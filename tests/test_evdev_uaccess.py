"""udev `uaccess` rule for evdev hotkeys (replaces the input-group re-login)."""
from __future__ import annotations

import tomllib
from pathlib import Path

from voiceio.hotkeys import evdev as evdev_mod

ROOT = Path(__file__).resolve().parent.parent


def test_rule_file_content():
    rule = evdev_mod.uaccess_rule()
    assert rule.strip() == (
        'ACTION=="add|change", KERNEL=="event*", '
        'ENV{ID_INPUT_KEY}=="1", TAG+="uaccess"'
    )
    assert rule.endswith("\n")


def test_rule_sorts_before_seat_late():
    # logind turns uaccess tags into ACLs in 73-seat-late.rules.
    assert evdev_mod.UACCESS_RULE_NAME < "73-seat-late.rules"
    assert evdev_mod.UACCESS_RULE_PATH == f"/etc/udev/rules.d/{evdev_mod.UACCESS_RULE_NAME}"


def test_rule_ships_as_package_data():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    globs = pyproject["tool"]["setuptools"]["package-data"]["voiceio.data"]
    assert any(Path(evdev_mod.UACCESS_RULE_NAME).match(g) for g in globs)
    assert (ROOT / "voiceio" / "data" / "__init__.py").exists()


def test_install_cmd_is_single_sudo_script():
    cmd = evdev_mod.uaccess_install_cmd()
    assert cmd[:2] == ["sh", "-c"] and len(cmd) == 3
    script = cmd[2]
    assert f"sudo tee {evdev_mod.UACCESS_RULE_PATH}" in script
    assert "sudo udevadm control --reload-rules" in script
    assert "sudo udevadm trigger" in script
    assert "uaccess" in script
    assert "usermod" not in script


def _deny_all_inputs(monkeypatch, tmp_path):
    (tmp_path / "event0").write_text("")
    real_glob = Path.glob

    def fake_glob(self, pattern):
        if str(self) == "/dev/input":
            return [tmp_path / "event0"]
        return real_glob(self, pattern)

    def deny(*a, **k):
        raise PermissionError

    monkeypatch.setattr(Path, "glob", fake_glob)
    monkeypatch.setattr("builtins.open", deny)


def test_probe_offers_uaccess_rule(monkeypatch, tmp_path):
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: False)
    _deny_all_inputs(monkeypatch, tmp_path)
    result = evdev_mod.EvdevHotkey().probe()
    assert not result.ok
    assert result.fix_cmd == evdev_mod.uaccess_install_cmd()
    assert "usermod -aG input" in result.fix_hint  # documented fallback


def test_probe_on_nixos_gives_config_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: True)
    _deny_all_inputs(monkeypatch, tmp_path)
    result = evdev_mod.EvdevHotkey().probe()
    assert not result.ok
    assert not result.fix_cmd
    assert "configuration.nix" in result.fix_hint
    assert "uaccess" in result.fix_hint


def test_access_hint_prefers_rule(monkeypatch):
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: False)
    hint = evdev_mod.access_hint()
    assert hint.index("doctor --fix") < hint.index("usermod")
    monkeypatch.setattr("voiceio.platform.is_nixos", lambda: True)
    assert "configuration.nix" in evdev_mod.access_hint()
