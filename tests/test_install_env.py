"""uvx runs voiceio from uv's cache: setup must not bake those paths in, doctor
must flag them."""
import sys
from pathlib import Path

import pytest

from voiceio import install_env, service

TAG = "Signature: 8a477f597d28d172789f06886806bc55\n# This file is a cache directory tag.\n"


@pytest.fixture
def uv_cache(monkeypatch, tmp_path):
    """An uvx environment with a voiceio script, in a sandboxed uv cache."""
    cache = tmp_path / "uv-cache"
    env = cache / "archive-v0" / "AbCd123"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "voiceio").write_text("#!/bin/sh\n")
    (cache / "CACHEDIR.TAG").write_text(TAG)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))
    return env


def test_cache_dir_follows_uv(monkeypatch, tmp_path):
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert install_env.uv_cache_dir() == tmp_path / "xdg" / "uv"
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/ignored")
    assert install_env.uv_cache_dir() == Path.home() / ".cache" / "uv"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert install_env.uv_cache_dir() == Path.home() / ".cache" / "uv"
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "custom"))
    assert install_env.uv_cache_dir() == tmp_path / "custom"


def test_paths_under_the_cache_dir_are_ephemeral_even_without_a_tag(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "c"))
    assert install_env.in_uv_cache(tmp_path / "c" / "archive-v0" / "x" / "bin" / "voiceio")
    assert not install_env.in_uv_cache(tmp_path / "elsewhere" / "bin" / "voiceio")


def test_a_tagged_cache_elsewhere_is_ephemeral(monkeypatch, tmp_path):
    # `uvx --cache-dir DIR` or `--no-cache` (a temp dir): no env var says so,
    # but uv tags every cache root.
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "default"))
    other = tmp_path / "tmpcache"
    (other / "archive-v0" / "x").mkdir(parents=True)
    (other / "CACHEDIR.TAG").write_text(TAG)
    assert install_env.in_uv_cache(other / "archive-v0" / "x")
    (other / "CACHEDIR.TAG").write_text("not a cache tag\n")
    assert not install_env.in_uv_cache(other / "archive-v0" / "x")


def test_persistent_installs_are_not_ephemeral(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    for prefix in (tmp_path / ".local/share/uv/tools/python-voiceio",
                   tmp_path / ".local/share/pipx/venvs/python-voiceio"):
        prefix.mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(prefix))
        assert not install_env.running_ephemeral()


def test_service_install_refuses_a_uv_cache_binary(monkeypatch, tmp_path, uv_cache, capsys):
    monkeypatch.setattr(sys, "prefix", str(uv_cache))
    monkeypatch.setattr(service, "has_systemd", lambda: True)
    monkeypatch.setattr(service, "_IS_WINDOWS", False)
    monkeypatch.setattr(service, "SERVICE_DIR", tmp_path / "units")
    monkeypatch.setattr(service, "SERVICE_PATH", tmp_path / "units" / service.SERVICE_NAME)
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", tmp_path / "units" / "l.service")
    monkeypatch.setattr(service, "LEARN_TIMER_PATH", tmp_path / "units" / "l.timer")
    monkeypatch.setattr(service, "LOCAL_BIN", tmp_path / "local-bin")
    monkeypatch.setattr(service, "_systemctl", lambda *a: pytest.fail("must not run"))

    assert not service.install_service()
    assert not service.install_learn_timer()
    assert service.install_symlinks() == []
    assert not (tmp_path / "units").exists() and not (tmp_path / "local-bin").exists()
    err = capsys.readouterr().err
    assert "uv tool install 'python-voiceio[desktop]'" in err and "voiceio setup" in err


def test_noninteractive_setup_refuses_from_uvx(monkeypatch, uv_cache, capsys):
    from voiceio.setup.noninteractive import run_setup_noninteractive
    monkeypatch.setattr(sys, "prefix", str(uv_cache))
    assert run_setup_noninteractive({}) == 3
    out = capsys.readouterr().out
    assert "step=install status=error" in out and "uv tool install" in out
    from voiceio.config import CONFIG_PATH
    assert not CONFIG_PATH.exists()


def test_wizard_refuses_from_uvx(monkeypatch, uv_cache, capsys):
    from voiceio.setup import steps
    monkeypatch.setattr(sys, "prefix", str(uv_cache))
    monkeypatch.setattr(steps, "_welcome", lambda: pytest.fail("must stop first"))
    with pytest.raises(SystemExit) as e:
        steps.run_wizard()
    assert e.value.code == 3
    assert "uv tool install 'python-voiceio[desktop]'" in capsys.readouterr().out


@pytest.fixture
def written(monkeypatch, tmp_path):
    """Point the service unit, learn unit, IBus launcher and links at tmp."""
    from voiceio.ibus import install as ibus_install
    paths = {
        "unit": tmp_path / "voiceio.service",
        "learn": tmp_path / "voiceio-learn.service",
        "launcher": tmp_path / "voiceio-ibus-engine",
        "bin": tmp_path / "local-bin",
    }
    paths["bin"].mkdir()
    monkeypatch.setattr(service, "SERVICE_PATH", paths["unit"])
    monkeypatch.setattr(service, "LEARN_SERVICE_PATH", paths["learn"])
    monkeypatch.setattr(service, "LOCAL_BIN", paths["bin"])
    monkeypatch.setattr(ibus_install, "LAUNCHER_PATH", paths["launcher"])
    return paths


def test_doctor_flags_units_and_launcher_in_uv_cache(monkeypatch, tmp_path, uv_cache, written):
    good = tmp_path / "tools" / "bin" / "voiceio"
    good.parent.mkdir(parents=True)
    good.write_text("#!/bin/sh\n")
    cached = uv_cache / "bin" / "voiceio"
    written["unit"].write_text(service._service_unit(str(cached)))
    written["learn"].write_text(f"[Service]\nExecStart={good} learn maintain\n")
    written["launcher"].write_text(
        f'VOICEIO_PKG_ROOT="{uv_cache}/lib/python3.14/site-packages"\n'
        'SYSTEM_PY="/usr/bin/python3"\n'
        f'INSTALL_PY="{uv_cache}/bin/python"\n')
    (written["bin"] / "voiceio").symlink_to(cached)

    stale = install_env.stale_install_paths()
    flagged = {path for _, path, _ in stale}
    assert str(cached) in flagged
    assert f"{uv_cache}/lib/python3.14/site-packages" in flagged
    assert f"{uv_cache}/bin/python" in flagged
    assert str(good) not in flagged and "/usr/bin/python3" not in flagged
    assert any(w.startswith("command link") for w, _, _ in stale)
    assert {p for _, _, p in stale} <= {"in uv's cache", "in uv's cache, already deleted"}

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "tools"))
    report = install_env.stale_paths_report(stale)
    assert "uv tool install 'python-voiceio[desktop]'" in report and "voiceio setup" in report


def test_doctor_flags_a_unit_whose_binary_is_gone(tmp_path, written, monkeypatch):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    gone = tmp_path / "old-venv" / "bin" / "voiceio"
    written["unit"].write_text(service._service_unit(str(gone)))
    assert install_env.stale_install_paths() == [
        (f"autostart service {written['unit']}", str(gone), "no longer exists")]


def test_doctor_is_quiet_for_a_durable_install(tmp_path, written, monkeypatch):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    good = tmp_path / "venv" / "bin" / "voiceio"
    good.parent.mkdir(parents=True)
    good.write_text("#!/bin/sh\n")
    written["unit"].write_text(service._service_unit(str(good)))
    written["learn"].write_text("[Service]\nExecStart=voiceio learn maintain\n")  # PATH lookup
    (written["bin"] / "voiceio").symlink_to(good)
    assert install_env.stale_install_paths() == []


def test_doctor_cli_exits_nonzero_on_stale_paths(monkeypatch, capsys):
    import argparse

    from voiceio.cli import doctor
    from voiceio import health, platform as plat, config
    monkeypatch.setattr(install_env, "stale_install_paths",
                        lambda: [("autostart service x", "/gone/voiceio", "no longer exists")])
    monkeypatch.setattr(install_env, "running_ephemeral", lambda: False)
    rep = health.HealthReport(platform=plat.detect())
    monkeypatch.setattr(health, "check_health", lambda: rep)
    monkeypatch.setattr(health, "format_report", lambda r: "")
    monkeypatch.setattr(health, "usage_findings", lambda: [])
    monkeypatch.setattr(plat, "check_system_deps", lambda: [])
    monkeypatch.setattr(config, "check_permissions", lambda: [])
    monkeypatch.setattr(doctor, "_offer_report", lambda *a: False)
    monkeypatch.setattr(type(rep), "all_ok", property(lambda self: True))
    args = argparse.Namespace(live=False, fix=False, report=False)
    with pytest.raises(SystemExit) as e:
        doctor._cmd_doctor(args)
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "/gone/voiceio (no longer exists)" in out and "uv tool install" in out
