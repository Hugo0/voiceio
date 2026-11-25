"""The setup wizard offering Parakeet to English users without sherpa-onnx."""
from __future__ import annotations

import sys

import pytest

from voiceio.core.decoders import catalog, fetch, sherpa
from voiceio.core.decoders.sherpa import SherpaFeatures
from voiceio.setup import noninteractive, speech, steps
from tests.test_setup import _patch_noninteractive, cfg_paths  # noqa: F401 — fixture

# conftest replaces `sherpa.features` in every test; keep the real one.
REAL_FEATURES = sherpa.features
FIXED = SherpaFeatures(installed=True, length_reward=True, pretokenized_hotwords=True)


@pytest.fixture
def installable(monkeypatch):
    """English user, no sherpa-onnx, an environment the installer can write to."""
    monkeypatch.setattr(fetch, "can_install_sherpa", lambda: True)


@pytest.fixture
def wizard(monkeypatch):
    """Run `_step_model` with the menu answer given; record downloads."""
    downloads: list[str] = []
    monkeypatch.setattr(steps, "_download_model", lambda name: downloads.append(name) or True)
    monkeypatch.setattr(steps, "_ask_yn", lambda *a, **k: False)
    monkeypatch.setattr(steps, "_print_step", lambda *a: None)

    def run(answer: int = 0, language: str = "en") -> tuple[steps.SetupState, list, list]:
        shown: list = []

        def ask(options, default=0):
            shown.append((list(options), default))
            return answer
        monkeypatch.setattr(steps, "_ask_choice", ask)
        state = steps.SetupState(language=language)
        steps._step_model(state)
        return state, shown, downloads
    return run


def test_english_without_sherpa_offers_parakeet_first_with_install_label(installable):
    choices = speech.model_choices("en", offer_install=True)
    assert choices[0] == speech.PARAKEET_INSTALL_CHOICE
    assert choices[0][0] == catalog.PARAKEET
    assert "installs sherpa-onnx" in choices[0][2]
    assert speech.default_choice(choices) == 0
    assert not any("recommended" in c[2] for c in choices[1:])
    # Only the interactive wizard asks for it.
    assert speech.model_choices("en") is speech.MODELS


def test_choosing_it_runs_the_shared_installer_then_selects_parakeet(
        installable, wizard, monkeypatch):
    calls = []

    def install(**kw):
        calls.append(kw)
        monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    monkeypatch.setattr(fetch, "install_sherpa", install)
    state, shown, downloads = wizard(0)
    assert shown[0][0][0] == speech.PARAKEET_INSTALL_CHOICE and shown[0][1] == 0
    assert calls == [{"quiet": True}]
    assert state.model_name == catalog.PARAKEET
    assert downloads == [catalog.PARAKEET]


def test_installer_failure_falls_back_to_small_and_setup_goes_on(
        installable, wizard, monkeypatch, capsys):
    def install(**kw):
        raise OSError("no network")
    monkeypatch.setattr(fetch, "install_sherpa", install)
    state, _, downloads = wizard(0)
    assert state.model_name == "small"
    assert downloads == ["small"]
    out = capsys.readouterr().out
    assert "Couldn't install sherpa-onnx (no network)" in out


def test_install_that_still_cannot_load_falls_back(installable, wizard, monkeypatch):
    monkeypatch.setattr(fetch, "install_sherpa", lambda **kw: None)  # features unchanged
    state, _, downloads = wizard(0)
    assert state.model_name == "small" and downloads == ["small"]


def test_picking_whisper_does_not_install(installable, wizard, monkeypatch):
    monkeypatch.setattr(fetch, "install_sherpa", lambda **kw: pytest.fail("installed"))
    state, shown, _ = wizard(3)
    assert state.model_name == shown[0][0][3][0] == "small"


def test_non_english_is_unchanged(installable, wizard, monkeypatch):
    assert speech.model_choices("es", offer_install=True) is speech.MODELS
    monkeypatch.setattr(fetch, "install_sherpa", lambda **kw: pytest.fail("installed"))
    state, shown, _ = wizard(2, language="es")
    assert shown[0][0] == speech.MODELS and state.model_name == "small"


def test_already_installed_sherpa_keeps_todays_label(installable, wizard, monkeypatch):
    monkeypatch.setattr(sherpa, "features", lambda: FIXED)
    monkeypatch.setattr(fetch, "install_sherpa", lambda **kw: pytest.fail("installed"))
    assert speech.model_choices("en", offer_install=True)[0] == speech.PARAKEET_CHOICE
    state, _, downloads = wizard(0)
    assert state.model_name == catalog.PARAKEET and downloads == [catalog.PARAKEET]


def test_an_older_sherpa_is_not_swapped_inside_the_wizard(installable, monkeypatch):
    monkeypatch.setattr(sherpa, "features", lambda: SherpaFeatures(installed=True))
    assert speech.model_choices("en", offer_install=True) is speech.MODELS


@pytest.mark.parametrize("where", ["deb", "nix", "uvx", "readonly", "no-pip", "armv7l"])
def test_not_offered_where_the_install_cannot_work(monkeypatch, tmp_path, where):
    import platform
    import sysconfig

    import voiceio.install_env as install_env
    import voiceio.service as service
    monkeypatch.setattr(service, "packaged_by", lambda: "deb" if where == "deb" else None)
    monkeypatch.setattr(sys, "prefix", "/nix/store/abc-voiceio" if where == "nix" else
                        str(tmp_path))
    monkeypatch.setattr(install_env, "running_ephemeral", lambda: where == "uvx")
    monkeypatch.setattr(platform, "machine", lambda: "armv7l" if where == "armv7l" else "x86_64")
    monkeypatch.setattr(fetch, "_pip_command",
                        lambda reinstall=False: None if where == "no-pip" else ["pip"])
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr(sysconfig, "get_paths",
                        lambda: {"purelib": str(site), "platlib": str(site)})
    if where == "readonly":
        monkeypatch.setattr(fetch.os, "access", lambda p, mode: False)
    assert fetch.can_install_sherpa() is False
    assert speech.model_choices("en", offer_install=True) is speech.MODELS


def test_offered_in_a_writable_pip_environment(monkeypatch, tmp_path):
    import platform
    import sysconfig

    import voiceio.install_env as install_env
    import voiceio.service as service
    monkeypatch.setattr(service, "packaged_by", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(install_env, "running_ephemeral", lambda: False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(fetch, "_pip_command", lambda reinstall=False: ["pip"])
    monkeypatch.setattr(sysconfig, "get_paths",
                        lambda: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    assert fetch.can_install_sherpa() is True


def test_noninteractive_setup_installs_nothing(installable, cfg_paths, monkeypatch):  # noqa: F811
    _patch_noninteractive(monkeypatch)
    monkeypatch.setattr(fetch, "install_sherpa", lambda **kw: pytest.fail("installed"))
    assert noninteractive.run_setup_noninteractive({}) == 0
    from voiceio import config
    assert config.load(cfg_paths).model.name == "small"


def test_install_sherpa_uses_the_shared_pip_path_and_refreshes_detection(monkeypatch):
    monkeypatch.setattr(sherpa, "features", REAL_FEATURES)
    REAL_FEATURES()
    assert REAL_FEATURES.cache_info().currsize == 1
    calls = []
    monkeypatch.setattr(fetch, "_pip_install", lambda args, **kw: calls.append((args, kw)))
    fetch.install_sherpa(quiet=True)
    assert calls == [(["--upgrade", fetch.SHERPA_REQUIREMENT], {"quiet": True})]
    assert REAL_FEATURES.cache_info().currsize == 0


def test_quiet_pip_failure_names_pips_last_line(monkeypatch):
    import subprocess
    import types
    monkeypatch.setattr(fetch, "_pip_command", lambda reinstall=False: ["pip", "install"])
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=1, stderr="Collecting x\nERROR: No matching distribution\n"))
    with pytest.raises(OSError, match="No matching distribution"):
        fetch._pip_install(["sherpa-onnx>=2.0.2"], quiet=True)
