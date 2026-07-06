"""Security & privacy hardening: file permissions, udev rule, consent, retention."""
from __future__ import annotations

import json
import stat
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

from voiceio import config, consent, history, llm_api, retention
from voiceio.config import AutocorrectConfig, HistoryConfig

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")


def _mode(p):
    return stat.S_IMODE(p.stat().st_mode)


# ── File permission hardening ────────────────────────────────────────────

@posix_only
def test_secure_write_creates_0600_file_in_0700_dir(tmp_path):
    target = tmp_path / "sub" / "secret.toml"
    config.secure_write(target, "api_key = \"x\"\n")
    assert target.read_text().startswith("api_key")
    assert _mode(target) == 0o600
    assert _mode(target.parent) == 0o700


@posix_only
def test_history_append_writes_0600(tmp_path, monkeypatch):
    p = tmp_path / "history.jsonl"
    history.append("hello world", path=p, cfg=HistoryConfig())
    assert _mode(p) == 0o600


@posix_only
def test_save_audio_writes_0600_wav_in_0700_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "rec")
    monkeypatch.setattr(retention, "RECORDINGS_DIR", tmp_path / "rec")
    cfg = config.DataConfig(retain_audio=True)
    audio = np.zeros(1600, dtype=np.float32)
    name = retention.save_audio(audio, time.time(), cfg)
    assert name is not None
    wav = (tmp_path / "rec") / name
    assert _mode(wav) == 0o600
    assert _mode(tmp_path / "rec") == 0o700


@posix_only
def test_harden_permissions_tightens_existing_loose_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config" / "config.toml")
    (tmp_path / "config").mkdir()
    loose = tmp_path / "config" / "config.toml"
    loose.write_text("x")
    loose.chmod(0o644)
    (tmp_path / "config").chmod(0o755)

    changed = config.harden_permissions()

    assert changed >= 2
    assert _mode(loose) == 0o600
    assert _mode(tmp_path / "config") == 0o700


@posix_only
def test_check_permissions_flags_loose_and_clears_after_harden(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config" / "config.toml")
    (tmp_path / "config").mkdir(mode=0o755)
    loose = tmp_path / "config" / "config.toml"
    loose.write_text("x")
    loose.chmod(0o644)

    issues = config.check_permissions()
    flagged = {str(p) for p, _, _ in issues}
    assert str(loose) in flagged

    config.harden_permissions()
    assert config.check_permissions() == []


# ── udev rule (replaces sudo chmod 0666 /dev/uinput) ─────────────────────

def test_uinput_udev_rule_content():
    from voiceio.typers import ydotool
    rule = ydotool.UINPUT_UDEV_RULE
    assert 'KERNEL=="uinput"' in rule
    assert "uaccess" in rule
    assert "0666" not in rule  # no world-writable device
    assert ydotool.UINPUT_UDEV_RULE_PATH.startswith("/etc/udev/rules.d/")


def test_uinput_install_cmd_uses_sudo_not_chmod_0666():
    from voiceio.typers import ydotool
    cmd = ydotool.uinput_udev_install_cmd()
    joined = " ".join(cmd)
    assert "sudo tee" in joined
    assert "udevadm control --reload-rules" in joined
    assert "chmod 0666" not in joined


def test_ydotool_probe_offers_udev_not_chmod(monkeypatch):
    from voiceio.typers import ydotool
    monkeypatch.setattr(ydotool.shutil, "which", lambda _: "/usr/bin/ydotool")
    monkeypatch.setattr(ydotool, "_needs_daemon", lambda: False)
    monkeypatch.setattr(ydotool, "_has_uinput_access", lambda: False)
    result = ydotool.YdotoolTyper().probe()
    assert not result.ok
    assert "chmod 0666" not in " ".join(result.fix_cmd)
    assert "sudo tee" in " ".join(result.fix_cmd)


# ── doctor --fix consent prompt for privileged actions ───────────────────

def test_is_privileged_cmd():
    from voiceio.cli import _is_privileged_cmd
    assert _is_privileged_cmd(["sudo", "modprobe", "uinput"])
    assert _is_privileged_cmd(["sh", "-c", "curl x | sh"])
    assert not _is_privileged_cmd(["ydotoold"])


def test_confirm_privileged_declined_by_default(capsys):
    from voiceio.cli import _confirm_privileged
    with patch("builtins.input", return_value=""):
        assert _confirm_privileged("Fix ydotool", "sudo tee /etc/...") is False
    out = capsys.readouterr().out
    assert "sudo tee /etc/..." in out  # exact command shown


def test_confirm_privileged_accepts_yes():
    from voiceio.cli import _confirm_privileged
    with patch("builtins.input", return_value="y"):
        assert _confirm_privileged("Fix", "sudo x") is True


# ── history retention ────────────────────────────────────────────────────

def _write_entries(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_history_prune_by_count(tmp_path):
    p = tmp_path / "history.jsonl"
    _write_entries(p, [{"ts": i, "text": f"entry {i}"} for i in range(10)])
    removed = history.prune(HistoryConfig(max_entries=3), path=p)
    assert removed == 7
    kept = [json.loads(line) for line in p.read_text().splitlines()]
    assert [e["text"] for e in kept] == ["entry 7", "entry 8", "entry 9"]


def test_history_prune_by_age(tmp_path):
    p = tmp_path / "history.jsonl"
    now = time.time()
    _write_entries(p, [
        {"ts": now - 40 * 86400, "text": "old"},
        {"ts": now - 5 * 86400, "text": "recent"},
    ])
    removed = history.prune(HistoryConfig(max_age_days=30), path=p)
    assert removed == 1
    kept = [json.loads(line) for line in p.read_text().splitlines()]
    assert [e["text"] for e in kept] == ["recent"]


def test_history_prune_noop_when_unlimited(tmp_path):
    p = tmp_path / "history.jsonl"
    _write_entries(p, [{"ts": i, "text": str(i)} for i in range(5)])
    assert history.prune(HistoryConfig(), path=p) == 0
    assert len(p.read_text().splitlines()) == 5


def test_history_append_disabled_skips_write(tmp_path):
    p = tmp_path / "history.jsonl"
    history.append("nope", path=p, cfg=HistoryConfig(enabled=False))
    assert not p.exists()


# ── cloud consent gate ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_consent_warned():
    llm_api._consent_warned = False
    yield
    llm_api._consent_warned = False


def test_consent_record_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_PATH", tmp_path / "consent.json")
    assert consent.has_cloud_consent() is False
    consent.record_consent(source="test")
    assert consent.has_cloud_consent() is True


def test_cloud_call_blocked_without_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_PATH", tmp_path / "consent.json")
    cfg = AutocorrectConfig(api_key="", base_url="https://openrouter.ai/api/v1")
    with patch.object(llm_api, "_openai_request") as req:
        # key comes from env-like param, but cfg.api_key is empty → no consent
        out = llm_api.chat(cfg, "sys", "hi", api_key="env-key")
    assert out is None
    req.assert_not_called()


def test_cloud_call_allowed_with_config_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_PATH", tmp_path / "consent.json")
    cfg = AutocorrectConfig(api_key="sk-configured", base_url="https://openrouter.ai/api/v1")
    with patch.object(llm_api, "_openai_request", return_value="corrected") as req:
        out = llm_api.chat(cfg, "sys", "hi")
    assert out == "corrected"
    req.assert_called_once()
    # configured key is treated as consent and recorded
    assert consent.has_cloud_consent() is True


def test_cloud_call_allowed_after_explicit_consent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_PATH", tmp_path / "consent.json")
    consent.record_consent(source="wizard")
    cfg = AutocorrectConfig(api_key="", base_url="https://openrouter.ai/api/v1")
    with patch.object(llm_api, "_openai_request", return_value="ok") as req:
        out = llm_api.chat(cfg, "sys", "hi", api_key="env-key")
    assert out == "ok"
    req.assert_called_once()


def test_local_endpoint_not_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONSENT_PATH", tmp_path / "consent.json")
    cfg = AutocorrectConfig(api_key="", base_url="http://localhost:11434/v1")
    with patch.object(llm_api, "_openai_request", return_value="local") as req:
        out = llm_api.chat(cfg, "sys", "hi", api_key="ollama")
    assert out == "local"
    req.assert_called_once()  # local Ollama never needs cloud consent
