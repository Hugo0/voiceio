"""Configuration schema, loading, and v1 migration."""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PYPI_NAME = "python-voiceio"

if sys.platform == "win32":
    _APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "voiceio"
    CONFIG_DIR = _APP_DIR / "config"
    LOG_DIR = _APP_DIR / "logs"
else:
    CONFIG_DIR = Path.home() / ".config" / "voiceio"
    LOG_DIR = Path.home() / ".local" / "state" / "voiceio"

CONFIG_PATH = CONFIG_DIR / "config.toml"
LOG_PATH = LOG_DIR / "voiceio.log"
PID_PATH = LOG_DIR / "voiceio.pid"


@dataclass
class HotkeyConfig:
    key: str = "ctrl+alt+v"
    backend: str = "auto"


@dataclass
class ModelConfig:
    name: str = "base"
    language: str = "en"
    device: str = "auto"
    compute_type: str = "int8"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    device: str = "default"
    prebuffer_secs: float = 1.0
    silence_threshold: float = 0.01
    silence_duration: float = 0.6
    auto_stop_silence_secs: float = 5.0


@dataclass
class OutputConfig:
    method: str = "auto"
    streaming: bool = True
    min_recording_secs: float = 1.5
    cancel_window_secs: float = 0.5


@dataclass
class FeedbackConfig:
    sound_enabled: bool = True
    notify_clipboard: bool = False


@dataclass
class TrayConfig:
    enabled: bool = False


@dataclass
class DaemonConfig:
    log_level: str = "INFO"


@dataclass
class HealthConfig:
    auto_fallback: bool = True


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    health: HealthConfig = field(default_factory=HealthConfig)


def _migrate_v1(raw: dict) -> dict:
    """Migrate v1 config values to v2."""
    # Remove deprecated cpu_threads
    if "model" in raw and "cpu_threads" in raw["model"]:
        del raw["model"]["cpu_threads"]
        log.info("Config migration: removed deprecated model.cpu_threads")

    # Migrate old output methods
    if "output" in raw and "method" in raw["output"]:
        method = raw["output"]["method"]
        if method in ("xclip", "wl-copy"):
            raw["output"]["method"] = "clipboard"
            log.info("Config migration: output.method '%s' → 'clipboard'", method)

    # Migrate old hotkey backend names
    if "hotkey" in raw and "backend" in raw["hotkey"]:
        if raw["hotkey"]["backend"] == "x11":
            raw["hotkey"]["backend"] = "pynput"
            log.info("Config migration: hotkey.backend 'x11' → 'pynput'")

    return raw


def _build(cls, section: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in section.items() if k in valid}
    unknown = set(section) - valid
    if unknown:
        log.warning("Ignoring unknown config keys in [%s]: %s", cls.__name__, ", ".join(unknown))
    return cls(**filtered)


def load(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    raw: dict = {}

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        raw = _migrate_v1(raw)
    except FileNotFoundError:
        pass

    return Config(
        hotkey=_build(HotkeyConfig, raw.get("hotkey", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        audio=_build(AudioConfig, raw.get("audio", {})),
        output=_build(OutputConfig, raw.get("output", {})),
        feedback=_build(FeedbackConfig, raw.get("feedback", {})),
        tray=_build(TrayConfig, raw.get("tray", {})),
        daemon=_build(DaemonConfig, raw.get("daemon", {})),
        health=_build(HealthConfig, raw.get("health", {})),
    )
