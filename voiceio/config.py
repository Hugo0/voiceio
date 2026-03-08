from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "voiceio"
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "hotkey": {"key": "Super_r"},
    "model": {
        "name": "base",
        "language": "en",
        "device": "auto",
        "compute_type": "int8",
    },
    "audio": {
        "sample_rate": 16000,
        "device": "default",
    },
    "output": {
        "method": "xdotool",
    },
    "feedback": {
        "sound_enabled": True,
    },
    "tray": {
        "enabled": False,
    },
    "daemon": {
        "log_level": "INFO",
    },
}


@dataclass
class HotkeyConfig:
    key: str = "Super_r"


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


@dataclass
class OutputConfig:
    method: str = "xdotool"  # "xdotool" or "xclip"


@dataclass
class FeedbackConfig:
    sound_enabled: bool = True


@dataclass
class TrayConfig:
    enabled: bool = False


@dataclass
class DaemonConfig:
    log_level: str = "INFO"


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    raw = DEFAULTS.copy()

    if path.exists():
        with open(path, "rb") as f:
            user = tomllib.load(f)
        raw = _deep_merge(raw, user)

    return Config(
        hotkey=HotkeyConfig(**raw.get("hotkey", {})),
        model=ModelConfig(**raw.get("model", {})),
        audio=AudioConfig(**raw.get("audio", {})),
        output=OutputConfig(**raw.get("output", {})),
        feedback=FeedbackConfig(**raw.get("feedback", {})),
        tray=TrayConfig(**raw.get("tray", {})),
        daemon=DaemonConfig(**raw.get("daemon", {})),
    )
