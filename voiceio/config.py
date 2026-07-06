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
CORRECTIONS_PATH = CONFIG_DIR / "corrections.json"
FLAGGED_PATH = CONFIG_DIR / "flagged.txt"
HISTORY_PATH = LOG_DIR / "history.jsonl"
RECORDINGS_DIR = LOG_DIR / "recordings"
LOG_PATH = LOG_DIR / "voiceio.log"
PID_PATH = LOG_DIR / "voiceio.pid"


@dataclass
class HotkeyConfig:
    key: str = "ctrl+alt+v"
    backend: str = "auto"


@dataclass
class ModelConfig:
    # "small" fixes most proper-noun/technical-term errors vs "base" and still
    # runs ~5x realtime on a modern CPU; "distil-large-v3" is the quality pick
    # for batch (non-streaming) use.
    name: str = "small"
    language: str = "en"
    device: str = "auto"
    compute_type: str = "int8"
    vocabulary_file: str = ""  # path to vocabulary.txt


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    device: str = "default"
    prebuffer_secs: float = 1.0
    silence_threshold: float = 0.01
    silence_duration: float = 0.6
    auto_stop_silence_secs: float = 5.0
    vad_backend: str = "silero"  # "silero" | "rms"
    vad_threshold: float = 0.5  # Silero speech probability threshold


@dataclass
class OutputConfig:
    method: str = "auto"
    streaming: bool = True
    min_recording_secs: float = 1.5
    cancel_window_secs: float = 0.5
    punctuation_cleanup: bool = True
    number_conversion: bool = True
    voice_input_prefix: str = ""           # e.g. "[voice]" — empty disables


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
class CommandsConfig:
    enabled: bool = True
    editing: bool = False


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = ""          # empty = auto-select first available
    base_url: str = "http://localhost:11434"
    timeout_secs: float = 15.0


@dataclass
class AutocorrectConfig:
    api_key: str = ""                      # API key, or set OPENROUTER_API_KEY env var
    base_url: str = "https://openrouter.ai/api/v1"  # Any OpenAI-compatible endpoint
    model: str = "moonshotai/kimi-k2-0905"   # Model ID (OpenRouter format) — fast, cheap, non-thinking
    timeout_secs: float = 30.0
    # Languages you also dictate in: mined corrections never rewrite words
    # that are real in these (e.g. ["es"] protects Spanish "harina").
    protect_languages: list[str] = field(default_factory=list)


@dataclass
class PostCorrectConfig:
    """Constrained LLM post-correction of final transcripts.

    Reuses the [autocorrect] section's API key / base_url resolution (config
    key → OPENROUTER_API_KEY → OPENAI_API_KEY → ANTHROPIC_API_KEY). Only the
    model can be overridden here; empty falls back to the autocorrect model.
    """
    enabled: bool = False
    model: str = ""              # empty = use [autocorrect].model
    timeout_secs: float = 8.0
    min_words: int = 4           # skip utterances shorter than this


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: str = "auto"         # "auto" | "piper" | "espeak" | "edge-tts"
    hotkey: str = "ctrl+alt+s"   # "s" for speak
    voice: str = ""              # empty = engine default
    speed: float = 1.0           # 0.5–2.0
    model: str = ""              # piper model name, empty = default


@dataclass
class HealthConfig:
    auto_fallback: bool = True


@dataclass
class DataConfig:
    """Local data retention for diagnostics and self-improvement.

    Everything stays on disk under ~/.local/state/voiceio/ — nothing is
    uploaded. retain_audio keeps a WAV per utterance so (audio, final text)
    pairs accumulate for later analysis or fine-tuning.
    """
    retain_audio: bool = True
    max_audio_mb: int = 4096   # prune oldest recordings beyond this
    capture_context: bool = True  # best-effort active-window title


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    autocorrect: AutocorrectConfig = field(default_factory=AutocorrectConfig)
    postcorrect: PostCorrectConfig = field(default_factory=PostCorrectConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    data: DataConfig = field(default_factory=DataConfig)


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
        commands=_build(CommandsConfig, raw.get("commands", {})),
        daemon=_build(DaemonConfig, raw.get("daemon", {})),
        llm=_build(LLMConfig, raw.get("llm", {})),
        autocorrect=_build(AutocorrectConfig, raw.get("autocorrect", {})),
        postcorrect=_build(PostCorrectConfig, raw.get("postcorrect", {})),
        tts=_build(TTSConfig, raw.get("tts", {})),
        health=_build(HealthConfig, raw.get("health", {})),
        data=_build(DataConfig, raw.get("data", {})),
    )
