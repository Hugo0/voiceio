"""Configuration schema, loading, and v1 migration."""
from __future__ import annotations

import dataclasses
import logging
import os
import stat
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
    def _xdg(var: str, default: str) -> Path:
        # XDG spec: ignore unset, empty, or relative values.
        val = os.environ.get(var, "")
        return Path(val) if os.path.isabs(val) else Path.home() / default

    CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / "voiceio"
    LOG_DIR = _xdg("XDG_STATE_HOME", ".local/state") / "voiceio"

CONFIG_PATH = CONFIG_DIR / "config.toml"
CORRECTIONS_PATH = CONFIG_DIR / "corrections.json"
FLAGGED_PATH = CONFIG_DIR / "flagged.txt"
HISTORY_PATH = LOG_DIR / "history.jsonl"
RECORDINGS_DIR = LOG_DIR / "recordings"
LOG_PATH = LOG_DIR / "voiceio.log"
PID_PATH = LOG_DIR / "voiceio.pid"
# The daemon's state (idle/recording/finalizing), so `voiceio service
# restart` can wait for a dictation to finish instead of killing it.
STATE_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio.state"
# Intermediate pipeline data kept for debugging/profiling and future training:
# per-pass streaming decode traces and postcorrect before/after pairs. These
# live in their own JSONL files (not the rotating log) so they survive.
TRACES_PATH = LOG_DIR / "streaming_trace.jsonl"
POSTCORRECT_PAIRS_PATH = LOG_DIR / "postcorrect_pairs.jsonl"
# State of pre-1.0 cloud rule mining, removed at daemon start. The first one
# archived a snippet of dictated text every time a correction fired.
LEGACY_STATE_FILES = (
    LOG_DIR / "corrections_audit.jsonl",
    LOG_DIR / "metrics.jsonl",
    CONFIG_DIR / "audit_state.json",
    CONFIG_DIR / "autocorrect_state.json",
)
# Explicit, per-user record that cloud LLM calls (text, never audio) are allowed.
CONSENT_PATH = CONFIG_DIR / "consent.json"


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
    # CTranslate2 threads per decode. 0 = auto (half the logical cores, 4-8);
    # the library default is a fixed 4, leaving most of a modern CPU idle.
    cpu_threads: int = 0
    # 2 = a second model process for interim previews, so pressing stop never
    # waits behind one (~+300 MB RAM for "small"). 1 = a single process.
    decode_lanes: int = 2
    # sherpa-onnx models only: "greedy_search" (default) or
    # "modified_beam_search" (required for hotwords).
    decoding: str = ""
    vocabulary_file: str = ""  # path to vocabulary.txt
    # Beam width of the passes whose text you keep (freezes and the final
    # tail); interim previews are always greedy. 1 is greedy: faster stop.
    beam_size: int = 5
    # Other languages you dictate in: a correction rule that would rewrite a
    # real word in one of these is flagged (e.g. ["es"] protects "harina").
    protect_languages: list[str] = field(default_factory=list)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    device: str = "default"
    prebuffer_secs: float = 1.0
    silence_threshold: float = 0.01
    silence_duration: float = 0.6
    auto_stop_silence_secs: float = 5.0
    # Safety net for a muted mic or a forgotten hotkey: if NO speech is heard
    # for this long from the start, stop and notify. The 5s auto-stop above
    # only fires once speech has been heard, so it can't catch a dead mic.
    # 0 disables.
    auto_stop_no_speech_secs: float = 20.0
    vad_backend: str = "silero"  # "silero" | "rms"
    vad_threshold: float = 0.5  # Silero speech probability threshold


@dataclass
class OutputConfig:
    method: str = "auto"
    streaming: bool = True
    punctuation_cleanup: bool = True
    number_conversion: bool = True
    # Strip spoken disfluencies so dictation reads as writing. Delete-only and
    # meaning-preserving: a free regex layer removes filler sounds (um/uh/er)
    # and duplicate re-decode sentences on every pass; when [postcorrect] is
    # also enabled, its LLM pass additionally removes false starts and filler
    # "like", guarded so it can only delete — never add, rephrase, or reorder.
    # On by default: nobody wants "um" in their notes. Set false for verbatim.
    remove_disfluencies: bool = True
    voice_input_prefix: str = ""           # e.g. "[voice]" — empty disables
    # Incremental finalization: once the un-finalized audio tail grows past
    # this many seconds, it is cut at the nearest interior speech pause,
    # beam-decoded and frozen during recording, so interim passes and the
    # stop-time final decode only cover the short remaining tail. Smaller
    # chunks decode faster (less worker blocking) at slightly more boundary
    # risk. 0 disables (full re-decode at stop).
    streaming_freeze_secs: float = 8.0
    # Mirror transcribed text to the system clipboard so it can be pasted:
    #   "off"   — never
    #   "final" — the corrected final text, once ready
    #   "live"  — also the best-so-far text at each streaming update and the
    #             instant recording stops (pasteable while the final decode runs)
    copy_to_clipboard: str = "final"


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
class CloudConfig:
    """The OpenAI-compatible endpoint for the opt-in cloud features (text only,
    never audio). Was `[autocorrect]` before 1.0; old files are migrated."""
    api_key: str = ""                      # API key, or set OPENROUTER_API_KEY env var
    base_url: str = "https://openrouter.ai/api/v1"  # Any OpenAI-compatible endpoint
    model: str = "moonshotai/kimi-k2-0905"   # Model ID (OpenRouter format) — fast, cheap, non-thinking
    timeout_secs: float = 30.0


@dataclass
class PostCorrectConfig:
    """Constrained LLM post-correction of final transcripts.

    Uses the [cloud] section's API key / base_url resolution (config key →
    OPENROUTER_API_KEY → OPENAI_API_KEY → ANTHROPIC_API_KEY). Only the model
    can be overridden here; empty falls back to the [cloud] model.
    """
    enabled: bool = False
    # Latency-critical (blocks the commit): default to a fast cheap model.
    # Bake-off on the real correction prompt: gemini-2.5-flash-lite 0.44s
    # with fixes identical to kimi-k2's 2.0s. OpenRouter model id — on a
    # custom base_url set a model your endpoint serves (empty = use
    # [cloud].model).
    model: str = "google/gemini-2.5-flash-lite"
    # It blocks the commit: past this the original text is kept.
    timeout_secs: float = 3.0
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
class HistoryConfig:
    """Retention policy for the transcription history log (history.jsonl).

    All fields stay local. `enabled=False` stops new entries from being
    written. Pruning runs on daemon start and periodically as history grows.
    """
    enabled: bool = True
    max_entries: int = 0   # 0 = unlimited
    max_age_days: int = 0  # 0 = keep forever


@dataclass
class HealthConfig:
    auto_fallback: bool = True


@dataclass
class DataConfig:
    """Local data retention for diagnostics and self-improvement.

    Opt-in: nothing here is kept unless you turn it on (`voiceio setup` asks).
    Everything stays on disk under ~/.local/state/voiceio/ — nothing is
    uploaded. retain_audio keeps a WAV per utterance so (audio, final text)
    pairs accumulate for `voiceio learn` to fine-tune on.
    """
    retain_audio: bool = False
    max_audio_mb: int = 4096   # prune oldest recordings beyond this
    # Never let retention squeeze the disk: skip saving and prune harder
    # when the filesystem has less than this much free space.
    min_free_gb: float = 5.0
    # Active-window title, stored with each history row and given to the
    # cloud pass as context when that is on.
    capture_context: bool = False
    # Per-pass streaming decode traces + postcorrect before/after pairs
    # (streaming_trace.jsonl / postcorrect_pairs.jsonl) for debugging,
    # profiling, and future fine-tuning.
    capture_intermediates: bool = False


@dataclass
class LearnConfig:
    # A command run after every model switch (promote, auto-promote,
    # rollback), with the model in $VOICEIO_MODEL — e.g. to push a fine-tune
    # to another machine: scripts/push_model.sh HOST. Empty: nothing runs.
    on_promote: str = ""
    # When nightly learning runs (`voiceio learn schedule on`): a time of day,
    # and "day" or "week" (weekly runs on Sunday).
    at: str = "03:30"
    every: str = "day"
    # It only starts on a quiet machine on mains power: 1-minute load per CPU
    # at most max_load. It waits up to wait_hours for that, then tries again
    # at the next scheduled run.
    max_load: float = 0.25
    wait_hours: float = 3.0
    # What a scheduled run trains (`voiceio learn maintain`'s flags override):
    # the Whisper size or Hugging Face checkpoint to fine-tune, e.g. "medium.en";
    # the model that labels new clips (empty: large-v3-turbo); epochs per run;
    # and how many new labeled clips it waits for before retraining.
    base: str = "small"
    teacher: str = ""
    epochs: int = 4
    min_new: int = 50


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
    cloud: CloudConfig = field(default_factory=CloudConfig)
    postcorrect: PostCorrectConfig = field(default_factory=PostCorrectConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    data: DataConfig = field(default_factory=DataConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)


_REMOVED_KEYS = {
    "output": ("min_recording_secs", "cancel_window_secs"),  # never read
    # Cloud mining was removed in 1.0 (387 mined rules, 4 ever fired).
    "autocorrect": ("mine_corrections",),
}


def _migrate_v1(raw: dict) -> dict:
    """Migrate v1 config values to v2."""
    for section, keys in _REMOVED_KEYS.items():
        for key in keys:
            if key in raw.get(section, {}):
                del raw[section][key]
                log.info("Config: ignoring removed option %s.%s", section, key)

    # [autocorrect] held only the cloud endpoint once mining was gone; a
    # [cloud] key written since wins. Its protect_languages is about
    # correction safety, so it joins the language setting in [model].
    if "autocorrect" in raw:
        old = raw.pop("autocorrect")
        protect = old.pop("protect_languages", None)
        if protect and "protect_languages" not in raw.get("model", {}):
            raw.setdefault("model", {})["protect_languages"] = protect
        raw["cloud"] = old | raw.get("cloud", {})

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
        cloud=_build(CloudConfig, raw.get("cloud", {})),
        postcorrect=_build(PostCorrectConfig, raw.get("postcorrect", {})),
        tts=_build(TTSConfig, raw.get("tts", {})),
        health=_build(HealthConfig, raw.get("health", {})),
        data=_build(DataConfig, raw.get("data", {})),
        history=_build(HistoryConfig, raw.get("history", {})),
        learn=_build(LearnConfig, raw.get("learn", {})),
    )


# ── File permission hardening ────────────────────────────────────────────
# Everything voiceio persists is either user content (transcripts, audio) or
# a secret (API key in config.toml). On multi-user machines the default umask
# can leave these world-readable, which contradicts the "your data, in files
# you own" promise. We keep files at 0600 and dirs at 0700.

_SECURE_FILE = 0o600
_SECURE_DIR = 0o700


def _content_files() -> list[Path]:
    """Files that may contain user content or secrets (0600)."""
    return [
        CONFIG_PATH,
        CORRECTIONS_PATH,
        FLAGGED_PATH,
        HISTORY_PATH,
        TRACES_PATH,
        POSTCORRECT_PAIRS_PATH,
        CONSENT_PATH,
    ]


def _content_dirs() -> list[Path]:
    """Directories that hold user content (0700)."""
    return [CONFIG_DIR, LOG_DIR, RECORDINGS_DIR]


def _chmod(path: Path, mode: int) -> None:
    """chmod, ignoring missing files and unsupported platforms (Windows)."""
    if sys.platform == "win32":
        return
    try:
        path.chmod(mode)
    except (OSError, NotImplementedError):
        log.debug("Could not chmod %s", path, exc_info=True)


def secure_write(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to a file guaranteed to be 0600 with a 0700 parent dir.

    The temp file is created 0600 before any byte is written and then renamed
    over the target, so there is no world-readable window and a crash never
    leaves a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod(path.parent, _SECURE_DIR)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECURE_FILE)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        _chmod(tmp, _SECURE_FILE)  # O_CREAT mode is ignored for existing files
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def set_value(section: str, key: str, value: str | int | float | bool,
              path: Path | None = None) -> None:
    """Set one key in config.toml, keeping the user's comments and layout.

    Replaces an existing (or commented-out) `key =` line inside `[section]`,
    else adds it under the header, else appends the section. Written 0600.
    """
    import json as _json
    import re

    path = path or CONFIG_PATH
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    rendered = (str(value).lower() if isinstance(value, bool)
                else str(value) if isinstance(value, (int, float))
                else _json.dumps(value))  # TOML basic strings share JSON escaping
    header = rf"^\[{re.escape(section)}\][ \t]*(#.*)?$"
    m = re.search(header, content, re.MULTILINE)
    if m is None:
        sep = "" if not content or content.endswith("\n\n") else ("\n" if content.endswith("\n") else "\n\n")
        content += f"{sep}[{section}]\n{key} = {rendered}\n"
    else:
        nxt = re.search(r"^\[", content[m.end():], re.MULTILINE)
        end = m.end() + nxt.start() if nxt else len(content)
        body = content[m.end():end]
        live = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*=.*$", re.MULTILINE)
        commented = re.compile(rf"^#[ \t]*{re.escape(key)}[ \t]*=.*$", re.MULTILINE)
        target = live if live.search(body) else commented if commented.search(body) else None
        if target is not None:
            body = target.sub(lambda _: f"{key} = {rendered}", body, count=1)
        else:
            body = f"\n{key} = {rendered}" + body
        content = content[:m.end()] + body + content[end:]
    # Never leave a file the daemon can't parse.
    tomllib.loads(content)
    secure_write(path, content)


def check_permissions() -> list[tuple[Path, int, int]]:
    """Return [(path, actual_mode, expected_mode)] for anything too permissive.

    Only existing paths are reported. A path is flagged when it grants any
    group/other bits beyond the expected 0600 (files) / 0700 (dirs).
    """
    issues: list[tuple[Path, int, int]] = []
    if sys.platform == "win32":
        return issues
    for f in _content_files():
        if f.is_file():
            mode = stat.S_IMODE(f.stat().st_mode)
            if mode & 0o077:
                issues.append((f, mode, _SECURE_FILE))
    for d in _content_dirs():
        if d.is_dir():
            mode = stat.S_IMODE(d.stat().st_mode)
            if mode & 0o077:
                issues.append((d, mode, _SECURE_DIR))
    # Retained WAVs live under RECORDINGS_DIR.
    if RECORDINGS_DIR.is_dir():
        for wav in RECORDINGS_DIR.glob("*.wav"):
            mode = stat.S_IMODE(wav.stat().st_mode)
            if mode & 0o077:
                issues.append((wav, mode, _SECURE_FILE))
    return issues


def harden_permissions() -> int:
    """Tighten permissions on all existing voiceio state. Idempotent + cheap.

    Called on daemon start and after config writes. Returns the number of
    paths whose mode was changed (best-effort; never raises).
    """
    if sys.platform == "win32":
        return 0
    changed = 0
    for d in _content_dirs():
        if d.is_dir() and stat.S_IMODE(d.stat().st_mode) != _SECURE_DIR:
            _chmod(d, _SECURE_DIR)
            changed += 1
    for f in _content_files():
        if f.is_file() and stat.S_IMODE(f.stat().st_mode) != _SECURE_FILE:
            _chmod(f, _SECURE_FILE)
            changed += 1
    if RECORDINGS_DIR.is_dir():
        for wav in RECORDINGS_DIR.glob("*.wav"):
            if stat.S_IMODE(wav.stat().st_mode) != _SECURE_FILE:
                _chmod(wav, _SECURE_FILE)
                changed += 1
    return changed


def remove_legacy_state() -> None:
    """Delete pre-1.0 mining state nothing reads any more."""
    for path in LEGACY_STATE_FILES:
        path.unlink(missing_ok=True)
