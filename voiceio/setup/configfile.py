"""Merge-preserving config.toml writer used by both setup paths."""
from __future__ import annotations

from voiceio.config import CONFIG_DIR, CONFIG_PATH
from voiceio.ui import DIM, GREEN, RESET


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'"{_toml_escape(v)}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    # Unknown scalar types round-trip as their string form.
    return f'"{_toml_escape(str(v))}"'


def _dump_toml(data: dict) -> str:
    """Serialize a 2-level config dict (top-level tables) back to TOML.

    Preserves every key present in ``data`` — including hand-edited and unknown
    keys — so re-running setup never drops user customizations.
    """
    lines: list[str] = ["# voiceio configuration — edit freely; 'voiceio setup' merges, never overwrites"]
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    if scalars:
        lines.append("")
        for k, v in scalars.items():
            lines.append(f"{k} = {_toml_value(v)}")
    for name, tbl in tables.items():
        lines.append("")
        lines.append(f"[{name}]")
        for k, v in tbl.items():
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _load_raw_config() -> dict:
    """Load the existing config.toml as a raw nested dict (empty if absent)."""
    import tomllib
    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def _record_cloud_consent() -> None:
    """Record that the user opted into a cloud LLM feature."""
    from voiceio.consent import record_consent
    record_consent(source="wizard")


def _write_config(
    model: str, language: str, hotkey: str, method: str, streaming: bool, backend: str,
    sound_enabled: bool = True, notify_clipboard: bool = False,
    tray_enabled: bool = False,
    commands_enabled: bool = True,
    punctuation_cleanup: bool = True,
    number_conversion: bool = True,
    llm_enabled: bool = False,
    llm_model: str = "",
    cloud_api_key: str = "",
    cloud_base_url: str = "",
    cloud_model: str = "",
    tts_enabled: bool = True,
    tts_engine: str = "auto",
    tts_speed: float = 1.0,
    voice_input_prefix: str = "",
    retain_audio: bool | None = None,
    quiet: bool = False,
) -> None:
    """Merge wizard answers into the existing config, preserving everything else.

    Only the keys the wizard actually asks about are overwritten. Hand-edited
    keys, unknown keys, and secrets (e.g. a saved API key when the user didn't
    enter a new one) are preserved verbatim.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    from voiceio.config import _migrate_v1
    cfg = _migrate_v1(_load_raw_config())   # e.g. [autocorrect] → [cloud]

    def section(name: str) -> dict:
        return cfg.setdefault(name, {})

    # Keys the wizard owns (asked about) → override only these.
    section("hotkey").update({"key": hotkey, "backend": backend})
    section("model").update({"name": model, "language": language})
    section("output").update({
        "method": method,
        "streaming": streaming,
        "punctuation_cleanup": punctuation_cleanup,
        "number_conversion": number_conversion,
        "voice_input_prefix": voice_input_prefix,
    })
    section("commands")["enabled"] = commands_enabled
    section("feedback").update({
        "sound_enabled": sound_enabled,
        "notify_clipboard": notify_clipboard,
    })
    section("tray")["enabled"] = tray_enabled
    section("llm").update({"enabled": llm_enabled, "model": llm_model})
    section("tts").update({
        "enabled": tts_enabled,
        "engine": tts_engine,
        "speed": tts_speed,
    })

    if retain_audio is not None:   # only when the user was asked
        section("data")["retain_audio"] = retain_audio

    # Cloud endpoint: only touch these keys when the wizard collected a NEW
    # key this run. Otherwise a saved secret / hand-edited endpoint is kept.
    if cloud_api_key:
        section("cloud").update({
            "api_key": cloud_api_key,
            "base_url": cloud_base_url or "https://openrouter.ai/api/v1",
            "model": cloud_model or "moonshotai/kimi-k2-0905",
        })

    # May hold a cloud API key: written 0600 from the first byte.
    from voiceio.config import secure_write
    secure_write(CONFIG_PATH, _dump_toml(cfg))
    if not quiet:
        print(f"\n  {GREEN}✓{RESET} Config saved to {DIM}{CONFIG_PATH}{RESET}")
