"""TTS engine selection chain — same pattern as hotkeys/chain.py."""
from __future__ import annotations

import logging

from voiceio.backends import ProbeResult
from voiceio.config import TTSConfig

log = logging.getLogger(__name__)

# Engine classes by name
_ENGINES = {
    "espeak": "voiceio.tts.espeak:EspeakEngine",
    "piper": "voiceio.tts.piper_engine:PiperEngine",
    "edge-tts": "voiceio.tts.edge_engine:EdgeEngine",
}

# Preference order when engine = "auto"
_AUTO_ORDER = ["piper", "edge-tts", "espeak"]


def _create(name: str, cfg: TTSConfig):
    """Instantiate an engine by name."""
    qualname = _ENGINES[name]
    module_path, cls_name = qualname.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    if name == "piper":
        return cls(model=cfg.model)
    return cls()


def probe_all(cfg: TTSConfig) -> list[tuple[str, ProbeResult]]:
    """Probe all engines in preference order. For doctor/health check."""
    results = []
    order = _AUTO_ORDER if cfg.engine == "auto" else [cfg.engine]
    for name in order:
        if name not in _ENGINES:
            results.append((name, ProbeResult(ok=False, reason=f"unknown engine '{name}'")))
            continue
        try:
            engine = _create(name, cfg)
            results.append((name, engine.probe()))
        except Exception as e:
            results.append((name, ProbeResult(ok=False, reason=str(e))))
    return results


def select(cfg: TTSConfig):
    """Select the first working TTS engine.

    Returns the engine instance, or None if none available.
    """
    if cfg.engine != "auto":
        if cfg.engine not in _ENGINES:
            log.warning("TTS: unknown engine '%s'", cfg.engine)
            return None
        engine = _create(cfg.engine, cfg)
        probe = engine.probe()
        if probe.ok:
            log.info("TTS: using %s", cfg.engine)
            return engine
        log.warning("TTS: %s unavailable: %s", cfg.engine, probe.reason)
        return None

    for name in _AUTO_ORDER:
        try:
            engine = _create(name, cfg)
            probe = engine.probe()
            if probe.ok:
                log.info("TTS: using %s", name)
                return engine
            log.debug("TTS: %s unavailable: %s", name, probe.reason)
        except Exception as e:
            log.debug("TTS: %s failed: %s", name, e)

    log.warning("TTS: no engine available")
    return None
