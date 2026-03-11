"""Load custom vocabulary for Whisper initial_prompt conditioning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.config import ModelConfig

log = logging.getLogger(__name__)

MAX_VOCAB_CHARS = 400  # Leave room for history in the ~224 token budget


def load_vocabulary(model_cfg: ModelConfig) -> str:
    """Load vocabulary terms and return a comma-separated string.

    Checks model_cfg.vocabulary_file first, then the default location.
    Returns empty string if no vocabulary file found.
    """
    from voiceio.config import CONFIG_DIR

    if model_cfg.vocabulary_file:
        path = Path(model_cfg.vocabulary_file).expanduser()
    else:
        path = CONFIG_DIR / "vocabulary.txt"

    if not path.exists():
        if model_cfg.vocabulary_file:
            log.warning("Vocabulary file not found: %s", path)
        return ""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        log.warning("Failed to read vocabulary file: %s", e)
        return ""

    terms = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)

    if not terms:
        return ""

    result = ", ".join(terms)
    if len(result) > MAX_VOCAB_CHARS:
        # Truncate by dropping terms from the end
        while len(result) > MAX_VOCAB_CHARS and terms:
            terms.pop()
            result = ", ".join(terms)
        log.warning("Vocabulary truncated to %d terms (%d chars)", len(terms), len(result))

    log.info("Loaded %d vocabulary terms from %s", len(terms), path)
    return result
