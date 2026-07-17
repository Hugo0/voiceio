"""Exact Whisper token counting.

Every budget in this codebase used to be a character count with a hand-guessed
chars→tokens ratio, and the guesses were wrong in the direction that matters:
proper nouns — the whole point of a custom vocabulary — tokenize at ~2.6
chars/token against the ~4.0 of ordinary prose. The old 600-char hotwords cap
was really 238 tokens, already past faster-whisper's 223-token ceiling, so the
library silently dropped the tail of the list.

Counting for real is cheap: `tokenizers` is already a faster-whisper dependency
and `tokenizer.json` ships inside the cached model snapshot, so this needs no
model weights and no network. Measured: ~53ms to load, ~17ms to encode a full
vocabulary — and callers only pay it when the vocabulary actually changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# faster-whisper truncates hotwords at `max_length // 2 - 1` (transcribe.py).
# Anything past this is silently discarded by the library.
HOTWORDS_TOKEN_CAP = 223

# Whisper's total sequence budget, shared by hotwords + initial_prompt + OUTPUT.
MAX_SEQUENCE_TOKENS = 448

# Tokens reserved for the transcription itself. A decode window is at most 30s
# of speech: ~100 words at a fast 200wpm, ~135 tokens, plus timestamps. 200 is
# that with headroom. Starve this and Whisper truncates the user mid-sentence —
# silently, which is how it went unnoticed before.
OUTPUT_RESERVE_TOKENS = 200

# Ceiling on initial_prompt + per-call freeze context combined. These are joined
# in Transcriber.transcribe and were previously uncapped: PromptBuilder's 300
# chars plus streaming's 400-char context ran ~171 tokens, which — on top of a
# 223-token hotwords list — left ~50 tokens of output. Freeze chunks hold 45-60
# words; they were being cut off.
PROMPT_TOKEN_BUDGET = 120


def truncate_to_tokens(text: str, max_tokens: int, model_name: str) -> str:
    """Trim `text` from the FRONT to fit `max_tokens`.

    Front-trimming because prompts are conditioning context: the words nearest
    the audio matter most, so the oldest context is what should go.
    """
    if not text or max_tokens <= 0:
        return "" if max_tokens <= 0 else text
    if count_tokens(text, model_name) <= max_tokens:
        return text
    # Binary search the cut point rather than encoding once per character.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[mid:], model_name) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    cut = text[lo:]
    # Snap to a word boundary so we never feed a half-word fragment.
    space = cut.find(" ")
    return cut[space + 1:] if 0 <= space < 20 else cut

# Fallback ratio when no tokenizer is available. Measured over Hugo's actual
# vocabulary (proper nouns/technical terms), NOT prose — deliberately
# pessimistic so an estimate never over-fills the budget.
_CHARS_PER_TOKEN = 2.6

_cache: dict[str, object] = {}
_warned = False


def _find_tokenizer(model_name: str) -> Path | None:
    """Locate tokenizer.json inside the cached model snapshot.

    Cache-only by construction: we glob the HF cache rather than asking
    huggingface_hub to resolve anything, because a cached model must never
    depend on the network (see worker.load_model).
    """
    p = Path(model_name).expanduser()
    if p.is_dir():  # explicit local model directory
        cand = p / "tokenizer.json"
        return cand if cand.exists() else None

    from huggingface_hub.constants import HF_HUB_CACHE

    # faster-whisper resolves bare names like "small" to Systran/faster-whisper-*.
    repo = model_name if "/" in model_name else f"Systran/faster-whisper-{model_name}"
    stem = "models--" + repo.replace("/", "--")
    hits = sorted(Path(HF_HUB_CACHE).glob(f"{stem}/snapshots/*/tokenizer.json"))
    return hits[-1] if hits else None


def _tokenizer(model_name: str):
    """Lazily load and cache the tokenizer for `model_name`, or None."""
    if model_name in _cache:
        return _cache[model_name]

    tok = None
    try:
        path = _find_tokenizer(model_name)
        if path is not None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(path))
    except Exception as e:  # noqa: BLE001 — never break dictation over a budget
        log.debug("Tokenizer unavailable for '%s': %s", model_name, e)

    _cache[model_name] = tok
    return tok


def count_tokens(text: str, model_name: str) -> int:
    """Count Whisper tokens in `text`, estimating if no tokenizer is cached.

    The leading space matches how faster-whisper encodes hotwords/prompts
    (`tokenizer.encode(" " + text.strip())`).
    """
    global _warned
    if not text:
        return 0

    tok = _tokenizer(model_name)
    if tok is None:
        if not _warned:
            _warned = True
            log.warning(
                "No tokenizer for '%s'; estimating budgets at %.1f chars/token",
                model_name, _CHARS_PER_TOKEN,
            )
        return int(len(text) / _CHARS_PER_TOKEN) + 1

    try:
        return len(tok.encode(" " + text.strip()).ids)
    except Exception:  # noqa: BLE001
        return int(len(text) / _CHARS_PER_TOKEN) + 1
