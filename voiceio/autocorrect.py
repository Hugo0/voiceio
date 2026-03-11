"""Auto-correction: frequency analysis + Levenshtein clustering + LLM review."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from wordfreq import top_n_list

from voiceio.wordfreq import extract_words, is_known

log = logging.getLogger(__name__)


# ── Suspicious word detection ────────────────────────────────────────────

@dataclass
class SuspiciousWord:
    """A word that might be a Whisper transcription error."""
    word: str
    count: int                       # how often it appears in history
    contexts: list[str] = field(default_factory=list)  # example sentences
    similar_common: list[str] = field(default_factory=list)  # nearby common words
    reason: str = ""                 # why it's suspicious


@dataclass
class ReviewResult:
    """3-bucket classification from LLM review."""
    auto_fix: list[dict] = field(default_factory=list)   # [{"wrong": ..., "right": ...}]
    ask_user: list[dict] = field(default_factory=list)    # [{"wrong": ..., "right": ..., "reason": ...}]
    vocabulary: list[str] = field(default_factory=list)   # proper nouns / tech terms


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def find_suspicious_words(
    entries: list[dict],
    language: str = "en",
    *,
    existing_corrections: set[str] | None = None,
    vocabulary: set[str] | None = None,
    min_word_length: int = 4,
) -> list[SuspiciousWord]:
    """Scan history entries for words that are likely Whisper mistakes."""
    skip = existing_corrections or set()
    vocab = vocabulary or set()
    skip_lower = {w.lower() for w in skip}
    vocab_lower = {w.lower() for w in vocab}

    word_counts: Counter[str] = Counter()
    word_contexts: dict[str, list[str]] = {}

    for entry in entries:
        text = entry.get("text", "").strip()
        if not text:
            continue
        words = extract_words(text)
        for w in words:
            wl = w.lower()
            word_counts[wl] += 1
            if wl not in word_contexts:
                word_contexts[wl] = []
            if len(word_contexts[wl]) < 3 and text not in word_contexts[wl]:
                word_contexts[wl].append(text)

    common_list = top_n_list(language, 20000)
    common_set = {w: i for i, w in enumerate(common_list)}

    candidates: list[SuspiciousWord] = []

    for word, count in word_counts.items():
        if len(word) < min_word_length:
            continue
        if word in skip_lower or word in vocab_lower:
            continue
        if is_known(word, language):
            continue
        similar = _find_similar_common(word, common_set, max_distance=2)
        reason = "not in common word list"
        if similar:
            reason += f", similar to: {', '.join(similar[:3])}"

        candidates.append(SuspiciousWord(
            word=word,
            count=count,
            contexts=word_contexts.get(word, []),
            similar_common=similar,
            reason=reason,
        ))

    candidates.sort(key=lambda s: (-len(s.similar_common), -s.count))
    return candidates


def _find_similar_common(
    word: str, common_words: dict[str, int], max_distance: int = 2,
) -> list[str]:
    """Find common words within Levenshtein distance of the target word."""
    results = []
    word_len = len(word)
    for common, freq_rank in common_words.items():
        if abs(len(common) - word_len) > max_distance:
            continue
        if _levenshtein(word, common) <= max_distance:
            results.append(common)
    results.sort(key=lambda w: common_words.get(w, 99999))
    return results[:5]


# ── Levenshtein clustering ───────────────────────────────────────────────

def cluster_variants(suspicious: list[SuspiciousWord]) -> list[list[SuspiciousWord]]:
    """Group suspicious words that are Levenshtein-close to each other."""
    if not suspicious:
        return []
    used = set()
    clusters: list[list[SuspiciousWord]] = []
    words = list(suspicious)

    for i, sw in enumerate(words):
        if i in used:
            continue
        cluster = [sw]
        used.add(i)
        for j in range(i + 1, len(words)):
            if j in used:
                continue
            if _levenshtein(sw.word, words[j].word) <= 2:
                cluster.append(words[j])
                used.add(j)
        clusters.append(cluster)

    clusters.sort(key=lambda c: (-len(c), -sum(s.count for s in c)))
    return clusters


# ── LLM-assisted review ─────────────────────────────────────────────────

_REVIEW_SYSTEM_PROMPT = """\
You are an expert at identifying speech-to-text (Whisper) transcription errors.

I'll give you suspicious words from dictation history with context and similar common words.
Classify each word into exactly one of three buckets:

1. auto_fix — Clearly a Whisper transcription error. You are confident in the correction.
2. ask_user — Might be an error, but you're not sure. Give your best guess and explain why.
3. vocabulary — A real proper noun, brand name, or technical term. Not an error.

Return ONLY a JSON object with three arrays:
{"auto_fix": [{"wrong": "olamma", "right": "Ollama"}], "ask_user": [{"wrong": "pinat", "right": "Peanut", "reason": "Could be brand name"}], "vocabulary": ["grafana", "postgres"]}"""


def _build_review_prompt(suspicious: list[SuspiciousWord]) -> str:
    """Build the user message listing all suspicious words with context."""
    parts = []
    for sw in suspicious:
        contexts = "; ".join(f'"{c[:100]}"' for c in sw.contexts[:2])
        similar = ", ".join(sw.similar_common[:3]) if sw.similar_common else "none"
        parts.append(
            f'- "{sw.word}" (appears {sw.count}x)\n'
            f"  Context: {contexts}\n"
            f"  Similar common words: {similar}"
        )
    return "\n\n".join(parts)


def review_suspicious(cfg, suspicious: list[SuspiciousWord]) -> ReviewResult:
    """Send suspicious words to an LLM for 3-bucket classification.

    Uses the OpenAI-compatible API (OpenRouter/Anthropic/OpenAI) if an API key
    is available, otherwise falls back to local Ollama.
    """
    if not suspicious:
        return ReviewResult()

    prompt = _build_review_prompt(suspicious)

    # Try cloud API first
    from voiceio.llm_api import resolve_api_key
    api_key = resolve_api_key(cfg.autocorrect)
    if api_key:
        from voiceio.llm_api import chat
        response = chat(
            cfg.autocorrect, _REVIEW_SYSTEM_PROMPT, prompt, api_key=api_key,
        )
        if response:
            return _parse_review_response(response)
        log.warning("Cloud API call failed, falling back to Ollama")

    # Fall back to local Ollama
    if cfg.llm.enabled:
        try:
            from voiceio.llm import LLMProcessor
            proc = LLMProcessor(cfg.llm)
            response = proc.generate(
                prompt, system=_REVIEW_SYSTEM_PROMPT,
                timeout=cfg.llm.timeout_secs * 3,
            )
            if response:
                return _parse_review_response(response)
        except Exception as e:
            log.warning("Ollama review failed: %s", e)

    return ReviewResult()


def _parse_review_response(response: str) -> ReviewResult:
    """Parse LLM JSON response into ReviewResult, handling formatting quirks."""
    text = response.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as JSON object with 3 buckets
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        return ReviewResult(
            auto_fix=_validate_fixes(parsed.get("auto_fix", [])),
            ask_user=_validate_fixes(parsed.get("ask_user", [])),
            vocabulary=[v for v in parsed.get("vocabulary", []) if isinstance(v, str)],
        )

    # Fallback: try old-style flat array (for Ollama compatibility)
    if isinstance(parsed, list):
        fixes = _validate_fixes(parsed)
        return ReviewResult(ask_user=fixes)

    return ReviewResult()


def _try_parse_json(text: str):
    """Try to extract and parse JSON from text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object or array from surrounding text
    for pattern in (r'\{.*\}', r'\[.*\]'):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


def _validate_fixes(items) -> list[dict]:
    """Validate a list of fix entries, filtering out garbage."""
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        wrong = item.get("wrong", "").strip()
        right = item.get("right", "").strip()
        if wrong and right and wrong.lower() != right.lower():
            valid.append({
                "wrong": wrong,
                "right": right,
                "reason": item.get("reason", ""),
            })
    return valid
