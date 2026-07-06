"""Auto-correction: frequency analysis + Levenshtein clustering + LLM review."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from wordfreq import top_n_list

from voiceio.wordfreq import extract_words, is_common, is_known

log = logging.getLogger(__name__)


# ── Safety gate for persisting mined correction rules ────────────────────

# A word is "real" in a protected language when it's reasonably common there.
# Higher than the English non-word bar (2.0) because sharing a rare cognate
# shouldn't block fixing an obvious English misrecognition.
_PROTECT_ZIPF = 3.0


def gate_correction(
    wrong: str, right: str,
    *,
    vocabulary: set[str] | None = None,
    language: str = "en",
    protect_languages: tuple[str, ...] | list[str] = (),
) -> str | None:
    """Guard against learning bad correction rules from mined pairs.

    A mined `wrong → right` pair is only safe to persist when:
      * `wrong` is a genuine non-word (zipf < 2.0, i.e. not is_known) — we
        never rewrite real words the user might legitimately dictate, and
      * `wrong` isn't a real word in any protected language (a bilingual
        user's Spanish "harina" must not be rewritten because it looks like
        a non-word in English), and
      * `right` is either a known-common word (is_common) or already a term
        in the user's vocabulary file — so we don't cement one misspelling
        into another (the historic "manteka"/"wordall" bug).

    Returns ``None`` when the pair passes, otherwise a short human-readable
    reason string explaining why it was rejected.
    """
    from wordfreq import zipf_frequency

    wrong = (wrong or "").strip()
    right = (right or "").strip()
    if not wrong or not right:
        return "empty term"
    vocab_lower = {v.lower() for v in (vocabulary or set())}
    if is_known(wrong, language):
        return f'"{wrong}" is a real word — refusing to auto-correct it'
    for pl in protect_languages:
        if pl != language and zipf_frequency(wrong.lower(), pl) >= _PROTECT_ZIPF:
            return f'"{wrong}" is a real word in protected language "{pl}"'
    if not (is_common(right, language) or right.lower() in vocab_lower):
        return f'"{right}" is neither a common word nor in your vocabulary'
    return None


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
    dismissed: set[str] | None = None,
    min_word_length: int = 4,
) -> list[SuspiciousWord]:
    """Scan history entries for words that are likely Whisper mistakes.

    History v2 entries may carry a ``raw`` field (the pre-correction Whisper
    text). We prefer it when present, since post-corrected ``text`` hides the
    very misrecognitions we're hunting for. Entries with only ``text`` (v1)
    still work.
    """
    skip = existing_corrections or set()
    vocab = vocabulary or set()
    dismiss = dismissed or set()
    skip_lower = {w.lower() for w in skip}
    vocab_lower = {w.lower() for w in vocab}
    dismiss_lower = {w.lower() for w in dismiss}

    word_counts: Counter[str] = Counter()
    word_contexts: dict[str, list[str]] = {}

    for entry in entries:
        text = (entry.get("raw") or entry.get("text") or "").strip()
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
        if word in skip_lower or word in vocab_lower or word in dismiss_lower:
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

CRITICAL: Every input word MUST appear in exactly one of the three buckets below. \
Do not omit any word. If you have no opinion, put it in `ask_user` with an empty \
`right` string and a short reason like "unclear".

Buckets:
1. auto_fix — Clearly a Whisper transcription error. You are confident in the correction.
2. ask_user — Might be an error, or you can't decide. Provide your best guess in `right` \
   (or "" if no guess) and a short `reason`.
3. vocabulary — A real proper noun, brand name, or technical term. Not an error.

Return ONLY a JSON object with three arrays:
{"auto_fix": [{"wrong": "olamma", "right": "Ollama"}], "ask_user": [{"wrong": "pinat", "right": "Peanut", "reason": "Could be brand name"}, {"wrong": "tridle", "right": "", "reason": "unclear"}], "vocabulary": ["grafana", "postgres"]}"""


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


# Each entry takes ~30-50 output tokens. Smaller batches mean shorter
# responses (faster) and less wall-clock impact when one batch is slow.
_REVIEW_BATCH_SIZE = 25


def review_suspicious(
    cfg, suspicious: list[SuspiciousWord],
    *, on_progress=None,
) -> ReviewResult:
    """Send suspicious words to an LLM for 3-bucket classification.

    Uses the OpenAI-compatible API (OpenRouter/Anthropic/OpenAI) if an API key
    is available, otherwise falls back to local Ollama. Cloud requests are
    batched so the JSON response can't be truncated by max_tokens.

    `on_progress(done, total)` is called after each batch finishes.
    """
    if not suspicious:
        return ReviewResult()

    from voiceio.llm_api import resolve_api_key
    api_key = resolve_api_key(cfg.autocorrect)
    if api_key:
        result = _review_cloud_batched(cfg, suspicious, api_key, on_progress)
        if result.auto_fix or result.ask_user or result.vocabulary:
            return result
        log.warning("Cloud API returned nothing usable, falling back to Ollama")

    # Ollama is single-shot — local models are too weak for this task to benefit from batching.
    if cfg.llm.enabled:
        try:
            from voiceio.llm import LLMProcessor
            proc = LLMProcessor(cfg.llm)
            response = proc.generate(
                _build_review_prompt(suspicious),
                system=_REVIEW_SYSTEM_PROMPT,
                timeout=cfg.llm.timeout_secs * 3,
            )
            if response:
                return _parse_review_response(response)
        except Exception as e:
            log.warning("Ollama review failed: %s", e)

    return ReviewResult()


_REVIEW_MAX_WORKERS = 4  # cap on concurrent API calls — balances speed vs rate limits
_REVIEW_OVERALL_TIMEOUT_PER_BATCH = 20.0  # seconds — used to compute overall deadline


def _review_cloud_batched(
    cfg, suspicious: list[SuspiciousWord], api_key: str, on_progress,
) -> ReviewResult:
    """Send `suspicious` to the cloud LLM in fixed-size batches, in parallel.

    Enforces a wall-clock deadline so a single hung request can't stall the
    entire review. Implemented with daemon threads + Queue so abandoned
    stragglers don't block this function from returning (a stuck urllib
    request inside ThreadPoolExecutor would otherwise hold up its `__exit__`).
    """
    import queue
    import threading
    import time

    from voiceio.llm_api import chat

    total = len(suspicious)
    batches = [
        (start, suspicious[start:start + _REVIEW_BATCH_SIZE])
        for start in range(0, total, _REVIEW_BATCH_SIZE)
    ]
    if not batches:
        return ReviewResult()

    workers = min(_REVIEW_MAX_WORKERS, len(batches))
    rounds = max(1, (len(batches) + workers - 1) // workers)
    # ×2 slack so a slow-but-eventually-finishing first round doesn't kill us.
    overall_timeout = rounds * _REVIEW_OVERALL_TIMEOUT_PER_BATCH * 2

    results_q: queue.Queue = queue.Queue()
    sem = threading.Semaphore(workers)

    def runner(start_batch):
        start, batch = start_batch
        with sem:
            try:
                response = chat(
                    cfg.autocorrect, _REVIEW_SYSTEM_PROMPT,
                    _build_review_prompt(batch),
                    api_key=api_key,
                )
                if not response:
                    log.warning(
                        "Cloud API empty response for batch %d-%d (size %d)",
                        start, start + len(batch), len(batch),
                    )
                    results_q.put((batch, ReviewResult()))
                    return
                results_q.put((batch, _parse_review_response(response)))
            except Exception as e:
                log.warning("Batch %d raised: %s", start, e)
                results_q.put((batch, ReviewResult()))

    for b in batches:
        threading.Thread(target=runner, args=(b,), daemon=True).start()

    merged = ReviewResult()
    done_count = 0
    deadline = time.monotonic() + overall_timeout
    received = 0
    while received < len(batches):
        remaining = deadline - time.monotonic()
        try:
            batch, r = results_q.get(timeout=max(remaining, 0))
        except queue.Empty:
            log.warning(
                "Review hit overall deadline (%.0fs) — %d batch(es) abandoned",
                overall_timeout, len(batches) - received,
            )
            break
        merged.auto_fix.extend(r.auto_fix)
        merged.ask_user.extend(r.ask_user)
        merged.vocabulary.extend(r.vocabulary)
        classified = (
            {f["wrong"].lower() for f in r.auto_fix}
            | {f["wrong"].lower() for f in r.ask_user}
            | {v.lower() for v in r.vocabulary}
        )
        omitted = [sw.word for sw in batch if sw.word.lower() not in classified]
        if omitted and (r.auto_fix or r.ask_user or r.vocabulary):
            log.info(
                "Batch returned %d/%d classifications; LLM omitted: %s",
                len(batch) - len(omitted), len(batch), omitted[:10],
            )
        done_count += len(batch)
        received += 1
        if on_progress:
            try:
                on_progress(min(done_count, total), total)
            except Exception:
                pass
    return merged


def rank_review_score(
    item: dict, sw: SuspiciousWord | None,
) -> float:
    """Score how likely a `to_review` item is a real Whisper error.

    Higher = more likely a misheard real word that needs correcting.
    Lower (incl. negative) = more likely a tech term / acronym / proper noun.
    """
    score = 0.0
    # An LLM-suggested correction is the strongest signal.
    if item.get("right"):
        score += 100.0
    if sw:
        # Words near a common dictionary word are likely Whisper mishearings.
        if sw.similar_common:
            score += 30.0 + min(len(sw.similar_common), 5) * 5.0
        # Higher count = more impactful when fixed (capped to avoid swamping).
        score += min(sw.count, 20)
        # Short all-lowercase ASCII with no similar word = probably an acronym.
        if (not sw.similar_common
                and len(sw.word) <= 5
                and sw.word.islower()
                and sw.word.isascii()):
            score -= 50.0
    return score


def rank_review_items(
    items: list[dict], sw_by_word: dict[str, SuspiciousWord],
) -> list[dict]:
    """Sort review items so most-likely-error words come first.

    Stable: ties preserve LLM-provided order.
    """
    return sorted(
        items,
        key=lambda it: -rank_review_score(it, sw_by_word.get(it.get("wrong", ""))),
    )


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
            ask_user=_validate_ask_user(parsed.get("ask_user", [])),
            vocabulary=[v for v in parsed.get("vocabulary", []) if isinstance(v, str)],
        )

    # Fallback: try old-style flat array (for Ollama compatibility)
    if isinstance(parsed, list):
        fixes = _validate_ask_user(parsed)
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
    """Validate auto_fix entries: must have a different `right` to apply."""
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


def _validate_ask_user(items) -> list[dict]:
    """Validate ask_user entries: keep anything with a `wrong` word.

    Unlike auto_fix, ask_user entries are useful even without a clean
    correction — the LLM's `reason` still tells the user what to look at.
    Items where `right == wrong` are normalized to empty `right`.
    """
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        wrong = item.get("wrong", "").strip()
        if not wrong:
            continue
        right = item.get("right", "").strip()
        if right.lower() == wrong.lower():
            right = ""  # no useful suggestion
        valid.append({
            "wrong": wrong,
            "right": right,
            "reason": item.get("reason", ""),
        })
    return valid


# ── Evidence-based multi-vote adjudication ───────────────────────────────
#
# The corrections dictionary is a *cache* of the runtime post-correction layer
# (voiceio/postcorrect.py): a MISSING rule costs almost nothing (postcorrect
# fixes it live), a WRONG rule causes regressions. So instead of queueing
# ambiguous candidates for a human, we adjudicate each one by asking the LLM
# `votes` times independently, with the word's FULL sentence contexts, and act
# only on unanimity:
#   * apply a correction only if ALL votes name the same target AND it passes
#     gate_correction,
#   * add to vocabulary only if ALL votes call it a real term,
#   * otherwise DROP (the caller defers it for later re-adjudication).


@dataclass
class AdjudicationResult:
    """Outcome of multi-vote adjudication over ambiguous candidates."""
    apply: list[dict] = field(default_factory=list)       # [{"wrong","right"}]
    vocabulary: list[str] = field(default_factory=list)   # real terms to keep
    deferred: list[dict] = field(default_factory=list)    # original item + "votes"


_ADJUDICATE_SYSTEM_PROMPT = """\
You judge whether an uncommon word from speech-to-text (Whisper) dictation is a
misrecognition or a real term the user meant to say. You are given each word
with up to three FULL sentences it appeared in — use that context carefully.

For every input word return exactly one verdict:
  - "correction": it is clearly a misrecognition. Put the intended word in
    `right` (the correctly spelled real word the user actually said).
  - "keep": it is a real proper noun, brand, or technical term to keep as-is.
  - "uncertain": you cannot tell from the context.

Be conservative: only say "correction" when the context makes the intended
word obvious, and only say "keep" when it reads as a deliberate real term. When
in doubt, say "uncertain".

Return ONLY a JSON object:
{"verdicts": [{"word": "olamma", "verdict": "correction", "right": "Ollama"}, \
{"word": "grafana", "verdict": "keep", "right": ""}, \
{"word": "tridle", "verdict": "uncertain", "right": ""}]}"""


def _build_adjudicate_prompt(items: list[dict], sw_by_word: dict) -> str:
    """List each candidate with up to 3 full sentence contexts."""
    parts = []
    for it in items:
        word = it.get("wrong", "")
        sw = sw_by_word.get(word)
        contexts = list(sw.contexts[:3]) if (sw and sw.contexts) else []
        if contexts:
            ctx = "\n".join(f'    - "{c}"' for c in contexts)
        else:
            ctx = "    (no context available)"
        parts.append(f'- "{word}"\n{ctx}')
    return "\n\n".join(parts)


def _parse_adjudication_response(response: str) -> list[dict]:
    """Parse the adjudication JSON into a flat list of verdict dicts."""
    text = response.strip()
    if "```" in text:
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    parsed = _try_parse_json(text)
    verdicts = None
    if isinstance(parsed, dict):
        verdicts = parsed.get("verdicts")
    elif isinstance(parsed, list):
        verdicts = parsed
    if not isinstance(verdicts, list):
        return []
    out = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        word = (v.get("word") or "").strip()
        if not word:
            continue
        verdict = (v.get("verdict") or "").strip().lower()
        out.append({
            "word": word,
            "verdict": verdict,
            "right": (v.get("right") or "").strip(),
        })
    return out


def _adjudicate_pass(cfg, items: list[dict], sw_by_word: dict, api_key: str) -> dict:
    """One independent vote over all `items`, batched. Returns word_lc -> verdict."""
    from voiceio.llm_api import chat

    out: dict[str, dict] = {}
    for start in range(0, len(items), _REVIEW_BATCH_SIZE):
        batch = items[start:start + _REVIEW_BATCH_SIZE]
        try:
            response = chat(
                cfg.autocorrect, _ADJUDICATE_SYSTEM_PROMPT,
                _build_adjudicate_prompt(batch, sw_by_word),
                api_key=api_key,
            )
        except Exception as e:
            log.warning("Adjudication batch %d raised: %s", start, e)
            continue
        if not response:
            continue
        for v in _parse_adjudication_response(response):
            out[v["word"].lower()] = v
    return out


def _decide_adjudication(
    word: str, votes: list[dict], required: int,
    vocabulary: set[str], language: str, protect_languages,
) -> tuple[str, str]:
    """Apply the unanimity rule to a word's collected votes.

    Returns ("apply", right) | ("vocab", "") | ("defer", "").
    """
    # Every independent pass must have returned a verdict for this word.
    if len(votes) < required:
        return ("defer", "")

    verdicts = [v.get("verdict") for v in votes]

    if all(v == "correction" for v in verdicts):
        rights = [(v.get("right") or "").strip() for v in votes]
        lowered = {r.lower() for r in rights if r}
        if len(lowered) == 1 and rights[0]:
            right = next(r for r in rights if r)
            gate = gate_correction(
                word, right, vocabulary=vocabulary, language=language,
                protect_languages=protect_languages,
            )
            if gate is None:
                return ("apply", right)
        return ("defer", "")

    if all(v == "keep" for v in verdicts):
        return ("vocab", "")

    return ("defer", "")


def adjudicate(
    cfg, items: list[dict], sw_by_word: dict, *,
    votes: int = 3,
    vocabulary: set[str] | None = None,
    language: str = "en",
) -> AdjudicationResult:
    """Evidence-based adjudication of ambiguous candidates via repeated voting.

    Each of `items` (dicts with at least a "wrong" key) is voted on `votes`
    times independently, using the word's full sentence contexts. A correction
    is applied only on unanimous agreement + gate_correction; vocabulary only
    on unanimous "keep"; everything else is deferred (returned in `deferred`
    carrying its accumulated `votes` for the caller to persist).

    Votes are batched per pass, so N items × `votes` costs about `votes` API
    calls (times ceil(N / batch_size)) rather than N × `votes`.
    """
    result = AdjudicationResult()
    if not items:
        return result

    vocab = vocabulary or set()
    protect = tuple(cfg.autocorrect.protect_languages)

    from voiceio.llm_api import resolve_api_key
    api_key = resolve_api_key(cfg.autocorrect)
    if not api_key:
        # No way to gather evidence — defer everything untouched.
        for it in items:
            result.deferred.append({**it, "votes": []})
        return result

    votes_by_word: dict[str, list[dict]] = {}
    for _ in range(max(1, votes)):
        pass_votes = _adjudicate_pass(cfg, items, sw_by_word, api_key)
        for wl, v in pass_votes.items():
            votes_by_word.setdefault(wl, []).append(v)

    for it in items:
        word = it.get("wrong", "")
        vlist = votes_by_word.get(word.lower(), [])
        decision, right = _decide_adjudication(
            word, vlist, max(1, votes), vocab, language, protect,
        )
        if decision == "apply":
            result.apply.append({"wrong": word, "right": right})
        elif decision == "vocab":
            result.vocabulary.append(word)
        else:
            result.deferred.append({**it, "votes": vlist})
    return result
