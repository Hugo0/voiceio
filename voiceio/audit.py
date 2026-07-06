"""Teacher-model audit: the measurement half of the self-correcting loop.

Every learned correction is probationary. This weekly offline job re-transcribes
recent retained audio with a stronger TEACHER model (distil-large-v3) and uses
it as ground truth to:

  * measure drift of the live model (WER proxy, raw vs teacher);
  * strike correction rules the teacher shows were wrong (the "misheard" word
    was actually real) and confirm the ones it validates;
  * auto-retire struck rules (>=2 strikes, strikes > confirmations), re-blocking
    them so mining can't silently re-learn them;
  * expire rules that haven't fired in 60 days (dictionary hygiene);
  * confirm / age out vocabulary terms against what the teacher actually heard;
  * roll back the whole run if drift regressed sharply and a snapshot exists.

Nothing here uses the network; the teacher runs locally, latency is irrelevant.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from voiceio import config

log = logging.getLogger(__name__)

_DAY = 86400.0
INACTIVITY_SECS = 60 * _DAY   # retire rules idle this long
VOCAB_AGE_SECS = 90 * _DAY    # age unseen vocab terms out of the hotword budget
# corrections_audit records within this window of an entry's finalize time are
# attributed to that utterance (the rule fires just before history append).
FIRE_WINDOW_SECS = 30.0
DRIFT_REGRESSION = 1.5        # >50% relative worsening vs median triggers rollback


@dataclass
class AuditState:
    """Retired rules, persisted so mining never silently re-learns them."""

    retired: set[str] = field(default_factory=set)

    def is_retired(self, wrong: str) -> bool:
        return bool(wrong) and wrong.lower() in self.retired

    def retire(self, wrong: str) -> None:
        if wrong:
            self.retired.add(wrong.lower())


def load_audit_state(path: Path | None = None) -> AuditState:
    p = path or config.AUDIT_STATE_PATH
    if not p.exists():
        return AuditState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AuditState()
    retired = raw.get("retired", []) if isinstance(raw, dict) else []
    if not isinstance(retired, list):
        retired = []
    return AuditState(retired={str(r).lower() for r in retired if isinstance(r, str)})


def save_audit_state(state: AuditState, path: Path | None = None) -> None:
    p = path or config.AUDIT_STATE_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"retired": sorted(state.retired)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Failed to write audit state: %s", e)


def is_retired(wrong: str, path: Path | None = None) -> bool:
    """Whether `wrong` was retired by a prior audit (mining consults this)."""
    return load_audit_state(path).is_retired(wrong)


@dataclass
class AuditReport:
    entries_audited: int = 0
    audio_secs: float = 0.0
    drift_wer: float = 0.0
    rules_confirmed: list[str] = field(default_factory=list)
    rules_retired: list[str] = field(default_factory=list)
    rules_expired: list[str] = field(default_factory=list)
    vocab_confirmed: list[str] = field(default_factory=list)
    vocab_aged: list[str] = field(default_factory=list)
    rolled_back: bool = False
    snapshot_dir: Path | None = None


# ── helpers ──────────────────────────────────────────────────────────────

def _wer(ref: str, hyp: str) -> float:
    """Word error rate proxy via difflib (edits / ref length)."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0
    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    edits = max(len(ref_words), len(hyp_words)) - matches
    return edits / len(ref_words)


def _contains_word(text: str, word: str) -> bool:
    if not word:
        return False
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
        return frames / rate
    except (OSError, wave.Error):
        return 0.0


def _read_fire_log(path: Path | None = None) -> list[dict]:
    p = path or config.CORRECTIONS_AUDIT_PATH
    if not p.exists():
        return []
    records = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records


def _read_metrics(path: Path | None = None) -> list[dict]:
    p = path or config.METRICS_PATH
    if not p.exists():
        return []
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _default_teacher():
    """Lazily build the offline teacher transcriber (distil-large-v3)."""
    from faster_whisper import WhisperModel

    model = WhisperModel("distil-large-v3", device="cpu", compute_type="int8")

    def _transcribe(wav_path: Path) -> str:
        segments, _ = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text for seg in segments).strip()

    return _transcribe


# ── main entry point ─────────────────────────────────────────────────────

def run_audit(
    cfg,
    *,
    max_audio_secs: float = 600,
    transcribe=None,
    snapshot_dir: Path | None = None,
) -> AuditReport:
    """Audit learned rules and vocabulary against a teacher transcription.

    `transcribe(wav_path) -> str` is injectable (tests pass a fake; production
    lazily loads distil-large-v3). `snapshot_dir` is the pre-mining snapshot
    from this run, restored if drift regressed sharply.
    """
    from voiceio import history
    from voiceio.corrections import CorrectionDict

    report = AuditReport(snapshot_dir=snapshot_dir)
    now = time.time()

    if transcribe is None:
        transcribe = _default_teacher()

    # ── Sample recent entries with retained audio, up to the time budget ──
    recordings_dir = config.RECORDINGS_DIR
    sampled: list[tuple[dict, Path, float]] = []
    total_secs = 0.0
    for entry in history.read(limit=0):  # newest first
        name = entry.get("audio")
        if not name:
            continue
        wav = recordings_dir / name
        if not wav.exists():
            continue
        secs = entry.get("duration") or _wav_duration(wav)
        if total_secs + secs > max_audio_secs and sampled:
            break
        sampled.append((entry, wav, secs))
        total_secs += secs
        if total_secs >= max_audio_secs:
            break

    report.entries_audited = len(sampled)
    report.audio_secs = round(total_secs, 2)

    cd = CorrectionDict()
    state = load_audit_state()
    fire_log = _read_fire_log()

    # Per-rule strike/confirm tally keyed by lowercased "wrong".
    strikes: dict[str, int] = {}
    confirms: dict[str, int] = {}
    wers: list[float] = []
    teacher_texts: list[str] = []

    for entry, wav, secs in sampled:
        try:
            teacher = transcribe(wav)
        except Exception:  # noqa: BLE001 — one bad clip must not kill the audit
            log.debug("teacher transcription failed for %s", wav.name, exc_info=True)
            continue
        teacher_texts.append(teacher)
        raw = entry.get("raw") or entry.get("text") or ""
        wers.append(_wer(raw, teacher))

        # Rules that fired around this utterance's finalize time.
        entry_ts = entry.get("ts", 0.0)
        lo = entry_ts - secs - FIRE_WINDOW_SECS
        hi = entry_ts + FIRE_WINDOW_SECS
        for rec in fire_log:
            rts = rec.get("ts", 0.0)
            if not (lo <= rts <= hi):
                continue
            wrong = rec.get("wrong", "")
            right = rec.get("right", "")
            key = wrong.lower()
            if _contains_word(teacher, wrong):
                # Teacher heard the "wrong" word — the rule rewrote a real word.
                strikes[key] = strikes.get(key, 0) + 1
            elif _contains_word(teacher, right):
                confirms[key] = confirms.get(key, 0) + 1

    report.drift_wer = round(statistics.fmean(wers), 4) if wers else 0.0

    # ── Auto-retire struck rules ──────────────────────────────────────────
    current = {w.lower(): (w, r) for w, r in cd.list_all().items()}
    for key, s in strikes.items():
        if s >= 2 and s > confirms.get(key, 0) and key in current:
            orig = current[key][0]
            cd.remove(orig)
            state.retire(orig)
            report.rules_retired.append(orig)

    # Confirmations for rules that survived retirement.
    retired_keys = {r.lower() for r in report.rules_retired}
    for key, c in confirms.items():
        if c > 0 and key not in retired_keys and key in current:
            report.rules_confirmed.append(current[key][0])

    # ── Inactivity expiry ─────────────────────────────────────────────────
    # A rule that fired at some point but not within 60 days is stale. A rule
    # that never fired is left alone (could be brand new — can't judge idleness).
    last_fire: dict[str, float] = {}
    for rec in fire_log:
        w = rec.get("wrong", "").lower()
        ts = rec.get("ts", 0.0)
        if w and ts > last_fire.get(w, 0.0):
            last_fire[w] = ts
    for key, (orig, _r) in list(current.items()):
        if key in retired_keys:
            continue
        lf = last_fire.get(key)
        if lf is not None and lf < now - INACTIVITY_SECS:
            if cd.remove(orig):
                report.rules_expired.append(orig)

    # ── Vocabulary verification / aging ───────────────────────────────────
    _audit_vocabulary(cfg, teacher_texts, now, report)

    # ── Metrics line ──────────────────────────────────────────────────────
    prior_metrics = _read_metrics()
    _append_metrics(report, now)

    # ── Drift alarm + rollback ────────────────────────────────────────────
    recent = [m.get("drift_wer", 0.0) for m in prior_metrics[-4:]]
    median = statistics.median(recent) if recent else 0.0
    regressed = median > 0 and report.drift_wer > median * DRIFT_REGRESSION
    if regressed and snapshot_dir is not None and Path(snapshot_dir).exists():
        from voiceio.snapshots import rollback
        if rollback(snapshot_dir):
            report.rolled_back = True

    save_audit_state(state)
    _notify(report)
    return report


def _audit_vocabulary(cfg, teacher_texts, now, report) -> None:
    """Confirm vocab terms the teacher heard; age out terms unseen for 90 days.

    Aging moves a term to the bottom of vocabulary.txt rather than deleting it:
    the 800-char loader truncates the tail, so stale terms fall out of the
    hotword budget but can rejoin if used again.
    """
    from voiceio import history
    from voiceio.vocabulary import _read_terms, resolve_vocab_path

    path = resolve_vocab_path(cfg.model)
    if not path.exists():
        return
    terms = _read_terms(path)
    if not terms:
        return

    teacher_blob = "\n".join(teacher_texts)
    # Last time each term appeared in a live transcript.
    entries = history.read(limit=0)
    last_seen: dict[str, float] = {}
    for e in entries:
        text = e.get("text", "")
        ts = e.get("ts", 0.0)
        low = text.lower()
        for term in terms:
            if term.lower() in low and ts > last_seen.get(term, 0.0):
                last_seen[term] = ts

    fresh: list[str] = []
    aged: list[str] = []
    for term in terms:
        seen_by_teacher = _contains_word(teacher_blob, term)
        if seen_by_teacher:
            report.vocab_confirmed.append(term)
        seen_ts = now if seen_by_teacher else last_seen.get(term, 0.0)
        if seen_ts and seen_ts >= now - VOCAB_AGE_SECS:
            fresh.append(term)
        elif seen_ts == 0.0:
            fresh.append(term)  # never seen anywhere — no aging signal, keep in place
        else:
            aged.append(term)

    if not aged:
        return
    report.vocab_aged = aged
    try:
        path.write_text("\n".join(fresh + aged) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("Failed to rewrite vocabulary during aging: %s", e)


def _append_metrics(report: AuditReport, ts: float, path: Path | None = None) -> None:
    p = path or config.METRICS_PATH
    line = {
        "ts": ts,
        "entries_audited": report.entries_audited,
        "audio_secs": report.audio_secs,
        "drift_wer": report.drift_wer,
        "rules_confirmed": len(report.rules_confirmed),
        "rules_retired": len(report.rules_retired),
        "rules_expired": len(report.rules_expired),
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("Failed to write audit metrics: %s", e)


def _notify(report: AuditReport) -> None:
    from voiceio.feedback import notify

    if report.rolled_back:
        notify(
            "voiceio self-corrected: rolled back",
            "Drift regressed after mining — restored the pre-mining snapshot.",
        )
    elif report.rules_retired:
        notify(
            "voiceio self-corrected",
            f"retired {len(report.rules_retired)} bad rule(s)",
        )
