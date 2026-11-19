"""`voiceio learn report`: a numbers-only summary of the last 7 days.

Made to be mailed off the machine (a weekly timer sends it to whoever looks
after the install), so it holds counts, rates and dates only: never a
transcript, a word, the vocabulary, a correction, a file path or audio. Log
lines are matched, never copied; the model is named by its catalog name or
fine-tune run, never by its path. A number that can't be computed is None,
with a short reason under "unavailable".
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

from voiceio import config

WINDOW_DAYS = 7
SCHEMA = 1

# Log messages (matched, never copied): the typer falling back to pasting
# from the clipboard, and errors from decoding (streaming.py, app.py,
# watchdog.py, the faster-whisper worker).
_CLIPBOARD_FALLBACK = re.compile(r"(?:Switched to|fallback) typer: clipboard\s*$")
_DECODER_ERROR = re.compile(r"decod|transcri|worker|Processing failed", re.I)
_EVAL_FILE = re.compile(r"^eval-(\d{4})(\d\d)(\d\d)-\d{6}\.json$")


def model_label(name: str) -> str:
    """A catalog name, 'fine-tune <run>' or 'local <basename>'; never a path."""
    from voiceio.core.decoders.catalog import info
    from voiceio.learn.store import learn_dir
    from voiceio.report import scrub
    m = info(name)
    if m is not None:
        return "whisper.cpp server" if m.name.startswith("whispercpp:") else m.name
    raw = name.removeprefix("sherpa:")
    path = Path(raw).expanduser()
    if raw.startswith(("/", "~", ".")) or ("/" in raw and path.exists()):
        try:
            label = "fine-tune " + path.resolve().relative_to(
                (learn_dir() / "models").resolve()).parts[0]
        except (ValueError, IndexError, OSError):
            label = "local " + path.name
    else:
        label = raw   # a Whisper size or a Hugging Face repo id
    return scrub(label)


def _dictations(cutoff: float, out: dict, unavailable: dict) -> None:
    from voiceio import history
    out["dictations_7d"] = out["audio_minutes_7d"] = None
    if not history.HISTORY_PATH.exists():
        unavailable["dictations_7d"] = unavailable["audio_minutes_7d"] = \
            "no history file (history off, or nothing dictated yet)"
        return
    rows = [r for r in history.read() if isinstance(r.get("ts"), (int, float))
            and r["ts"] >= cutoff]
    out["dictations_7d"] = len(rows)
    secs = [r["duration"] for r in rows if isinstance(r.get("duration"), (int, float))]
    if secs or not rows:
        out["audio_minutes_7d"] = round(sum(secs) / 60, 1)
    else:
        unavailable["audio_minutes_7d"] = "history rows have no duration"


def _eval(model: str, beam: int, out: dict, unavailable: dict) -> None:
    from voiceio.learn import evaluate
    from voiceio.learn.store import learn_dir
    keys = ("wer", "vocab_recall", "eval_dataset", "eval_clips", "eval_date")
    for key in keys:
        out[key] = None
    root = learn_dir() / "datasets"
    datasets = sorted((p.name for p in root.glob("v[0-9]*") if p.is_dir()),
                      reverse=True) if root.is_dir() else []
    if not datasets:
        unavailable.update(dict.fromkeys(keys, "no dataset yet (voiceio learn build)"))
        return
    for ds in datasets:   # the newest dataset the active model was scored on
        if found := evaluate.latest_run(model, ds, beam=beam):
            path, run = found
            break
    else:
        unavailable.update(dict.fromkeys(
            keys, "no stored eval of the active model (voiceio learn eval)"))
        return
    out["wer"] = round(float(run["wer"]), 4)
    out["vocab_recall"] = round(float(run["term_recall"]), 4)
    out["eval_dataset"] = ds
    out["eval_clips"] = int(run.get("clips", 0))
    if m := _EVAL_FILE.match(path.name):
        out["eval_date"] = "-".join(m.groups())
    else:
        unavailable["eval_date"] = "eval file has no date"


def _log_paths() -> list[Path]:
    base = config.LOG_DIR / "voiceio.log"
    return [base, *(base.with_name(f"{base.name}.{i}") for i in (1, 2))]


def _log_counts(now: dt.datetime, out: dict, unavailable: dict,
                paths: list[Path] | None = None) -> None:
    from voiceio.errorlog import _LOG_LINE, _read, _time
    cutoff = now - dt.timedelta(days=WINDOW_DAYS)
    out["clipboard_fallbacks_7d"] = out["decoder_errors_7d"] = out["log_days_covered"] = None
    lines = [ln for p in (paths or _log_paths()) for ln in _read(p)]
    if not lines:
        reason = "no service log"
        for key in ("clipboard_fallbacks_7d", "decoder_errors_7d", "log_days_covered"):
            unavailable[key] = reason
        return
    fallbacks = errors = 0
    oldest = None
    for line in lines:
        m = _LOG_LINE.match(line)
        when = _time(m.group(1)) if m else None
        if when is None:
            continue
        oldest = when if oldest is None else min(oldest, when)
        if when < cutoff:
            continue
        message = line[m.end():]
        if _CLIPBOARD_FALLBACK.search(message):
            fallbacks += 1
        elif m.group(3) in ("ERROR", "CRITICAL") and _DECODER_ERROR.search(message):
            errors += 1
    out["clipboard_fallbacks_7d"] = fallbacks
    out["decoder_errors_7d"] = errors
    if oldest is not None:   # the log rotates by size: it may not reach back 7 days
        out["log_days_covered"] = round(min(WINDOW_DAYS, (now - oldest).total_seconds() / 86400), 1)


def _learning(cutoff: float, out: dict, unavailable: dict) -> None:
    from voiceio import service
    from voiceio.learn import lastrun, store
    labels = store.load_labels()
    out["clips_labelled_7d"] = sum(1 for lab in labels.values() if lab.ts >= cutoff)
    out["learn_scheduled"] = out["learn_ran_7d"] = out["learn_last_run"] = None
    out["learn_last_result"] = None
    systemd = service.has_systemd()
    if systemd:
        out["learn_scheduled"] = service.learn_timer_enabled()
    else:
        unavailable["learn_scheduled"] = "no systemd"
    # The run's own record first: systemd can't tell a unit that never ran
    # (Result=success) from one that did, and its timestamps vary by version.
    rec = lastrun.read()
    if rec is not None:
        when, result = float(rec["started"]), rec["outcome"]
        if result == "running" and not (systemd and service.learning_now()):
            result = "unfinished"   # killed before it could record an outcome
    elif systemd:
        when, result = service.learn_last_run()
    else:
        for key in ("learn_ran_7d", "learn_last_run", "learn_last_result"):
            unavailable[key] = "no systemd and no recorded learning run"
        return
    out["learn_ran_7d"] = when is not None and when >= cutoff
    if when is None:
        unavailable["learn_last_run"] = unavailable["learn_last_result"] = \
            "no recorded learning run (the learning timer has never fired)"
        return
    out["learn_last_run"] = dt.date.fromtimestamp(when).isoformat()
    if result:
        out["learn_last_result"] = result
    else:
        unavailable["learn_last_result"] = "unknown"


def build(now: float | None = None, cfg=None, *, log_paths: list[Path] | None = None) -> dict:
    """The summary: numbers, dates and labels only."""
    from voiceio import __version__
    from voiceio.core.decoders.catalog import family
    now = time.time() if now is None else now
    cfg = cfg or config.load()
    cutoff = now - WINDOW_DAYS * 86400
    out: dict = {
        "schema": SCHEMA,
        "date": dt.date.fromtimestamp(now).isoformat(),
        "window_days": WINDOW_DAYS,
        "version": __version__,
        "model_family": family(cfg.model.name),
        "model_name": model_label(cfg.model.name),
    }
    unavailable: dict[str, str] = {}
    _dictations(cutoff, out, unavailable)
    _eval(cfg.model.name, cfg.model.beam_size, out, unavailable)
    _log_counts(dt.datetime.fromtimestamp(now), out, unavailable, log_paths)
    _learning(cutoff, out, unavailable)
    out["unavailable"] = unavailable
    return out


_LABELS = (
    ("version", "voiceio"),
    ("model_name", "model"),
    ("model_family", "family"),
    ("dictations_7d", "dictations"),
    ("audio_minutes_7d", "audio minutes"),
    ("wer", "WER"),
    ("vocab_recall", "vocab recall"),
    ("eval_dataset", "eval dataset"),
    ("eval_clips", "eval clips"),
    ("eval_date", "eval date"),
    ("clipboard_fallbacks_7d", "clipboard fallbacks"),
    ("decoder_errors_7d", "decoder errors"),
    ("log_days_covered", "log covers days"),
    ("learn_scheduled", "learning scheduled"),
    ("learn_ran_7d", "learning ran"),
    ("learn_last_run", "last learning run"),
    ("learn_last_result", "last result"),
    ("clips_labelled_7d", "clips labelled"),
)


def format_text(summary: dict) -> str:
    """The human-readable form of `build()`."""
    lines = [f"voiceio, last {summary['window_days']} days to {summary['date']}"]
    for key, label in _LABELS:
        value = summary.get(key)
        if value is None:
            shown = "n/a — " + summary["unavailable"].get(key, "unavailable")
        elif key in ("wer", "vocab_recall"):
            shown = f"{value:.1%}"
        elif isinstance(value, bool):
            shown = "yes" if value else "no"
        else:
            shown = str(value)
        lines.append(f"  {label:<20} {shown}")
    return "\n".join(lines)


def to_json(summary: dict) -> str:
    return json.dumps(summary, indent=2)
