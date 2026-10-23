#!/usr/bin/env python3
"""Finalize latency: hotkey release → final text, on your own retained audio.

Replays retained dictations (~/.local/state/voiceio/recordings/*.wav) through
the real `StreamingSession` at real-time pace: 100 ms chunks go through the
real `AudioRecorder` callback (Silero VAD, speech-pause events, the same
buffer), so interim, freeze and stop scheduling happen exactly as they do
live. At the end of the clip the "hotkey is released": the recorder stops and
`session.stop()` is timed until it returns the final text.

Accuracy is scored against the teacher labels (`voiceio learn` labels.jsonl)
with `voiceio.learn.metrics` WER, on the raw (pre-pipeline) final text.

Arms compare versions of `voiceio/streaming.py` side by side in one process,
sharing one decoder (same model processes, same lanes), interleaved per clip
with the order rotated, so every arm sees the same machine load:

    python scripts/bench/finalize_latency.py run \\
        --arm base=HEAD~1 --arm new=WORKTREE --out /tmp/bench.jsonl
    python scripts/bench/finalize_latency.py report /tmp/bench.jsonl

An arm is LABEL=REF[,freeze=SECS][,model=NAME][,_CONST=VALUE...]: REF is a
git ref, a .py file, or WORKTREE for the file on disk; _CONST overrides a
module constant of that streaming.py (e.g. _PAUSE_FREEZE_MIN_SECS=3); NAME
is a `[model] name` (default: the configured one), so decoders can be
compared under the same load:

    python scripts/bench/finalize_latency.py run --out /tmp/d.jsonl \\
        --arm whisper=WORKTREE --arm parakeet=WORKTREE,model=parakeet-tdt-0.6b-v2

Everything else (recorder, VAD) comes from this checkout and is shared by
all arms, so put the base arm on the oldest ref whose streaming.py still
runs against it.

Clips are held-out (dev/test split) labeled recordings, stratified by length.
Transcripts stay in the --out file, never in the repo; nothing leaves the
machine. Decoding is not deterministic (the temperature fallback samples), so
judge WER deltas against a second copy of the base arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Everything but the arms' streaming.py comes from this checkout, not from
# whatever voiceio the interpreter has installed.
sys.path.insert(0, str(REPO))
SR = 16000
CHUNK = 1600  # 100 ms, what capture delivers

BUCKETS = [(0, 15, "<15s"), (15, 30, "15-30s"), (30, 60, "30-60s"),
           (60, 120, "60-120s"), (120, 1e9, "120s+")]
# Clips per bucket in the default selection (24 in total).
PLAN = {"<15s": 5, "15-30s": 5, "30-60s": 5, "60-120s": 5, "120s+": 4}


def bucket_of(secs: float) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= secs < hi)


# ── arms ────────────────────────────────────────────────────────────────

def load_streaming(label: str, ref: str) -> types.ModuleType:
    """voiceio/streaming.py as of `ref`, imported under a private name."""
    if ref == "WORKTREE":
        src = (REPO / "voiceio" / "streaming.py").read_text()
    elif ref.endswith(".py"):
        src = Path(ref).read_text()
    else:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{ref}:voiceio/streaming.py"],
            capture_output=True, text=True, check=True).stdout
    name = f"voiceio._bench_streaming_{label}"
    mod = types.ModuleType(name)
    mod.__file__ = f"{ref}:voiceio/streaming.py"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def parse_arm(spec: str) -> dict:
    label, _, rest = spec.partition("=")
    ref, *opts = rest.split(",")
    arm = {"label": label, "ref": ref or "WORKTREE", "freeze": None, "consts": {},
           "model": ""}
    for opt in opts:
        k, _, v = opt.partition("=")
        if k == "freeze":
            arm["freeze"] = float(v)
        elif k == "model":
            arm["model"] = v   # compare decoders: each model loads once, shared by its arms
        elif k.startswith("_") and k == k.upper():
            arm["consts"][k] = float(v)   # a module constant, e.g. _FREEZE_SEARCH_SECS
        else:
            raise SystemExit(f"unknown arm option {opt!r}")
    return arm


# ── clips ───────────────────────────────────────────────────────────────

def wav_secs(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def select_clips(n_scale: float, min_secs: float, max_secs: float) -> list[dict]:
    from voiceio.config import RECORDINGS_DIR as rec
    from voiceio.learn.store import load_labels, split_of

    pool: dict[str, list[dict]] = {}
    for audio, label in load_labels().items():
        path = rec / audio
        if split_of(audio) == "train" or not path.exists():
            continue
        secs = wav_secs(path)
        if not min_secs <= secs <= max_secs:
            continue
        ref = " ".join(s.text.strip() for s in label.segments).strip()
        if not ref:
            continue
        pool.setdefault(bucket_of(secs), []).append(
            {"clip": audio, "secs": round(secs, 2), "ref": ref, "path": str(path)})
    out = []
    for name, k in PLAN.items():
        clips = sorted(pool.get(name, []),
                       key=lambda c: hashlib.sha256(c["clip"].encode()).hexdigest())
        out.extend(clips[:max(1, round(k * n_scale))])
    return sorted(out, key=lambda c: c["secs"])


def load_wav(path: str) -> np.ndarray:
    from voiceio.learn.teacher import load_wav as lw
    return lw(Path(path))


# ── replay ──────────────────────────────────────────────────────────────

class _NoStream:
    """Stands in for the sounddevice stream: the recorder never opens a mic."""

    def stop(self) -> None: ...
    def close(self) -> None: ...


def drain(decoder) -> None:
    """Wait until no decode is running on any lane (an abandoned interim must
    not steal CPU from the next run)."""
    for lane in (getattr(decoder, "draft", None), getattr(decoder, "final", decoder)):
        if hasattr(lane, "wait_idle"):   # re-entrant decoders (sherpa-onnx)
            lane.wait_idle()
            continue
        lock = getattr(lane, "_lock", None)
        if lock is not None:
            with lock:
                pass


def replay(mod, engine, bias, audio: np.ndarray, audio_cfg, vad, freeze: float) -> dict:
    from voiceio.recorder import AudioRecorder
    from voiceio.typers.collecting import CollectingTyper

    rec = AudioRecorder(audio_cfg, vad=vad)
    rec._stream = _NoStream()
    rec.start()
    session = mod.StreamingSession(
        engine.decoder, CollectingTyper(), rec,
        pipeline=engine.pipeline, freeze_secs=freeze, bias=bias)
    session.start()
    t_next = time.monotonic()
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i:i + CHUNK].reshape(-1, 1)
        rec._callback(chunk, len(chunk), None, None)
        t_next += len(chunk) / SR
        delay = t_next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    late = max(0.0, time.monotonic() - t_next)  # feeder fell behind real time
    t0 = time.monotonic()
    final_audio = rec.stop()
    text = session.stop(final_audio)
    wait = time.monotonic() - t0
    if session._worker_thread is not None:
        session._worker_thread.join(timeout=300)
    drain(engine.decoder)
    return {
        "wait": round(wait, 3),
        "late": round(late, 3),
        "text": text,
        "raw": session.raw_final_text or "",
        "latency": session.final_latency,
        "passes": [{k: v for k, v in p.items() if k != "text"} for p in session.trace],
    }


def conditions() -> dict:
    out = {"loadavg": round(os.getloadavg()[0], 1)}
    try:
        out["profile"] = Path("/sys/firmware/acpi/platform_profile").read_text().strip()
    except OSError:
        pass
    return out


def cmd_run(args) -> None:
    from voiceio.config import load
    from voiceio.core.engine import Engine
    from voiceio.learn import metrics
    from voiceio.vad import load_vad

    cfg = load()
    if args.beam:
        cfg.model.beam_size = args.beam
    arms = [parse_arm(a) for a in args.arm]
    mods = {a["label"]: load_streaming(a["label"], a["ref"]) for a in arms}
    for a in arms:
        for name, value in a["consts"].items():
            if not hasattr(mods[a["label"]], name):
                raise SystemExit(f"{a['ref']}:voiceio/streaming.py has no {name}")
            setattr(mods[a["label"]], name, value)
    clips = select_clips(args.scale, args.min_secs, args.max_secs)
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(clips)} clips, {sum(c['secs'] for c in clips) / 60:.1f} min of audio; "
          f"arms: {', '.join(a['label'] + '=' + a['ref'] for a in arms)}", flush=True)
    import copy
    engines, biases = {}, {}
    for model in dict.fromkeys(a["model"] for a in arms):
        mcfg = copy.deepcopy(cfg)
        if model:
            mcfg.model.name = model
        engines[model] = Engine.from_config(mcfg, prefix=False)
        # One bias for every run: the live prompt builder learns from each
        # commit, which would hand the arms different prompts.
        biases[model] = engines[model].bias()
        warm = np.zeros(SR, dtype=np.float32)
        engines[model].decoder.decode(warm, final=True, bias=biases[model])
        engines[model].decoder.decode(warm, final=False, bias=biases[model])
    vad = load_vad(cfg.audio)

    out = open(args.out, "a", encoding="utf-8")
    try:
        for rnd in range(args.rounds):
            for i, clip in enumerate(clips):
                audio = load_wav(clip["path"])
                k = (i + rnd) % len(arms)
                for arm in arms[k:] + arms[:k]:
                    freeze = arm["freeze"] if arm["freeze"] is not None \
                        else cfg.output.streaming_freeze_secs
                    cond = conditions()
                    r = replay(mods[arm["label"]], engines[arm["model"]], biases[arm["model"]],
                               audio, cfg.audio, vad, freeze)
                    al = metrics.align(clip["ref"], r["raw"])
                    row = {"round": rnd, "clip": clip["clip"], "secs": clip["secs"],
                           "bucket": bucket_of(clip["secs"]), "arm": arm["label"],
                           "ref_spec": arm["ref"], "freeze": freeze, "consts": arm["consts"],
                           "model": engines[arm["model"]].decoder.name,
                           "errors": al.errors, "ref_words": al.ref_words,
                           **r, **cond}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"[{rnd}:{i + 1}/{len(clips)}] {clip['secs']:6.1f}s {arm['label']:>8} "
                          f"wait {r['wait']:5.2f}s  WER {al.wer:6.1%}  load {cond['loadavg']}",
                          flush=True)
    finally:
        out.close()
        for engine in engines.values():
            engine.close()
    report([args.out])


# ── report ──────────────────────────────────────────────────────────────

def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def report(paths: list[str]) -> None:
    rows = [json.loads(line) for p in paths for line in open(p, encoding="utf-8")]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    groups = ["all"] + [b for _, _, b in BUCKETS if any(r["bucket"] == b for r in rows)]
    print(f"\n{len(rows)} runs; arms: {', '.join(arms)}; "
          f"load avg median {statistics.median(r['loadavg'] for r in rows):.1f}; "
          f"profile {sorted({r.get('profile', '?') for r in rows})}")
    print(f"{'bucket':>8} {'arm':>8} {'n':>4} {'p50':>6} {'p90':>6} {'mean':>6} {'WER':>7}")
    for g in groups:
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and (g == "all" or r["bucket"] == g)]
            if not sel:
                continue
            waits = [r["wait"] for r in sel]
            wer = sum(r["errors"] for r in sel) / max(1, sum(r["ref_words"] for r in sel))
            print(f"{g:>8} {arm:>8} {len(sel):>4} {_pct(waits, .5):6.2f} {_pct(waits, .9):6.2f} "
                  f"{statistics.mean(waits):6.2f} {wer:7.2%}")
    if len(arms) < 2:
        return
    base = arms[0]
    by = {}
    for r in rows:
        by.setdefault((r["round"], r["clip"]), {})[r["arm"]] = r
    for arm in arms[1:]:
        pairs = [(v[base], v[arm]) for v in by.values() if base in v and arm in v]
        d_wait = [b["wait"] - a["wait"] for a, b in pairs]
        d_err = [b["errors"] - a["errors"] for a, b in pairs]
        faster = sum(d < 0 for d in d_wait)
        print(f"\n{arm} vs {base}: {len(pairs)} paired runs; wait diff median "
              f"{statistics.median(d_wait):+.2f}s, faster in {faster}/{len(pairs)}; "
              f"errors {sum(d_err):+d} total "
              f"(worse in {sum(d > 0 for d in d_err)}, better in {sum(d < 0 for d in d_err)})")
        for a, b in pairs:
            if b["errors"] > a["errors"]:
                print(f"  worse: {a['clip']} ({a['secs']}s) errors {a['errors']} -> "
                      f"{b['errors']} of {a['ref_words']} words")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay clips through each arm")
    r.add_argument("--arm", action="append", required=True,
                   help="LABEL=REF[,freeze=SECS][,_CONST=VALUE]; REF: git ref or WORKTREE")
    r.add_argument("--out", required=True, help="JSONL results (appended)")
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--scale", type=float, default=1.0, help="clips per bucket multiplier")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--min-secs", type=float, default=5.0)
    r.add_argument("--max-secs", type=float, default=180.0)
    r.add_argument("--beam", type=int, default=0, help="override [model] beam_size")
    s = sub.add_parser("report", help="summarize result files")
    s.add_argument("files", nargs="+")
    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        report(args.files)


if __name__ == "__main__":
    main()
