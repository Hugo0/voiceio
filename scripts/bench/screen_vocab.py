#!/usr/bin/env python3
"""Benchmark on-screen vocabulary (voiceio/context_terms.py).

    python scripts/bench/screen_vocab.py overhead [--runs 300]
    python scripts/bench/screen_vocab.py accuracy [--clips 60] [--out DIR]

overhead: capture (window title + PRIMARY selection, the real sources on this
session) and extraction timings, plus the cost of merging terms into the bias.
Prints sizes and timings only, never the captured text.

accuracy: decodes retained clips (~/.local/state/voiceio/recordings) with the
configured model and the user's real static hotwords, with and without the
screen terms, scored against teacher labels
(~/.local/share/voiceio/learn/labels.jsonl). Contexts:

  title    the window title stored with the utterance (history "context",
           only present while [data] capture_context was on)
  prev     REALISTIC PROXY: the text of the user's own dictations committed in
           the 10 minutes before the clip — what sat on screen above the
           cursor in the chat/terminal being dictated into
  oracle   SYNTHETIC: a "selection" holding the clip's own rare reference
           terms inside another clip's unrelated text (the term is on screen)
  noise    SYNTHETIC: a "selection" of another clip's unrelated text only
           (nothing relevant on screen: measures harm)
  decoy    SYNTHETIC worst case: a "selection" of only other clips' rare
           terms, none spoken here (measures false insertions)

Arms are decoded back to back per clip, in alternating order, so a background
load (e.g. `voiceio learn maintain`) hits both alike. No initial_prompt in
either arm. Writes per-clip results (with transcripts) to --out, which
defaults to a private temp dir: nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

HOME = Path.home()
RECORDINGS = HOME / ".local/state/voiceio/recordings"
HISTORY = HOME / ".local/state/voiceio/history.jsonl"
LABELS = HOME / ".local/share/voiceio/learn/labels.jsonl"
PREV_WINDOW_SECS = 600
PREV_MAX_CHARS = 3000

# Keep the bench's worker.log etc. out of the user's state dir. Config,
# vocabulary and corrections are still read (read-only) from ~/.config.
_STATE = Path(tempfile.mkdtemp(prefix="voiceio-bench-state-"))
os.environ["XDG_STATE_HOME"] = str(_STATE)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# ── overhead ──────────────────────────────────────────────────────────────


def overhead(args) -> None:
    from voiceio import config, context_terms
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders.base import BiasMode, Capabilities
    from voiceio.corrections import CorrectionDict
    from voiceio.vocabulary import VocabularyLoader

    cfg = config.load()
    t0 = time.perf_counter()
    context_terms.warm(cfg.model.language)
    print(f"wordfreq warm-up (once per daemon): {(time.perf_counter() - t0) * 1e3:.0f} ms")

    caps = Capabilities(bias=BiasMode.HOTWORDS | BiasMode.PROMPT, max_bias_tokens=120)
    corrections = CorrectionDict()
    corrections.load()
    bp = BiasProvider(VocabularyLoader(cfg.model), corrections, language=cfg.model.language)
    bp.bias_for(caps)   # warm vocabulary ranking + tokenizer, as a daemon is

    cap, ext, merge, total, n_terms, sizes, timeouts = [], [], [], [], [], [], 0
    for _ in range(args.runs):
        t0 = time.perf_counter()
        screen = context_terms.capture()
        t1 = time.perf_counter()
        terms = bp.screen_terms([screen.title, screen.selection], caps)
        t2 = time.perf_counter()
        bp.bias_for(caps, terms)
        t3 = time.perf_counter()
        cap.append((t1 - t0) * 1e3)
        ext.append((t2 - t1) * 1e3)
        merge.append((t3 - t2) * 1e3)
        total.append((t3 - t0) * 1e3)
        n_terms.append(len(terms))
        sizes.append((len(screen.title), len(screen.selection)))
        timeouts += screen.secs >= context_terms.CAPTURE_TIMEOUT_SECS
    print(f"runs {args.runs}; title chars {sizes[-1][0]}, selection chars {sizes[-1][1]}, "
          f"terms {statistics.median(n_terms):.0f}; capture hit the deadline {timeouts}x")
    for name, xs in (("capture", cap), ("extract", ext), ("bias merge", merge),
                     ("total", total)):
        print(f"  {name:11s} p50 {pct(xs, .5):6.1f} ms  p95 {pct(xs, .95):6.1f} ms  "
              f"max {max(xs):6.1f} ms")

    # Extraction alone on a full-size selection (MAX_SCAN_CHARS of prose).
    text = (" ".join(_labels().values()))[:context_terms.MAX_SCAN_CHARS]
    xs = []
    for _ in range(50):
        t0 = time.perf_counter()
        bp.screen_terms(["", text], caps)
        xs.append((time.perf_counter() - t0) * 1e3)
    print(f"  extract on {len(text)} chars: p50 {pct(xs, .5):.1f} ms  p95 {pct(xs, .95):.1f} ms")


# ── accuracy ──────────────────────────────────────────────────────────────


def _labels() -> dict[str, str]:
    out = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["audio"]] = " ".join(s["text"].strip() for s in d.get("segments", [])).strip()
    return out


def _history() -> list[dict]:
    rows = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return sorted((r for r in rows if r.get("ts") and r.get("text")), key=lambda r: r["ts"])


def _plausible(ref: str) -> bool:
    """Skip garbage teacher labels (wrong-language hallucinations)."""
    ascii_share = sum(c.isascii() for c in ref) / max(1, len(ref))
    return len(ref.split()) >= 4 and ascii_share > 0.97


def accuracy(args) -> None:
    import numpy as np  # noqa: F401 — the decoder needs it importable

    from voiceio import config
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders import build_decoder
    from voiceio.corrections import CorrectionDict
    from voiceio.learn import metrics
    from voiceio.learn.teacher import load_wav
    from voiceio.vocabulary import VocabularyLoader

    cfg = config.load()
    rng = random.Random(args.seed)
    labels = _labels()
    history = _history()
    by_audio = {r["audio"]: r for r in history if r.get("audio")}

    decoder = build_decoder(cfg.model, lanes=1)
    caps = decoder.capabilities
    corrections = CorrectionDict()
    corrections.load()
    bp = BiasProvider(VocabularyLoader(cfg.model), corrections, language=cfg.model.language)
    base_bias = bp.bias_for(caps)
    print(f"model {cfg.model.name}  beam {cfg.model.beam_size}  "
          f"static hotwords {len(base_bias.hotwords.split(', ')) if base_bias.hotwords else 0} terms")

    def short(a: str) -> bool:
        return (RECORDINGS / a).stat().st_size <= args.max_secs * 32000 + 44

    pool = [a for a, ref in labels.items()
            if (RECORDINGS / a).exists() and a in by_audio and _plausible(ref) and short(a)]
    pool.sort()

    def prev_text(audio: str) -> str:
        ts = by_audio[audio]["ts"]
        prior = [r["text"] for r in history if ts - PREV_WINDOW_SECS <= r["ts"] < ts - 0.5]
        return "\n".join(prior)[-PREV_MAX_CHARS:]

    def other(audio: str) -> str:
        while True:
            o = rng.choice(pool)
            if o != audio:
                return labels[o]

    contexts: dict[str, dict[str, list[str]]] = {
        "title": {}, "prev": {}, "oracle": {}, "noise": {}, "decoy": {}}
    for a in pool:
        title = by_audio[a].get("context")
        if title:
            contexts["title"][a] = [title, ""]
    with_prev = [a for a in pool if bp.screen_terms(["", prev_text(a)], caps)]
    for a in rng.sample(with_prev, min(args.clips, len(with_prev))):
        contexts["prev"][a] = ["", prev_text(a)]
    # Oracle: clips whose reference holds a term the static hotwords lack.
    with_terms = [a for a in pool if bp.screen_terms(["", labels[a]], caps)]
    for a in rng.sample(with_terms, min(args.clips, len(with_terms))):
        own = bp.screen_terms(["", labels[a]], caps)
        filler = other(a).split()
        for t in own:
            filler.insert(rng.randrange(len(filler) + 1), t)
        contexts["oracle"][a] = ["", " ".join(filler)]
    for a in rng.sample(pool, min(args.clips, len(pool))):
        contexts["noise"][a] = ["", other(a)]
    # Decoy: the worst case for false insertions — a selection made only of
    # OTHER clips' rare terms, none of which this clip says.
    for a in rng.sample(pool, min(args.clips, len(pool))):
        decoys: list[str] = []
        while len(decoys) < 6:
            o = rng.choice(with_terms)
            decoys += [t for t in bp.screen_terms(["", labels[o]], caps)
                       if o != a and t.lower() not in labels[a].lower()]
        contexts["decoy"][a] = ["", ", ".join(decoys)]
    if args.only:
        contexts = {k: v for k, v in contexts.items() if k in args.only.split(",")}

    out_dir = Path(args.out or tempfile.mkdtemp(prefix="voiceio-bench-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "screen_vocab.jsonl"
    summary = {}
    flip = False
    with open(rows_path, "w", encoding="utf-8") as fh:
        for cname, clips in contexts.items():
            agg = dict(clips=0, ref_words=0, err_base=0, err_screen=0, spoken=0,
                       hit_base=0, hit_screen=0, fi_base=0, fi_screen=0,
                       secs_base=[], secs_screen=[], audio_secs=0.0, terms=0,
                       better=0, worse=0)
            for i, (a, texts) in enumerate(sorted(clips.items())):
                ref = labels[a]
                terms = bp.screen_terms(texts, caps)
                screen_bias = bp.bias_for(caps, terms)
                audio = load_wav(RECORDINGS / a)

                def run(bias):
                    t0 = time.monotonic()
                    try:
                        text = decoder.decode(audio, final=True, bias=bias).text
                    except Exception as e:  # noqa: BLE001
                        print(f"  decode failed on {a}: {e}", file=sys.stderr)
                        text = None
                    return text, time.monotonic() - t0

                # Interleave: alternate which arm goes first. The base arm is
                # re-decoded per condition so latency pairs are adjacent.
                flip = not flip
                if flip:
                    hb, sb = run(base_bias)
                    hs, ss = run(screen_bias) if terms else (hb, sb)
                else:
                    hs, ss = run(screen_bias) if terms else (None, 0.0)
                    hb, sb = run(base_bias)
                    if not terms:
                        hs, ss = hb, sb
                if hb is None or hs is None:
                    continue
                ab, as_ = metrics.align(ref, hb), metrics.align(ref, hs)
                rw = set(metrics.words(ref))
                in_ref = [t for t in terms if metrics._count_term(metrics.words(ref), metrics.words(t))]
                not_ref = [t for t in terms if t not in in_ref and not set(metrics.words(t)) <= rw]
                spoken, hit_b = metrics.term_hits(ref, hb, in_ref)
                _, hit_s = metrics.term_hits(ref, hs, in_ref)
                fi_b = sum(metrics._count_term(metrics.words(hb), metrics.words(t)) for t in not_ref)
                fi_s = sum(metrics._count_term(metrics.words(hs), metrics.words(t)) for t in not_ref)
                agg["clips"] += 1
                agg["ref_words"] += ab.ref_words
                agg["err_base"] += ab.errors
                agg["err_screen"] += as_.errors
                agg["better"] += as_.errors < ab.errors
                agg["worse"] += as_.errors > ab.errors
                agg["spoken"] += spoken
                agg["hit_base"] += hit_b
                agg["hit_screen"] += hit_s
                agg["fi_base"] += fi_b
                agg["fi_screen"] += fi_s
                agg["terms"] += len(terms)
                agg["audio_secs"] += len(audio) / 16000
                if terms:
                    agg["secs_base"].append(sb)
                    agg["secs_screen"].append(ss)
                fh.write(json.dumps({
                    "context": cname, "audio": a, "terms": terms, "in_ref": in_ref,
                    "ref": ref, "base": hb, "screen": hs,
                    "err": [ab.errors, as_.errors], "ref_words": ab.ref_words,
                    "secs": [round(sb, 3), round(ss, 3)],
                }, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {cname} {i + 1}/{len(clips)}  terms {len(terms)}  "
                      f"err {ab.errors}->{as_.errors}  hits {hit_b}->{hit_s}/{spoken}  "
                      f"fi {fi_b}->{fi_s}", flush=True)
            summary[cname] = agg
    decoder.close()

    print("\n=== on-screen vocabulary: base (static hotwords) vs + screen terms ===")
    for cname, g in summary.items():
        if not g["clips"]:
            print(f"{cname}: no clips")
            continue
        wb = g["err_base"] / g["ref_words"]
        ws = g["err_screen"] / g["ref_words"]
        lat = ""
        if g["secs_base"]:
            ratios = [s / b for b, s in zip(g["secs_base"], g["secs_screen"]) if b > 0]
            lat = (f"  decode {sum(g['secs_base']):.0f}s -> {sum(g['secs_screen']):.0f}s "
                   f"(median paired ratio {statistics.median(ratios):.3f})")
        print(f"{cname:6s} clips {g['clips']:3d} ({g['audio_secs'] / 60:.0f} min) "
              f"terms/clip {g['terms'] / g['clips']:.1f}  "
              f"WER {wb:.2%} -> {ws:.2%}  (clips better {g['better']}, worse {g['worse']})  "
              f"term recall {g['hit_base']}/{g['spoken']} -> {g['hit_screen']}/{g['spoken']}  "
              f"false insertions {g['fi_base']} -> {g['fi_screen']}{lat}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"per-clip rows: {rows_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overhead")
    o.add_argument("--runs", type=int, default=300)
    a = sub.add_parser("accuracy")
    a.add_argument("--clips", type=int, default=60, help="per synthetic/proxy context")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--only", default="", help="comma-separated contexts to run")
    a.add_argument("--max-secs", type=float, default=30.0, help="skip longer clips")
    a.add_argument("--out", default="")
    args = ap.parse_args()
    {"overhead": overhead, "accuracy": accuracy}[args.cmd](args)


if __name__ == "__main__":
    main()
