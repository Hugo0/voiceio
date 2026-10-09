"""`voiceio models` — what can transcribe for you, and which is best on your voice."""
from __future__ import annotations

import argparse

from voiceio.core.decoders.base import BiasMode


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("models", help="List, compare and choose speech models",
                       description=__doc__)
    msub = p.add_subparsers(dest="models_cmd", metavar="ACTION")
    p.set_defaults(func=cmd_list)
    msub.add_parser("list", help="Known models, fine-tunes and the active one").set_defaults(
        func=cmd_list)
    b = msub.add_parser("bench", help="Score models on your held-out clips (needs `learn build`)")
    b.add_argument("models", nargs="*", help="names/paths (default: active + fine-tunes)")
    b.add_argument("--limit", type=int, default=None, help="max clips")
    b.set_defaults(func=cmd_bench)
    u = msub.add_parser("use", help="Dictate with this model from now on")
    u.add_argument("name")
    u.set_defaults(func=cmd_use)


def _bias_label(bias: BiasMode) -> str:
    if bias & BiasMode.UNBOUNDED:
        return "unlimited terms"
    parts = []
    if bias & BiasMode.HOTWORDS:
        parts.append("~35 terms")
    if bias & BiasMode.PROMPT:
        parts.append("prompt")
    return " + ".join(parts) or "none"


def _fine_tunes() -> list[str]:
    from voiceio.learn.store import learn_dir
    root = learn_dir() / "models"
    return sorted(str(p / "ct2") for p in root.glob("*") if (p / "ct2" / "model.bin").exists()) \
        if root.exists() else []


def cmd_list(args: argparse.Namespace) -> None:
    from voiceio import config
    from voiceio.core.decoders.catalog import CATALOG
    active = config.load().model.name
    print(f"{'':2}{'model':44} {'runtime':15} {'biasing':18} license")
    for m in CATALOG:
        mark = "* " if m.name == active else "  "
        tag = " (experimental)" if m.experimental else ""
        print(f"{mark}{m.name:44} {m.runtime:15} {_bias_label(m.bias):18} {m.license}{tag}")
        if m.notes:
            print(f"{'':4}{m.notes}")
    tunes = _fine_tunes()
    if tunes:
        print("\nyour fine-tunes (voiceio learn):")
        for t in tunes:
            print(f"{'* ' if t == active else '  '}{t}")
    if all(m.name != active for m in CATALOG) and active not in tunes:
        print(f"\n* active: {active}")
    print("\nCompare on your own voice: voiceio models bench MODEL…")


def cmd_bench(args: argparse.Namespace) -> None:
    from voiceio import config
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders import build_decoder
    from voiceio.learn import evaluate, store
    from voiceio.vocabulary import VocabularyLoader

    ds = store.latest_dataset()
    if ds is None:
        raise SystemExit("needs labeled clips: voiceio learn label && voiceio learn build")
    cfg = config.load()
    names = args.models or [cfg.model.name, *_fine_tunes()]
    terms = VocabularyLoader(cfg.model).get_all()
    results = []
    for name in dict.fromkeys(names):
        cfg.model.name = name
        print(f"\n{name}: loading…", flush=True)
        try:
            dec = build_decoder(cfg.model, lanes=1)
        except Exception as e:  # noqa: BLE001 — one broken model must not end the bench
            print(f"  unavailable: {e}")
            continue
        try:
            bias = BiasProvider(VocabularyLoader(cfg.model)).bias_for(dec.capabilities)
            res = evaluate.evaluate(dec, ds, bias=bias, terms=terms, limit=args.limit)
        finally:
            dec.close()
        evaluate.save(res)
        print(f"  {res.summary()}")
        results.append((name, res))
    if len(results) > 1:
        best = max(results, key=lambda r: (r[1].term_recall, -r[1].wer))
        print(f"\nbest vocabulary recall: {best[0]} — `voiceio models use {best[0]}`")


def cmd_use(args: argparse.Namespace) -> None:
    from voiceio.config import set_value
    from voiceio.core.decoders.catalog import info
    m = info(args.name)
    if m is not None and m.experimental:
        print(f"note: {args.name} is experimental — {m.notes}")
    set_value("model", "name", args.name)
    print(f"model set to {args.name} — restart the daemon: voiceio service restart")
