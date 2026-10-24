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
    u = msub.add_parser("use", help="Dictate with this model from now on (downloads it)")
    u.add_argument("name", help="a model name, path, or `auto` for the default")
    u.set_defaults(func=cmd_use)
    d = msub.add_parser("download", help="Fetch a sherpa-onnx model (e.g. parakeet-tdt-0.6b-v2)")
    d.add_argument("name")
    d.set_defaults(func=cmd_download)
    s = msub.add_parser("sherpa", help="Install sherpa-onnx with the Parakeet fixes "
                                       "(until upstream releases them)")
    s.add_argument("--check", action="store_true", help="only say what is installed")
    s.set_defaults(func=cmd_sherpa)


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


def _sherpa_note() -> str:
    """Why Parakeet is or isn't usable with the installed sherpa-onnx."""
    from voiceio.core.decoders.sherpa import features
    f = features()
    if not f.installed:
        return "needs sherpa-onnx: voiceio models sherpa"
    if not f.length_reward:
        return ("this sherpa-onnx lacks the NeMo TDT beam-search fix (no hotwords): "
                "voiceio models sherpa")
    if not f.pretokenized_hotwords:
        return "hotwords via sherpa's own tokenization (fewer names caught): voiceio models sherpa"
    return ""


def cmd_list(args: argparse.Namespace) -> None:
    from voiceio import config
    from voiceio.core.decoders.catalog import CATALOG, default_model
    cfg = config.load()
    active = cfg.model.name
    default = default_model(cfg.model.language)
    print(f"{'':2}{'model':44} {'runtime':15} {'biasing':18} license")
    for m in CATALOG:
        mark = "* " if m.name == active else "  "
        tag = " (experimental)" if m.experimental else ""
        tag += " (default)" if m.name == default else ""
        print(f"{mark}{m.name:44} {m.runtime:15} {_bias_label(m.bias):18} {m.license}{tag}")
        if m.notes:
            print(f"{'':4}{m.notes}")
        if m.runtime == "sherpa-onnx" and (note := _sherpa_note()):
            print(f"{'':4}{note}")
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
    from voiceio.core.decoders.catalog import info, runtime_of
    name = args.name
    if name != "auto":
        m = info(name)
        if m is not None and m.experimental:
            print(f"note: {name} is experimental — {m.notes}")
        if runtime_of(name) == "sherpa-onnx":
            from voiceio.core.decoders.sherpa import features
            if not features().installed:
                raise SystemExit("needs sherpa-onnx first: voiceio models sherpa")
            if note := _sherpa_note():
                print(f"note: {note}")
            if m is not None and m.url:
                _download(name)
    set_value("model", "name", name)
    print(f"model set to {name} — restart the daemon: voiceio service restart")


def _download(name: str) -> None:
    from voiceio.core.decoders import fetch
    if fetch.is_downloaded(name):
        return
    try:
        path = fetch.download(name, progress=fetch.print_progress(name))
    except OSError as e:
        raise SystemExit(f"download failed: {e}") from e
    print(f"{name} ready in {path}")


def cmd_download(args: argparse.Namespace) -> None:
    from voiceio.core.decoders import fetch
    from voiceio.core.decoders.catalog import info
    m = info(args.name)
    if m is None or not m.url:
        raise SystemExit(f"{args.name} isn't downloaded by voiceio (Whisper models come "
                         "from Hugging Face on first use)")
    if fetch.is_downloaded(args.name):
        print(f"{args.name} is already in {fetch.model_path(args.name)}")
        return
    _download(args.name)


def cmd_sherpa(args: argparse.Namespace) -> None:
    """Parakeet's hotwords need two sherpa-onnx fixes upstream hasn't
    released yet; install voiceio's build of them into this Python."""
    from voiceio.core.decoders import fetch
    note = _sherpa_note()
    if args.check or not note:
        print(note or "sherpa-onnx has the Parakeet fixes (beam-search fix, pre-tokenized "
                      "hotwords)")
        return
    print(f"installing the patched sherpa-onnx from {fetch.SHERPA_WHEELS}…", flush=True)
    try:
        wheel = fetch.install_patched_sherpa()
    except OSError as e:
        raise SystemExit(str(e)) from e
    from voiceio import config
    from voiceio.core.decoders.catalog import PARAKEET, PARAKEET_V3
    name = PARAKEET if config.load().model.language == "en" else PARAKEET_V3
    print(f"installed {wheel}\nnext: voiceio models use {name}")
