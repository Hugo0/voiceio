"""`voiceio learn` — personalize recognition from your own dictation."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from voiceio.learn import store


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("learn", help="Personalize recognition from your own dictation",
                       description=__doc__)
    lsub = p.add_subparsers(dest="learn_cmd", metavar="STEP")
    p.set_defaults(func=lambda a: (p.print_help(), cmd_status(a)))

    lsub.add_parser("status", help="What has been captured, labeled and trained").set_defaults(
        func=cmd_status)

    c = lsub.add_parser("clean", help="Remove test-fixture rows from your data files")
    c.add_argument("--apply", action="store_true", help="rewrite the files (backups first)")
    c.set_defaults(func=cmd_clean)

    im = lsub.add_parser("import", help="Add your own recordings from elsewhere (e.g. phone voice notes)")
    im.add_argument("folder", type=Path, help="searched recursively for audio files")
    im.add_argument("--source", default="import",
                    help="tag for where these came from, e.g. phone (default: import)")
    im.set_defaults(func=cmd_import)

    lab = lsub.add_parser("label", help="Transcribe retained clips with a stronger teacher model")
    lab.add_argument("--teacher", default=None, help="model name or path (default large-v3-turbo)")
    lab.add_argument("--limit", type=int, default=None)
    lab.add_argument("--relabel", action="store_true")
    lab.set_defaults(func=cmd_label)

    r = lsub.add_parser("review", help="Fix the teacher's text where it matters")
    r.add_argument("--limit", type=int, default=None, help="max segments this session")
    r.add_argument("--test", action="store_true",
                   help="verify every segment of the held-out clips (the eval's ground truth)")
    r.set_defaults(func=cmd_review)

    a = lsub.add_parser("augment", help="Synthesize extra examples of your hard words (local TTS)")
    a.add_argument("--terms", type=int, default=40, help="how many vocabulary terms")
    a.add_argument("--per-term", type=int, default=8)
    a.set_defaults(func=cmd_augment)

    b = lsub.add_parser("build", help="Freeze labeled clips into a dataset version")
    b.add_argument("--reviewed-only", action="store_true")
    b.set_defaults(func=cmd_build)

    e = lsub.add_parser("eval", help="Score a model on your held-out clips")
    e.add_argument("--model", default=None, help="model name or path (default: configured)")
    e.add_argument("--dataset", default=None, help="dataset version (default: latest)")
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_eval)

    t = lsub.add_parser("train", help="Fine-tune on the latest dataset (LoRA)")
    t.add_argument("--base", default="small")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--synthetic-share", type=float, default=0.3)
    t.add_argument("--full", action="store_true",
                   help="also adapt the encoder to your voice/mic (slow on CPU; default on GPU)")
    t.add_argument("--remote", action="store_true",
                   help="only export, and print the commands for a rented GPU")
    t.set_defaults(func=cmd_train)

    pr = lsub.add_parser("promote", help="Use a trained model for dictation")
    pr.add_argument("run", help="run name under learn/models (see `learn status`)")
    pr.set_defaults(func=cmd_promote)

    lsub.add_parser("rollback", help="Return to the model used before the last promote"
                    ).set_defaults(func=cmd_rollback)


def _yield_cpu() -> None:
    """Run heavy learn jobs on CPU nobody else wants, so dictation never waits.

    Re-execs into a transient systemd scope with CPUWeight=idle: a cgroup-
    level idle class that yields to every other cgroup. SCHED_IDLE or nice
    alone only rank threads *inside* one cgroup — measured here, a SCHED_IDLE
    labeler in another slice still took ~5 cores from the dictation service.
    """
    if os.environ.get("VOICEIO_IDLE_SCOPE") != "1" and _idle_scope_available():
        import sys
        os.environ["VOICEIO_IDLE_SCOPE"] = "1"
        os.execvp("systemd-run", [*_IDLE_SCOPE, "--", sys.executable, *sys.argv])
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        try:
            os.nice(19)
        except OSError:
            pass


_IDLE_SCOPE = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
               "-p", "CPUWeight=idle", "-p", "IOWeight=1"]


def _idle_scope_available() -> bool:
    import shutil
    import subprocess
    if not shutil.which("systemd-run"):
        return False
    try:  # needs a user manager and systemd >= 252 for CPUWeight=idle
        return subprocess.run([*_IDLE_SCOPE, "--", "true"], capture_output=True,
                              timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _cfg():
    from voiceio import config
    return config.load()


def _vocabulary() -> list[str]:
    from voiceio.vocabulary import VocabularyLoader
    return VocabularyLoader(_cfg().model).get_all()


def _models_dir() -> Path:
    return store.learn_dir() / "models"


# ── commands ─────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> None:
    clips = list(store.clips())
    labels = store.load_labels()
    hours = sum(c.duration for c in clips) / 3600
    human = sum(1 for c in clips if c.audio in labels and labels[c.audio].reviewed)
    teacher = sum(1 for c in clips if c.audio in labels) - human
    import collections
    by_source = collections.Counter(c.source for c in clips)
    print(f"captured   {len(clips)} clips with audio ({hours:.1f} h): "
          + ", ".join(f"{n} {s}" for s, n in by_source.most_common()))
    print(f"labeled    {teacher} by teacher, {human} reviewed by you")
    ds = store.latest_dataset()
    if ds:
        m = json.loads((ds / "manifest.json").read_text())
        print(f"dataset    {m['version']}: " + ", ".join(
            f"{s} {m['counts'][s]} ({m['hours'][s]} h)" for s in store.SPLITS))
    from voiceio.learn import augment
    synth = augment.augment_dir() / "manifest.jsonl"
    if synth.exists():
        print(f"synthetic  {len(synth.read_text().splitlines())} clips")
    runs = sorted(p.name for p in _models_dir().glob("*") if (p / "ct2").exists()) \
        if _models_dir().exists() else []
    if runs:
        print(f"models     {', '.join(runs)}")
    print(f"active     {_cfg().model.name}")
    nxt = ("label" if teacher + human < len(clips) else
           "review" if human == 0 else "build" if not ds else "eval / train")
    print(f"\nnext: voiceio learn {nxt}")


def cmd_clean(args: argparse.Namespace) -> None:
    from voiceio.learn.clean import clean
    for r in clean(apply=args.apply):
        verb = "removed" if args.apply else "would remove"
        detail = ", ".join(f"{n} {why}" for why, n in r.reasons.items()) or "nothing"
        print(f"{r.path.name}: {verb} {r.removed}/{r.total} ({detail})")
    if not args.apply:
        print("\ndry run — re-run with --apply to rewrite (backups are kept)")


def cmd_import(args: argparse.Namespace) -> None:
    import re

    from voiceio.learn import importer
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.source):
        raise SystemExit("--source: letters, digits, - and _ only")
    r = importer.import_dir(args.folder, args.source)
    print(f"imported {r.added} clips ({r.seconds / 3600:.2f} h) as '{args.source}'"
          f"; {r.duplicate} already imported, {r.rejected} unreadable or out of range")
    if r.added:
        print("next: voiceio learn label")


def cmd_label(args: argparse.Namespace) -> None:
    _yield_cpu()
    from voiceio.learn.teacher import DEFAULT_TEACHER, WhisperTeacher, label_clips
    cfg = _cfg()
    name = args.teacher or DEFAULT_TEACHER
    print(f"Loading teacher {name}…", flush=True)
    teacher = WhisperTeacher(name, vocabulary=_vocabulary(), language=cfg.model.language)
    clips = sorted(store.clips(), key=lambda c: c.duration)
    t0 = time.monotonic()

    def progress(n, clip, secs):
        print(f"  [{n}] {clip.audio} {clip.duration:.0f}s audio in {secs:.0f}s", flush=True)

    n = label_clips(clips, teacher, relabel=args.relabel, limit=args.limit, progress=progress)
    print(f"labeled {n} clips in {time.monotonic() - t0:.0f}s")


def cmd_review(args: argparse.Namespace) -> None:
    from voiceio.learn import review
    from voiceio.learn.teacher import load_wav

    clips, labels = list(store.clips()), store.load_labels()
    if args.test:
        items = review.queue_all([c for c in clips if store.split_of(c.audio) == "test"], labels)
    else:
        items = review.queue(clips, labels, _vocabulary())
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("Nothing to review.")
        return
    try:
        from voiceio.tts.player import TTSPlayer
        player = TTSPlayer()
    except Exception:  # noqa: BLE001 — review works without audio
        player = None
    print(f"{len(items)} segments to review. Enter = keep, type a correction, "
          "'p' = replay, 'q' = stop.\n")
    audio_cache: dict[str, object] = {}

    def ask(it: review.Item) -> str | None:
        seg = it.label.segments[it.index]
        print(f"── {it.clip.audio} {seg.start:.1f}-{seg.end:.1f}s  ({it.reason})")
        print(f"   teacher: {seg.text}")
        while True:
            if player is not None:
                if it.clip.audio not in audio_cache:
                    audio_cache.clear()
                    audio_cache[it.clip.audio] = load_wav(it.clip.path)
                a = audio_cache[it.clip.audio][int(seg.start * 16000):int(seg.end * 16000)]
                player.play((a * 32767).astype("int16"), 16000)
            answer = input("   > ").strip()
            if answer == "q":
                return None
            if answer != "p":
                return answer or seg.text

    saved = review.run(items, ask=ask,
                       save=lambda clip, segs: store.add_label(clip.audio, segs, "human"))
    print(f"\nsaved {saved} reviewed clips")


def cmd_augment(args: argparse.Namespace) -> None:
    _yield_cpu()
    from voiceio import tts
    from voiceio.learn import augment
    cfg = _cfg()
    engine = tts.select(cfg.tts, allow_network=False)
    if engine is None:
        raise SystemExit("no local TTS engine (install piper-tts or espeak-ng)")
    # Rarest words first: common words need no help.
    from voiceio.wordfreq import is_common
    terms = [t for t in _vocabulary() if not is_common(t, cfg.model.language)][:args.terms]
    n = augment.synthesize(terms, engine, per_term=args.per_term,
                           progress=lambda term, n: print(f"  {term}: {n} clips", flush=True))
    print(f"synthesized {n} clips with {engine.name}")


def cmd_build(args: argparse.Namespace) -> None:
    out = store.build_dataset(reviewed_only=args.reviewed_only)
    m = json.loads((out / "manifest.json").read_text())
    print(f"built {m['version']}: " + ", ".join(
        f"{s} {m['counts'][s]} ({m['hours'][s]} h)" for s in store.SPLITS))


def cmd_eval(args: argparse.Namespace) -> None:
    _yield_cpu()
    from voiceio.core.bias import BiasProvider
    from voiceio.learn import evaluate
    from voiceio.transcriber import Transcriber
    from voiceio.vocabulary import VocabularyLoader
    cfg = _cfg()
    ds = store.learn_dir() / "datasets" / args.dataset if args.dataset else store.latest_dataset()
    if ds is None or not ds.exists():
        raise SystemExit("no dataset — run `voiceio learn build` first")
    if args.model:
        cfg.model.name = args.model
    print(f"Loading {cfg.model.name}…", flush=True)
    decoder = Transcriber(cfg.model)
    try:
        bias = BiasProvider(VocabularyLoader(cfg.model)).bias_for(decoder.capabilities)
        result = evaluate.evaluate(decoder, ds, bias=bias, terms=_vocabulary(), limit=args.limit)
    finally:
        decoder.close()
    print(result.summary())
    print(f"saved {evaluate.save(result)}")


def cmd_train(args: argparse.Namespace) -> None:
    _yield_cpu()
    from voiceio.learn import augment, train
    ds = store.latest_dataset()
    if ds is None:
        raise SystemExit("no dataset — run `voiceio learn build` first")
    real = len(store.read_split(ds, "train"))
    synth = augment.rows(max_share=args.synthetic_share, real_rows=real)
    folder = store.learn_dir() / "exports" / ds.name
    counts = train.export(ds, folder, extra_train_rows=synth)
    print(f"exported {counts} windows ({len(synth)} synthetic) to {folder}")
    run = _models_dir() / f"{ds.name}-{args.base}-{time.strftime('%Y%m%d-%H%M')}"
    if args.remote:
        print("\n" + train.remote_recipe(folder, run, args.base))
        return
    train.require_train_deps()
    hf = train.train(folder, run, train.TrainConfig(
        base=args.base, epochs=args.epochs, decoder_only=False if args.full else None))
    train.convert(hf, run / "ct2")
    print(f"\ntrained {run.name}\nnext: voiceio learn eval --model {run / 'ct2'}"
          f"  (compare with the current model, then `voiceio learn promote {run.name}`)")


def cmd_promote(args: argparse.Namespace) -> None:
    from voiceio.config import set_value
    ct2 = _models_dir() / args.run / "ct2"
    if not (ct2 / "model.bin").exists():
        raise SystemExit(f"no converted model at {ct2}")
    history = _models_dir() / "promotions.jsonl"
    with open(history, "a") as f:
        f.write(json.dumps({"ts": time.time(), "from": _cfg().model.name, "to": str(ct2)}) + "\n")
    set_value("model", "name", str(ct2))
    print(f"now dictating with {args.run} — restart the daemon: voiceio service restart")


def cmd_rollback(args: argparse.Namespace) -> None:
    from voiceio.config import set_value
    history = _models_dir() / "promotions.jsonl"
    rows = history.read_text().splitlines() if history.exists() else []
    if not rows:
        raise SystemExit("nothing to roll back")
    last = json.loads(rows[-1])
    set_value("model", "name", last["from"])
    history.write_text("".join(r + "\n" for r in rows[:-1]))
    print(f"back to {last['from']} — restart the daemon: voiceio service restart")
