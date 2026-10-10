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
    e.add_argument("--streaming", action="store_true",
                   help="replay clips in real time through the dictation path; "
                        "also measures stop→text latency")
    e.add_argument("--lanes", type=int, default=None, help="decode lanes (default: config)")
    e.add_argument("--freeze", type=float, default=None,
                   help="streaming freeze seconds (default: config)")
    e.add_argument("--pipeline", action="store_true",
                   help="no decoding: score what each text stage (corrections, numbers, "
                        "cloud postcorrect…) did to past dictations")
    e.add_argument("--beam", type=int, default=None,
                   help="beam width of the passes you keep (default: [model] beam_size)")
    e.add_argument("--speed", type=float, default=1.0,
                   help="streaming replay speed; >1 is faster but skews latency")
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

    mt = lsub.add_parser("maintain", help="Label new clips; retrain and compare once enough piled up")
    mt.add_argument("--min-new", type=int, default=50,
                    help="new labeled clips needed before retraining (default 50)")
    mt.add_argument("--base", default="small")
    mt.add_argument("--epochs", type=int, default=4)
    mt.add_argument("--auto-promote", action="store_true",
                    help="switch to the fine-tune when it wins, and restart the daemon")
    mt.set_defaults(func=cmd_maintain)

    sc = lsub.add_parser("schedule", help="Run `maintain` every night (systemd timer)")
    sc.add_argument("state", nargs="?", choices=("on", "off"), help="omit to show")
    sc.add_argument("--auto-promote", action="store_true",
                    help="with on: switch to a fine-tune that wins, without asking")
    sc.set_defaults(func=cmd_schedule)

    pr = lsub.add_parser("promote", help="Use a trained model for dictation")
    pr.add_argument("run", help="run name under learn/models (see `learn status`)")
    pr.add_argument("--force", action="store_true",
                    help="promote without a winning eval on the latest dataset")
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
        # orig_argv keeps "-m voiceio": re-running __main__.py as a script
        # would put voiceio/ first on sys.path, where numbers.py and
        # platform.py shadow the standard library.
        os.execvp("systemd-run", [*_IDLE_SCOPE, "--", sys.executable, *sys.orig_argv[1:]])
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
    active = _cfg().model.name
    print(f"active     {active}")
    print(f"nightly    {_schedule_state()}")
    promotable = _print_scores(ds, active, runs) if ds else None
    if promotable:
        nxt = f"promote {promotable}"
    else:
        nxt = ("label" if teacher + human < len(clips) else
               "review" if human == 0 else "build" if not ds else "eval / train")
    print(f"\nnext: voiceio learn {nxt}")


def _print_scores(ds: Path, active: str, runs: list[str]) -> str | None:
    """Scores on `ds` for the active model and each fine-tune; returns the
    run that beats the active model on everything, if any."""
    from voiceio.learn import evaluate
    current = evaluate.latest(active, ds.name)
    rows = [(active + " (active)", current)] + [
        (run, evaluate.latest(str(_models_dir() / run / "ct2"), ds.name)) for run in runs]
    if not any(r for _, r in rows):
        return None
    print(f"\nscores on {ds.name} test clips:")
    best = None
    for name, r in rows:
        if r is None:
            print(f"  {name:<34} not evaluated — voiceio learn eval --model …")
            continue
        print(f"  {name:<34} WER {r['wer']:.1%}  vocab {r['term_recall']:.0%}  "
              f"loops {r['repetitions']}  [{r.get('gold', 0)}/{r['clips']} human-verified]")
        if current and r is not current and evaluate.promotion_verdict(r, current)[0]:
            if best is None or r["wer"] < best[1]["wer"]:
                best = (name, r)
    return best[0] if best else None


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
        live = review.live_span(seg.text, it.clip.raw) if it.clip.raw else None
        if live:
            print(f"   live:    {live}")
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
    if args.pipeline:
        return _eval_pipeline()
    if not args.streaming:  # streaming measures latency: run at dictation priority
        _yield_cpu()
    ds = store.learn_dir() / "datasets" / args.dataset if args.dataset else store.latest_dataset()
    if ds is None or not ds.exists():
        raise SystemExit("no dataset — run `voiceio learn build` first")
    _run_eval(args.model, ds, limit=args.limit, streaming=args.streaming, lanes=args.lanes,
              freeze=args.freeze, speed=args.speed, beam=args.beam)


def _run_eval(model: str | None, ds: Path, *, limit: int | None = None,
              streaming: bool = False, lanes: int | None = None,
              freeze: float | None = None, speed: float = 1.0, beam: int | None = None):
    """Score `model` (default: the configured one) on `ds`; print and save."""
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders import build_decoder
    from voiceio.learn import evaluate
    from voiceio.vocabulary import VocabularyLoader
    cfg = _cfg()
    if model:
        cfg.model.name = model
    if beam:
        cfg.model.beam_size = beam
    print(f"Loading {cfg.model.name}…", flush=True)
    # Offline scoring needs one lane; streaming mirrors the live setup.
    decoder = build_decoder(cfg.model, lanes=lanes if lanes is not None else
                            (None if streaming else 1))
    try:
        bias = BiasProvider(VocabularyLoader(cfg.model)).bias_for(decoder.capabilities)
        result = evaluate.evaluate(
            decoder, ds, bias=bias, terms=_vocabulary(), limit=limit, streaming=streaming,
            freeze_secs=cfg.output.streaming_freeze_secs if freeze is None else freeze,
            speed=speed)
    finally:
        decoder.close()
    print(result.summary())
    print(f"saved {evaluate.save(result)}")
    return result


def _live_hotwords() -> str:
    """The hotword string dictation prepends to every decode right now."""
    from voiceio.core.bias import BiasProvider
    from voiceio.core.decoders.faster_whisper import Transcriber
    from voiceio.vocabulary import VocabularyLoader
    model = _cfg().model
    return BiasProvider(VocabularyLoader(model)).bias_for(Transcriber.capabilities).hotwords or ""


def _eval_pipeline() -> None:
    from voiceio.commands import CommandProcessor
    from voiceio.core.pipeline import Pipeline
    from voiceio.corrections import CorrectionDict
    from voiceio.learn import evaluate
    cfg = _cfg()
    pipeline = Pipeline.from_config(cfg, corrections=CorrectionDict(),
                                    commands=CommandProcessor(), prefix=False)
    audit = evaluate.pipeline_audit(store.clips(), store.load_labels(), pipeline)
    print(audit.summary())
    if not any(lab.reviewed for lab in store.load_labels().values()):
        print("References are teacher transcripts: a stage that deletes fillers on purpose "
              "scores as \"further\".")


def cmd_train(args: argparse.Namespace) -> None:
    _yield_cpu()
    ds = store.latest_dataset()
    if ds is None:
        raise SystemExit("no dataset — run `voiceio learn build` first")
    run = _train(ds, base=args.base, epochs=args.epochs, synthetic_share=args.synthetic_share,
                 full=args.full, remote=args.remote)
    if run is not None:
        print(f"\ntrained {run.name}\nnext: voiceio learn eval --model {run / 'ct2'}"
              f"  (compare with the current model, then `voiceio learn promote {run.name}`)")


def _train(ds: Path, *, base: str = "small", epochs: int = 4, synthetic_share: float = 0.3,
           full: bool = False, remote: bool = False) -> Path | None:
    """Export `ds` and fine-tune on it; returns the run dir (None for --remote)."""
    from voiceio.learn import augment, train
    real = len(store.read_split(ds, "train"))
    synth = augment.rows(max_share=synthetic_share, real_rows=real)
    folder = store.learn_dir() / "exports" / ds.name
    counts = train.export(ds, folder, extra_train_rows=synth, hotwords=_live_hotwords())
    print(f"exported {counts} windows ({len(synth)} synthetic) to {folder}")
    run = _models_dir() / f"{ds.name}-{base}-{time.strftime('%Y%m%d-%H%M')}"
    if remote:
        print("\n" + train.remote_recipe(folder, run, base))
        return None
    train.require_train_deps()
    hotwords = (folder / train.HOTWORDS_FILE).read_text(encoding="utf-8")
    hf = train.train(folder, run, train.TrainConfig(
        base=base, epochs=epochs, decoder_only=False if full else None, hotwords=hotwords))
    train.convert(hf, run / "ct2")
    return run


def cmd_maintain(args: argparse.Namespace) -> None:
    """Label what's new; once enough has piled up, rebuild, retrain and say
    whether the result beats the current model. Never promotes by itself."""
    _yield_cpu()
    from voiceio.learn import evaluate
    cmd_label(argparse.Namespace(teacher=None, limit=None, relabel=False))
    from voiceio.learn import train
    if missing := train.missing_train_deps():
        # Labeling alone is worth it; retraining waits for the extra. Exit 0:
        # a nightly timer must not fail every night over an optional install.
        print(f"labeled; fine-tuning needs {', '.join(missing)}: {train.TRAIN_HINT}")
        return
    labels = store.load_labels()
    labeled = {c.audio for c in store.clips() if c.audio in labels}
    ds = store.latest_dataset()
    new = len(labeled - store.dataset_clips(ds)) if ds else len(labeled)
    if ds is not None and new < args.min_new:
        print(f"{new} new labeled clips since {ds.name}; retraining waits for {args.min_new}.")
        return
    ds = store.build_dataset()
    print(f"built {ds.name} ({new} new clips)")
    current = _run_eval(None, ds)
    run = _train(ds, base=args.base, epochs=args.epochs)
    candidate = _run_eval(str(run / "ct2"), ds)
    ok, table = evaluate.promotion_verdict(_scores(candidate), _scores(current))
    print(f"\n{_cfg().model.name} → {run.name} on {ds.name}:\n{table}")
    print(f"better or equal on everything — `voiceio learn promote {run.name}`" if ok
          else "not better on everything — keeping the current model")
    if not ok:
        return
    # Often runs unattended at night: say so where it will be seen.
    from voiceio import feedback
    if getattr(args, "auto_promote", False):
        cmd_promote(argparse.Namespace(run=run.name, force=False))
        from voiceio import service
        if service.is_installed():
            service.restart_when_idle()
        feedback.notify("VoiceIO: switched to a better model",
                        f"Trained on your dictation: {run.name}. Undo: voiceio learn rollback",
                        urgent=True)
    else:
        feedback.notify("VoiceIO: a better model is ready",
                        f"Trained on your dictation. Use it: voiceio learn promote {run.name}",
                        urgent=True)


def _schedule_state() -> str:
    from voiceio import service
    if not service.learn_timer_enabled():
        return "off (voiceio learn schedule on)"
    return "on (auto-promote)" if service.learn_timer_auto_promotes() else "on"


def cmd_schedule(args: argparse.Namespace) -> None:
    from voiceio import config, service
    if args.state == "on":
        if not config.load().data.retain_audio:
            print("note: [data] retain_audio is off, so there is nothing new to learn from;\n"
                  "      turn it on with: voiceio config set data.retain_audio true")
        from voiceio.learn import train
        if missing := train.missing_train_deps():
            print(f"note: until {', '.join(missing)} are installed it only labels new clips;\n"
                  f"      to also retrain: {train.TRAIN_HINT}")
        if not service.install_learn_timer(auto_promote=args.auto_promote):
            raise SystemExit("could not enable the timer (needs systemd --user)")
        print("on: every night ~03:30, `voiceio learn maintain` labels new dictation at idle "
              "priority, retrains once 50+ clips are new, and "
              + ("switches to the result when it wins on your held-out clips "
                 "(undo: voiceio learn rollback)." if args.auto_promote else
                 "tells you if the result is better. It never switches models by itself."))
    elif args.state == "off":
        service.uninstall_learn_timer()
        print("off")
    else:
        print(_schedule_state())


def _scores(result) -> dict:
    return {"wer": result.wer, "term_recall": result.term_recall,
            "repetitions": result.repetitions, "per_clip": result.per_clip}


def cmd_promote(args: argparse.Namespace) -> None:
    from voiceio.config import set_value
    ct2 = _models_dir() / args.run / "ct2"
    if not (ct2 / "model.bin").exists():
        raise SystemExit(f"no converted model at {ct2}")
    if not args.force:
        _check_beats_current(ct2)
    history = _models_dir() / "promotions.jsonl"
    with open(history, "a") as f:
        f.write(json.dumps({"ts": time.time(), "from": _cfg().model.name, "to": str(ct2)}) + "\n")
    set_value("model", "name", str(ct2))
    print(f"now dictating with {args.run} — restart the daemon: voiceio service restart")


def _check_beats_current(ct2: Path) -> None:
    """Refuse a promotion the evals don't support (a fine-tune can regress)."""
    from voiceio.learn import evaluate
    ds = store.latest_dataset()
    current = _cfg().model.name
    if ds is None:
        raise SystemExit("no dataset to compare on — `voiceio learn build`, or --force")
    missing = [m for m in (current, str(ct2)) if evaluate.latest(m, ds.name) is None]
    if missing:
        raise SystemExit("evaluate both models on the same clips first:\n"
                         + "".join(f"  voiceio learn eval --model {m}\n" for m in missing)
                         + "(or promote anyway with --force)")
    ok, table = evaluate.promotion_verdict(evaluate.latest(str(ct2), ds.name),
                                           evaluate.latest(current, ds.name))
    print(f"on {ds.name} test clips, {current} → fine-tune:\n{table}")
    if not ok:
        raise SystemExit("the fine-tune is worse on something — not promoting (--force overrides)")


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
