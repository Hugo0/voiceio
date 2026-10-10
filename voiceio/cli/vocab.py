"""`voiceio vocab` — the words you want recognized."""
from __future__ import annotations

import argparse

from voiceio.ui import BOLD, DIM, GREEN, RESET


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("vocab", help="Your vocabulary: names and jargon to recognize",
                       description=__doc__)
    vs = p.add_subparsers(dest="vocab_cmd", metavar="ACTION")
    p.set_defaults(func=cmd_show)
    vs.add_parser("show", help="Which terms bias the decoder right now").set_defaults(
        func=cmd_show)
    a = vs.add_parser("add", help="Add terms")
    a.add_argument("terms", nargs="+")
    a.set_defaults(func=cmd_add)
    r = vs.add_parser("remove", help="Remove terms")
    r.add_argument("terms", nargs="+")
    r.set_defaults(func=cmd_remove)
    s = vs.add_parser("suggest", help="Unusual words from your history that may belong here")
    s.add_argument("-n", "--limit", type=int, default=25)
    s.set_defaults(func=cmd_suggest)


def cmd_show(args: argparse.Namespace) -> None:
    """Split the vocabulary by what actually reaches the decoder.

    A bounded decoder (Whisper) can only be biased with ~35 terms; the rest
    still help elsewhere (eval, learn, the optional cloud pass) but do NOT
    bias decoding — this makes that line visible.
    """
    from voiceio.config import load
    from voiceio.tokens import HOTWORDS_TOKEN_BUDGET, count_tokens
    from voiceio.vocab_stats import VocabStats
    from voiceio.vocabulary import load_terms, select_terms

    cfg = load()
    terms = load_terms(cfg.model)
    if not terms:
        print("No vocabulary yet. Add names and jargon: voiceio vocab add Kalshi HyperNEAT")
        print("Find candidates in your history:     voiceio vocab suggest")
        return
    stats = VocabStats()
    stats.load()
    selected = select_terms(terms, token_budget=HOTWORDS_TOKEN_BUDGET,
                            model_name=cfg.model.name, stats=stats)
    used = count_tokens(", ".join(selected), cfg.model.name)
    print(f"{BOLD}Biasing the decoder{RESET} {DIM}({used}/{HOTWORDS_TOKEN_BUDGET} tokens, "
          f"ranked by recent use){RESET}")
    for t in selected:
        hits = stats.get(t).get("hits", 0)
        print(f"  {GREEN}●{RESET} {t}" + (f" {DIM}({hits} uses){RESET}" if hits else ""))
    tail = [t for t in terms if t not in set(selected)]
    if tail:
        print(f"\n{BOLD}Not biasing{RESET} {DIM}({len(tail)} terms past the budget — "
              f"a fine-tune learns them: voiceio learn){RESET}")
        print(f"  {DIM}{', '.join(tail[:12])}{'…' if len(tail) > 12 else ''}{RESET}")
    print(f"\n{len(terms)} terms: {len(selected)} biasing, {len(tail)} not")


def cmd_add(args: argparse.Namespace) -> None:
    from voiceio.config import load
    from voiceio.vocabulary import add_terms
    n = add_terms(args.terms, load().model)
    print(f"added {n} term(s)" + ("" if n == len(args.terms) else
                                  " (others were duplicates or near-duplicates)"))


def cmd_remove(args: argparse.Namespace) -> None:
    from voiceio.config import load
    from voiceio.vocabulary import remove_terms
    removed = remove_terms(args.terms, load().model)
    print(f"removed: {', '.join(removed)}" if removed else "none of those terms were present")


def cmd_suggest(args: argparse.Namespace) -> None:
    """Offline: words absent from the language's frequency list, ranked by
    how often you said them. Nothing leaves the machine."""
    from voiceio import history
    from voiceio.suggest import find_suspicious_words
    from voiceio.config import load
    from voiceio.corrections import CorrectionDict
    from voiceio.vocabulary import load_terms

    cfg = load()
    found = find_suspicious_words(
        history.read(), cfg.model.language if cfg.model.language != "auto" else "en",
        existing_corrections=set(CorrectionDict().list_all()),
        vocabulary=set(load_terms(cfg.model)),
    )
    if not found:
        print("No unusual words found in your history.")
        return
    found.sort(key=lambda s: -s.count)
    for s in found[:args.limit]:
        like = f" {DIM}(like {', '.join(s.similar_common[:2])}){RESET}" if s.similar_common else ""
        print(f"  {s.word:24} {s.count:4}×{like}")
    print("\nKeep the real names: voiceio vocab add WORD…  ·  "
          "fix mishearings: voiceio corrections add WRONG RIGHT")
