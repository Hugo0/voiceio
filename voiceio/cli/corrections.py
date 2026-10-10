"""`voiceio corrections` — always replace a misheard word with the right one."""
from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("corrections", help="Replacements for words that keep being misheard",
                       description=__doc__)
    cs = p.add_subparsers(dest="corrections_cmd", metavar="ACTION")
    p.set_defaults(func=cmd_list)
    cs.add_parser("list", help="All corrections").set_defaults(func=cmd_list)
    a = cs.add_parser("add", help="Add: corrections add WRONG RIGHT")
    a.add_argument("wrong")
    a.add_argument("right")
    a.set_defaults(func=cmd_add)
    r = cs.add_parser("remove", help="Remove a correction")
    r.add_argument("wrong")
    r.set_defaults(func=cmd_remove)
    cs.add_parser("audit", help="Rules worth a second look (unrelated or real-word triggers)"
                  ).set_defaults(func=cmd_audit)
    f = cs.add_parser("flagged", help="Words you flagged by saying \"correct that\"")
    f.add_argument("--clear", action="store_true")
    f.set_defaults(func=cmd_flagged)


def cmd_list(args: argparse.Namespace) -> None:
    from voiceio.corrections import CorrectionDict
    items = CorrectionDict().list_all()
    if not items:
        print('No corrections. Add one: voiceio corrections add "wrong" "right"')
        return
    for wrong, right in sorted(items.items()):
        print(f"  {wrong} → {right}")
    print(f"\n{len(items)} correction(s)")


def cmd_audit(args: argparse.Namespace) -> None:
    from voiceio.config import load
    from voiceio.corrections import CorrectionDict, risky_rules
    risky = risky_rules(CorrectionDict().list_all(), load().model.language or "en")
    if not risky:
        print("Every correction looks like a plain mishearing fix.")
        return
    for wrong, right, why in risky:
        print(f"  {wrong!r:24} → {right!r:24} {why}")
    print(f"\n{len(risky)} rule(s) to review. Remove one: voiceio corrections remove WORD")


def cmd_add(args: argparse.Namespace) -> None:
    from voiceio.autocorrect import gate_correction
    from voiceio.config import load
    from voiceio.corrections import CorrectionDict
    cfg = load()
    # Manual adds are allowed, but warn when the rule would rewrite real words.
    reason = gate_correction(
        args.wrong, args.right,
        language=cfg.model.language if cfg.model.language != "auto" else "en",
        protect_languages=cfg.autocorrect.protect_languages,
    )
    if reason:
        print(f"warning: {reason}")
    CorrectionDict().add(args.wrong, args.right)
    print(f"added: {args.wrong} → {args.right}")


def cmd_remove(args: argparse.Namespace) -> None:
    from voiceio.corrections import CorrectionDict
    if not CorrectionDict().remove(args.wrong):
        print(f"not found: {args.wrong}")
        sys.exit(1)
    print(f"removed: {args.wrong}")


def cmd_flagged(args: argparse.Namespace) -> None:
    from voiceio.corrections import CorrectionDict
    cd = CorrectionDict()
    if args.clear:
        cd.clear_flagged()
        print("flagged words cleared")
        return
    words = cd.list_flagged()
    if not words:
        print('Nothing flagged. Say "correct that" while dictating to flag the last word.')
        return
    for w in words:
        print(f"  {w}")
    print(f'\nFix with: voiceio corrections add "{words[0]}" "RIGHT"')
