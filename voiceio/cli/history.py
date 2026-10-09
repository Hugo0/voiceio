"""`voiceio history` — what you dictated."""
from __future__ import annotations

import argparse
import time


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("history", help="What you dictated", description=__doc__)
    p.add_argument("-n", "--limit", type=int, default=20, help="entries to show (0 = all)")
    p.add_argument("-s", "--search", default=None, help="only entries containing this")
    p.add_argument("--clear", action="store_true", help="delete all history")
    p.add_argument("--path", action="store_true", help="print the history file path")
    p.set_defaults(func=cmd_history)


def cmd_history(args: argparse.Namespace) -> None:
    from voiceio import history
    from voiceio.config import HISTORY_PATH
    if args.path:
        print(HISTORY_PATH)
        return
    if args.clear:
        history.clear()
        print("History cleared.")
        return
    entries = history.search(args.search) if args.search else history.read(limit=args.limit)
    if not entries:
        print(f"No matches for: {args.search}" if args.search
              else "No history yet. Start dictating to build history.")
        return
    for e in reversed(entries):  # oldest first
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("ts", 0)))
        print(f"  [{ts}] {e.get('text', '')}")
    print(f"\n{len(entries)} shown")
    if not args.search and args.limit and len(entries) == args.limit:
        print(f"More: voiceio history -n {args.limit * 2}")
