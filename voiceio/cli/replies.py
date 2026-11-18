"""`voiceio replies` — answers to the reports you posted."""
from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "replies", help="Read the answers to reports you posted with doctor --report",
        description="Shows the replies to each report this computer posted in the last "
                    "60 days. It fetches only those threads from swarmmemo.com.",
    ).set_defaults(func=cmd_replies)


def cmd_replies(args: argparse.Namespace) -> None:
    from voiceio.replies import show
    sys.exit(show())
