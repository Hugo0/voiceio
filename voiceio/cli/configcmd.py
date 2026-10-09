"""`voiceio config` — inspect and edit config.toml."""
from __future__ import annotations

import argparse
import os
import re
import subprocess

_SECRET = re.compile(r'^(\s*(?:api_key|token)\s*=\s*)"[^"]*"', re.MULTILINE)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("config", help="Show or edit your configuration", description=__doc__)
    cs = p.add_subparsers(dest="config_cmd", metavar="ACTION")
    p.set_defaults(func=cmd_show)
    cs.add_parser("show", help="Print config.toml with secrets redacted").set_defaults(
        func=cmd_show)
    cs.add_parser("path", help="Print the config file path").set_defaults(func=cmd_path)
    cs.add_parser("edit", help="Open config.toml in $EDITOR").set_defaults(func=cmd_edit)
    s = cs.add_parser("set", help="Set one value: config set SECTION.KEY VALUE")
    s.add_argument("key")
    s.add_argument("value")
    s.set_defaults(func=cmd_set)


def redact(text: str) -> str:
    return _SECRET.sub(r'\1"•••"', text)


def cmd_show(args: argparse.Namespace) -> None:
    from voiceio.config import CONFIG_PATH
    if not CONFIG_PATH.exists():
        print(f"No config yet ({CONFIG_PATH}). Run: voiceio setup")
        return
    print(redact(CONFIG_PATH.read_text(encoding="utf-8")), end="")


def cmd_path(args: argparse.Namespace) -> None:
    from voiceio.config import CONFIG_PATH
    print(CONFIG_PATH)


def cmd_edit(args: argparse.Namespace) -> None:
    from voiceio.config import CONFIG_PATH, load
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
    subprocess.call([editor, str(CONFIG_PATH)])
    load(path=CONFIG_PATH)  # surfaces parse warnings right away


def cmd_set(args: argparse.Namespace) -> None:
    import tomllib

    from voiceio.config import set_value
    section, _, key = args.key.rpartition(".")
    if not section:
        raise SystemExit("use SECTION.KEY, e.g. voiceio config set output.streaming_freeze_secs 8")
    try:
        value = tomllib.loads(f"v = {args.value}")["v"]  # numbers, booleans, quoted strings
    except tomllib.TOMLDecodeError:
        value = args.value
    set_value(section, key, value)
    print(f"[{section}] {key} = {value!r}")
