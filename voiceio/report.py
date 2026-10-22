"""Install/usage reports for the public feedback board on voiceio.dev.

A report is posted publicly, so it carries only what helps reproduce an
install problem: versions, desktop, which backends were picked and which
checks failed. Never hostnames, usernames, paths under $HOME, device names,
transcripts, vocabulary, corrections or config values beyond backend/model
names. Everything that goes out passes through ``scrub()``.
"""
from __future__ import annotations

import getpass
import json
import os
import platform as _py_platform
import re
import socket
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voiceio.health import HealthReport

BOARD_URL = "https://voiceio.dev/#feedback"
POST_URL = "https://swarmmemo.com/w/voiceio/feedback"
TIMEOUT_SECS = 8.0
MAX_COMMENT = 300
MAX_DETAIL = 90

_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"\b(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}\b", re.I)
_MAC = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HOME_LIKE = re.compile(r"(?:/home|/Users|/var/home)/[^/\s'\"]+")
# /dev/input/by-id/usb-Vendor_Model_SERIAL-event-kbd names carry serials.
_DEVICE_ID = re.compile(r"/dev/(?:input|disk|serial)/by-(?:id|path|uuid)/\S+")
_LONG_HEX = re.compile(r"\b[0-9a-f]{12,}\b", re.I)


def _identities() -> list[tuple[str, str]]:
    """(string, placeholder) for the user and machine names, longest first."""
    out: dict[str, str] = {}
    for get, tag in ((socket.gethostname, "<host>"), (_py_platform.node, "<host>"),
                     (getpass.getuser, "<user>"),
                     (lambda: os.environ.get("USER", ""), "<user>"),
                     (lambda: os.environ.get("LOGNAME", ""), "<user>")):
        try:
            v = get()
        except Exception:  # noqa: BLE001
            continue
        for s in (v, v.split(".")[0]) if v else ():
            if len(s) >= 2:
                out[s] = tag
    return sorted(out.items(), key=lambda kv: len(kv[0]), reverse=True)


def _homes() -> list[str]:
    homes = {str(Path.home()), os.path.expanduser("~")}
    try:
        import pwd
        homes.add(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:  # noqa: BLE001 — not on every platform
        pass
    return sorted((h for h in homes if h and h != "/"), key=len, reverse=True)


def scrub(text: str) -> str:
    """Remove anything that identifies the machine or its user."""
    for home in _homes():
        text = text.replace(home, "~")
    text = _HOME_LIKE.sub("~", text)
    text = _DEVICE_ID.sub("<device>", text)
    text = _EMAIL.sub("<email>", text)
    text = _MAC.sub("<mac>", text)
    text = _IPV6.sub("<ip>", text)
    text = _IPV4.sub(lambda m: m.group(0) if _is_version(m.group(0)) else "<ip>", text)
    text = _LONG_HEX.sub("<id>", text)
    for ident, tag in _identities():
        text = re.sub(rf"(?<!\w){re.escape(ident)}(?!\w)", tag, text, flags=re.I)
    return text


def _is_version(dotted: str) -> bool:
    """1.10.0.1-style strings are versions, not addresses: keep small parts."""
    return all(int(p) < 100 for p in dotted.split("."))


def _distro() -> str:
    if sys.platform.startswith("linux"):
        try:
            rel = _py_platform.freedesktop_os_release()
        except OSError:
            return "Linux (unknown distro)"
        name = rel.get("NAME") or rel.get("ID") or "Linux"
        version = rel.get("VERSION_ID") or rel.get("BUILD_ID") or ""
        return f"{name} {version}".strip()
    if sys.platform == "darwin":
        return f"macOS {_py_platform.mac_ver()[0]}".strip()
    return f"{_py_platform.system()} {_py_platform.release()}".strip()


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"


def _model_label(name: str) -> str:
    """The model's catalog name; a local path says only that it is one."""
    if name.startswith(("/", "~", ".")) or (os.sep in name and Path(name).expanduser().exists()):
        from voiceio.learn.store import learn_dir
        try:
            if learn_dir() / "models" in Path(name).expanduser().parents:
                return "fine-tuned (local)"
        except Exception:  # noqa: BLE001
            pass
        return "custom (local path)"
    return name


def build_report(
    health: HealthReport,
    *,
    missing_deps: list[str] | None = None,
    perm_issues: int = 0,
    comment: str = "",
    cfg=None,
) -> str:
    """The exact public text: short, plain, and scrubbed."""
    from voiceio import __version__
    if cfg is None:
        from voiceio import config
        cfg = config.load()

    p = health.platform
    hotkey = next((b for b in health.hotkey_backends if b.ok), None)
    typer = next((b for b in health.typer_backends if b.ok), None)
    audio = "ok" if health.audio_ok else "FAILED"

    failed: list[str] = []
    warned: list[str] = []
    if hotkey is None:
        failed += [f"hotkey {b.name}: {_short(b.reason)}" for b in health.hotkey_backends] \
            or ["hotkey: no backend"]
    if typer is None:
        failed += [f"typer {b.name}: {_short(b.reason)}" for b in health.typer_backends] \
            or ["typer: no backend"]
    if not health.audio_ok:
        failed.append(f"audio: {_short(health.audio_reason)}")
    if missing_deps:
        from voiceio import platform as plat
        failed.append("missing system deps: " + ", ".join(plat.dep_label(k) for k in missing_deps))
    if health.ibus_checks is not None:
        ib = health.ibus_checks
        for label, ok in (("component", ib.component_installed), ("daemon", ib.daemon_running),
                          ("engine ping", ib.socket_reachable),
                          ("env persisted", ib.env_persisted)):
            if not ok:
                warned.append(f"ibus {label}")
        if p.is_gnome and not ib.gnome_source_configured:
            warned.append("ibus gnome input source")
    if not health.cli_in_path:
        warned.append("cli not in PATH")
    if not health.tray_ok:
        warned.append(f"tray: {_short(health.tray_reason)}" if health.tray_reason else "tray")
    if perm_issues:
        warned.append(f"insecure permissions on {perm_issues} file(s)")
    unavailable = [b.name for chosen, chain in ((hotkey, health.hotkey_backends),
                                                (typer, health.typer_backends))
                   if chosen is not None for b in chain if not b.ok]
    warned += [_short(w) for w in health.warnings]

    lines = [
        f"voiceio {__version__} report",
        f"system: {_distro()} · {p.desktop} · {p.display_server} · "
        f"python {_py_platform.python_version()}",
        f"backends: hotkey {hotkey.name if hotkey else 'none'} · "
        f"typer {typer.name if typer else 'none'} · audio {audio} "
        f"(vad {cfg.audio.vad_backend})",
        f"model: {_model_label(cfg.model.name)}",
    ]
    if unavailable:
        lines.append("fallbacks unavailable: " + ", ".join(unavailable))
    lines.append("failed: " + "; ".join(failed) if failed else "failed: none")
    if warned:
        lines.append("warnings: " + "; ".join(warned))
    comment = " ".join(comment.split())[:MAX_COMMENT]
    if comment:
        lines.append(f"comment: {comment}")
    return scrub("\n".join(lines))


def post(text: str, *, timeout: float = TIMEOUT_SECS) -> tuple[bool, str]:
    """Post to the board. (True, receipt id) or (False, why). Never raises."""
    query = urllib.parse.urlencode({
        "format": "json", "text": text, "request_id": f"voiceio-{uuid.uuid4().hex}",
    })
    try:
        with _urlopen(f"{POST_URL}?{query}", timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001 — a feedback post must never break the CLI
        return False, type(e).__name__ + (f": {e}" if str(e) else "")
    if not isinstance(body, dict):
        return False, "unexpected response"
    receipt = body.get("receipt")
    if body.get("ok") is True and isinstance(receipt, dict) and receipt.get("id"):
        return True, str(receipt["id"])
    return False, str(body.get("error") or "unexpected response")[:200]


def _urlopen(url: str, *, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "voiceio-report"})
    return urllib.request.urlopen(req, timeout=timeout)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def share(text: str, *, yes: bool = False) -> bool:
    """Show the exact report, confirm, post. True only if it was posted."""
    print("\nThis report will be posted PUBLICLY to the voiceio feedback board:\n")
    print("\n".join("  " + ln for ln in text.splitlines()))
    print()
    if not yes:
        if not _interactive():
            print("Not posted: no terminal to confirm on. Re-run with --yes to post it.")
            return False
        if _ask("Post it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Not posted.")
            return False
    ok, detail = post(text)
    if ok:
        print(f"Posted, thanks. See it at {BOARD_URL}")
        return True
    print(f"Could not confirm the post ({scrub(detail)}). Check {BOARD_URL} "
          f"and paste it there if it's missing.")
    return False


def gather(comment: str = "") -> tuple[HealthReport, str]:
    """Run the doctor checks and build the report from them."""
    from voiceio import config
    from voiceio import platform as plat
    from voiceio.health import check_health
    health = check_health()
    return health, build_report(
        health,
        missing_deps=plat.check_system_deps(),
        perm_issues=len(config.check_permissions()),
        comment=comment,
    )


def offer_after_setup() -> None:
    """End of the setup wizard: one optional line for the board. Enter skips;
    nothing here can fail setup."""
    try:
        if not _interactive():
            return
        comment = _ask("\n  How did setup go? One line for the public feedback board "
                       "(Enter to skip): ").strip()
        if not comment:
            return
        _, text = gather(comment)
        share(text)
    except Exception as e:  # noqa: BLE001
        print(f"  (Skipped the feedback report: {type(e).__name__})")
