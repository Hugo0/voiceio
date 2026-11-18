"""Answers to the reports you posted with `voiceio doctor --report`.

The maintainer answers each report as a reply on the public board. voiceio
remembers the receipt id of each report it posted (nothing else), so
`voiceio replies` can read those threads and `voiceio doctor` can say when an
answer arrived. The only requests are GETs for those ids; nothing about the
user is sent. Board text is written by anyone, so it is cleaned of terminal
escapes and control characters before it is printed.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

THREAD_API = "https://swarmmemo.com/api/thread/"
# swarmmemo.com serves one post and its replies as a page to browsers (plain
# text to curl). The board on voiceio.dev only loads its comments when you
# scroll to them, so it cannot open at one report.
POST_PAGE = "https://swarmmemo.com/e/"
MAINTAINER_KEY = "3965ebe50d0e7427bea1441b7f997581299eaeadde993b3a1e90466411a64769"
MAX_SAVED = 20
KEEP_DAYS = 60
NUDGE_DAYS = 30
TIMEOUT_SECS = 5.0
NUDGE_TIMEOUT_SECS = 2.0

_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# C0/C1 controls and DEL, except newline and tab.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _path() -> Path:
    from voiceio.config import LOG_DIR
    return LOG_DIR / "reports.json"


def post_url(report_id: str) -> str:
    return POST_PAGE + urllib.parse.quote(report_id, safe="")


# ── saved reports ──

def _load() -> list[dict]:
    try:
        raw = json.loads(_path().read_text())
    except (OSError, ValueError):
        return []
    reports = raw.get("reports") if isinstance(raw, dict) else None
    out = []
    for r in reports if isinstance(reports, list) else ():
        if (isinstance(r, dict) and isinstance(r.get("id"), str) and _ID.match(r["id"])
                and isinstance(r.get("posted_at"), (int, float))):
            out.append({"id": r["id"], "posted_at": float(r["posted_at"]),
                        "read_until": float(r.get("read_until") or 0)})
    return out


def _prune(reports: list[dict], now: float) -> list[dict]:
    fresh = [r for r in reports if now - r["posted_at"] < KEEP_DAYS * 86400]
    fresh.sort(key=lambda r: r["posted_at"], reverse=True)
    return fresh[:MAX_SAVED]


def _save(reports: list[dict]) -> None:
    from voiceio.config import _SECURE_FILE, _chmod, private_dir
    path = _path()
    private_dir(path.parent)
    path.write_text(json.dumps({"reports": reports}, indent=1))
    _chmod(path, _SECURE_FILE)


def saved(now: float | None = None) -> list[dict]:
    """Saved reports, newest first, without the expired ones."""
    return _prune(_load(), time.time() if now is None else now)


def remember(report_id: str, now: float | None = None) -> None:
    """Keep the receipt id of a report just posted. Never raises."""
    now = time.time() if now is None else now
    try:
        if not _ID.match(report_id):
            return
        reports = [r for r in _load() if r["id"] != report_id]
        reports.append({"id": report_id, "posted_at": now, "read_until": 0.0})
        _save(_prune(reports, now))
    except Exception:  # noqa: BLE001 — remembering must never fail a post
        pass


def _mark_read(read: dict[str, float]) -> None:
    """Record, per report, the board time of the newest reply shown."""
    if not read:
        return
    try:
        reports = _load()
        for r in reports:
            if r["id"] in read:
                r["read_until"] = max(r["read_until"], read[r["id"]])
        _save(_prune(reports, time.time()))
    except Exception:  # noqa: BLE001
        pass


# ── the board ──

@dataclass
class Reply:
    id: str
    maintainer: bool
    created_at: float
    text: str
    new: bool = False

    @property
    def label(self) -> str:
        return "voiceio maintainer" if self.maintainer else "reply"


def clean(text: str) -> str:
    """Board text made safe for a terminal: no escapes, no control chars."""
    from voiceio.report import _ESCAPE
    text = _ESCAPE.sub("", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL.sub("", text).strip()


def parse_thread(body: object, report_id: str, read_until: float = 0.0) -> list[Reply]:
    """The visible replies in a thread, oldest first. Raises ValueError on a
    response that is not a thread."""
    if not isinstance(body, dict) or body.get("ok") is not True \
            or not isinstance(body.get("messages"), list):
        raise ValueError("unexpected response")
    messages = [m for m in body["messages"] if isinstance(m, dict)]
    # A thread is the whole conversation from its first post: keep only what
    # answers this report, directly or down a chain of replies.
    under = {report_id}
    grew = True
    while grew:
        grew = False
        for m in messages:
            if m.get("reply_to") in under and m.get("id") not in under:
                under.add(m.get("id"))
                grew = True
    out = []
    for m in messages:
        if m.get("id") == report_id or m.get("id") not in under or m.get("hidden"):
            continue
        try:
            created = float(m.get("created_at") or 0)
        except (TypeError, ValueError):
            created = 0.0
        # The key, not the handle: only the key proves who wrote it.
        maintainer = m.get("author") == MAINTAINER_KEY
        out.append(Reply(id=str(m.get("id", "")), maintainer=maintainer, created_at=created,
                         text=clean(m.get("text", "")), new=created > read_until))
    out.sort(key=lambda r: r.created_at)
    return out


def fetch(report_id: str, *, timeout: float = TIMEOUT_SECS) -> dict:
    """GET one thread as JSON. Raises on any failure; callers catch."""
    from voiceio import report
    url = THREAD_API + urllib.parse.quote(report_id, safe="") + "?limit=100"
    with report._urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


# ── front ends ──

def _when(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"


def show() -> int:
    """`voiceio replies`: print each saved report's replies. Exit code."""
    reports = saved()
    if not reports:
        print("You have no reports from this computer in the last "
              f"{KEEP_DAYS} days.\nFile one with: voiceio doctor --report")
        return 0
    failed = 0
    read: dict[str, float] = {}
    for r in reports:
        print(f"Report from {_when(r['posted_at'])}: {post_url(r['id'])}")
        try:
            replies = parse_thread(fetch(r["id"]), r["id"], r["read_until"])
        except Exception as e:  # noqa: BLE001 — offline, a bad answer: say so, go on
            failed += 1
            print(f"  Could not load the replies ({type(e).__name__}).\n")
            continue
        if not replies:
            print("  No replies yet. The maintainer usually answers within a day.\n")
            continue
        for rep in replies:
            mark = "  [new]" if rep.new else ""
            print(f"  {rep.label} · {_when(rep.created_at)}{mark}")
            for line in rep.text.splitlines() or [""]:
                print(f"    {line}".rstrip())
            print()
        read[r["id"]] = max(rep.created_at for rep in replies)
    _mark_read(read)
    if failed == len(reports):
        print("Could not reach swarmmemo.com. Are you offline? The links above open "
              "the reports in a browser.")
        return 1
    if read:
        print("To answer, open the report's link and reply there.")
    return 0


def nudge() -> str:
    """One line for `voiceio doctor` when the maintainer answered a recent
    report since you last ran `voiceio replies`, else "". One request to the
    newest recent report, short timeout, never raises."""
    try:
        now = time.time()
        recent = [r for r in saved(now) if now - r["posted_at"] < NUDGE_DAYS * 86400]
        if not recent:
            return ""
        r = recent[0]
        replies = parse_thread(fetch(r["id"], timeout=NUDGE_TIMEOUT_SECS),
                               r["id"], r["read_until"])
        if any(rep.maintainer and rep.new for rep in replies):
            return "The maintainer answered your report. Read it with: voiceio replies"
    except Exception:  # noqa: BLE001 — offline or anything else: say nothing
        pass
    return ""
