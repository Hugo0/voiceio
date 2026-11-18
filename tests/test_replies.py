"""`voiceio replies` and doctor's nudge: saved receipts, parsing, offline."""
from __future__ import annotations

import io
import json

import pytest

from voiceio import replies, report

DAY = 86400
NOW = 1_800_000_000.0
MINE = "a" * 32
OTHER_KEY = "10d3d7e7" + "0" * 56


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _msg(id, author, created, text="hi", reply_to=MINE, **kw):
    m = {"id": id, "author": author, "created_at": created, "text": text,
         "hidden": False, "reply_to": reply_to}
    m.update(kw)
    return m


def _thread(*msgs, root_at=NOW - DAY):
    root = {"id": MINE, "author": "anonymous", "created_at": root_at,
            "text": "voiceio 1.18.5 report", "hidden": False}
    return {"ok": True, "messages": [root, *msgs]}


@pytest.fixture
def board(monkeypatch):
    """Serve threads by id; records the URLs and timeouts asked for."""
    threads: dict[str, dict] = {}
    calls: list[tuple[str, float]] = []

    def fake(url, *, timeout):
        calls.append((url, timeout))
        rid = url.removeprefix(replies.THREAD_API).split("?")[0]
        if rid not in threads:
            raise ConnectionRefusedError("offline")
        return _Resp(json.dumps(threads[rid]).encode())

    monkeypatch.setattr(report, "_urlopen", fake)
    return threads, calls


# ── save / prune ──

def test_remember_saves_id_and_time_only():
    replies.remember(MINE, now=NOW)
    raw = json.loads(replies._path().read_text())
    assert raw == {"reports": [{"id": MINE, "posted_at": NOW, "read_until": 0.0}]}
    assert replies._path().stat().st_mode & 0o077 == 0


def test_prune_keeps_twenty_newest_and_drops_old():
    for i in range(25):
        replies.remember(f"r{i:02d}", now=NOW - 30 * DAY + i)
    replies.remember("old", now=NOW - 61 * DAY)
    kept = replies.saved(now=NOW)
    assert len(kept) == replies.MAX_SAVED
    assert kept[0]["id"] == "r24" and kept[-1]["id"] == "r05"
    assert "old" not in {r["id"] for r in kept}
    assert replies.saved(now=NOW + 31 * DAY) == []


def test_remember_ignores_bad_ids_and_bad_files():
    replies._path().write_text("not json")
    assert replies.saved() == []
    replies.remember("../etc/passwd")
    assert replies.saved() == []
    replies.remember(MINE)
    assert [r["id"] for r in replies.saved()] == [MINE]


def test_remember_never_raises(monkeypatch):
    monkeypatch.setattr(replies, "_save", lambda r: 1 / 0)
    replies.remember(MINE)


def test_share_remembers_the_receipt(monkeypatch, capsys):
    monkeypatch.setattr(report, "post", lambda text: (True, MINE))
    monkeypatch.setattr(report, "_interactive", lambda: False)
    assert report.share("r", yes=True) is True
    assert [r["id"] for r in replies.saved()] == [MINE]
    out = capsys.readouterr().out
    assert f"https://swarmmemo.com/e/{MINE}" in out and "voiceio replies" in out
    assert "within a day" in out


def test_failed_post_remembers_nothing(monkeypatch):
    monkeypatch.setattr(report, "post", lambda text: (False, "offline"))
    report.share("r", yes=True)
    assert replies.saved() == []


# ── parsing ──

def test_parse_labels_maintainer_by_key_not_handle():
    body = _thread(
        _msg("m1", replies.MAINTAINER_KEY, NOW - 100, author_handle="voiceio"),
        _msg("f1", OTHER_KEY, NOW - 50, author_handle="voiceio"),  # a lookalike
        _msg("x1", "anonymous", NOW - 10),
    )
    got = replies.parse_thread(body, MINE)
    assert [(r.id, r.label) for r in got] == [
        ("m1", "voiceio maintainer"), ("f1", "reply"), ("x1", "reply")]


def test_parse_skips_own_post_and_hidden_and_sorts():
    body = _thread(_msg("b", "anonymous", NOW - 1), _msg("a", "anonymous", NOW - 9),
                   _msg("h", replies.MAINTAINER_KEY, NOW - 5, hidden=True))
    assert [r.id for r in replies.parse_thread(body, MINE)] == ["a", "b"]


def test_parse_strips_escapes_and_control_chars():
    evil = "ok\x1b[2J\x1b]0;pwned\x07 \x1b[31mred\x1b[0m\r\nline2\x00\x9b"
    [r] = replies.parse_thread(_thread(_msg("e", "anonymous", NOW, evil)), MINE)
    assert "\x1b" not in r.text and "\x07" not in r.text and "\x00" not in r.text
    assert "\x9b" not in r.text and "\r" not in r.text
    assert r.text.startswith("ok") and "red" in r.text and r.text.endswith("line2")


@pytest.mark.parametrize("body", [[], {"ok": False}, {"ok": True}, {"ok": True, "messages": 3}])
def test_parse_rejects_non_threads(body):
    with pytest.raises(ValueError):
        replies.parse_thread(body, MINE)


def test_new_is_relative_to_read_until():
    body = _thread(_msg("old", "anonymous", NOW - 100), _msg("new", "anonymous", NOW - 1))
    got = replies.parse_thread(body, MINE, read_until=NOW - 50)
    assert [(r.id, r.new) for r in got] == [("old", False), ("new", True)]


# ── voiceio replies ──

def test_show_without_reports_says_how_to_file(capsys, board):
    assert replies.show() == 0
    out = capsys.readouterr().out
    assert "voiceio doctor --report" in out
    assert board[1] == []


def test_show_prints_marks_new_and_then_not(capsys, board):
    threads, calls = board
    replies.remember(MINE)
    threads[MINE] = _thread(_msg("m1", replies.MAINTAINER_KEY, NOW, "Try xdotool."))
    assert replies.show() == 0
    out = capsys.readouterr().out
    assert "voiceio maintainer" in out and "[new]" in out and "Try xdotool." in out
    assert calls[0][1] <= 5
    assert replies.show() == 0
    assert "[new]" not in capsys.readouterr().out


def test_show_newest_report_first_and_one_offline(capsys, board):
    threads, _ = board
    replies.remember("older", now=NOW - 5 * DAY)
    replies.remember(MINE, now=NOW - DAY)
    threads[MINE] = _thread()
    assert replies.show() == 0
    out = capsys.readouterr().out
    assert out.index(MINE) < out.index("older")
    assert "No replies yet" in out and "Could not load the replies" in out


def test_show_offline_is_a_clear_message(capsys):
    # conftest refuses every connection.
    replies.remember(MINE)
    assert replies.show() == 1
    out = capsys.readouterr().out
    assert "offline" in out and f"https://swarmmemo.com/e/{MINE}" in out


def test_show_failure_does_not_mark_read(capsys, board):
    threads, _ = board
    replies.remember(MINE)
    replies.show()  # offline for this id
    threads[MINE] = _thread(_msg("m1", replies.MAINTAINER_KEY, NOW))
    replies.show()
    assert "[new]" in capsys.readouterr().out


# ── doctor's nudge ──

def test_nudge_when_maintainer_answered_since_last_read(board):
    threads, calls = board
    replies.remember(MINE)
    threads[MINE] = _thread(_msg("m1", replies.MAINTAINER_KEY, NOW))
    assert "voiceio replies" in replies.nudge()
    assert len(calls) == 1 and calls[0][1] <= 2


def test_nudge_quiet_after_reading(board, capsys):
    threads, _ = board
    replies.remember(MINE)
    threads[MINE] = _thread(_msg("m1", replies.MAINTAINER_KEY, NOW))
    replies.show()
    assert replies.nudge() == ""


def test_nudge_ignores_non_maintainer_replies(board):
    threads, _ = board
    replies.remember(MINE)
    threads[MINE] = _thread(_msg("x", OTHER_KEY, NOW))
    assert replies.nudge() == ""


def test_nudge_makes_no_request_without_recent_reports(board):
    _, calls = board
    assert replies.nudge() == ""
    replies.remember(MINE, now=replies.time.time() - 40 * DAY)
    assert replies.nudge() == ""
    assert calls == []


def test_nudge_only_asks_about_the_newest_report(board):
    threads, calls = board
    replies.remember("older", now=replies.time.time() - 3 * DAY)
    replies.remember(MINE)
    threads[MINE] = _thread()
    replies.nudge()
    assert len(calls) == 1 and MINE in calls[0][0]


def test_nudge_offline_is_silent():
    replies.remember(MINE)
    assert replies.nudge() == ""


def test_replies_subcommand_is_registered(monkeypatch, capsys):
    import sys

    from voiceio.cli import main
    monkeypatch.setattr(sys, "argv", ["voiceio", "replies"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    assert "voiceio doctor --report" in capsys.readouterr().out


def test_parse_keeps_only_answers_below_the_report():
    """The API returns the conversation from its first post; a report that was
    itself a reply must not list its ancestors or side branches."""
    body = {"ok": True, "messages": [
        _msg("root", "anonymous", NOW - 9, reply_to=None),
        _msg(MINE, "anonymous", NOW - 8, reply_to="root"),
        _msg("side", replies.MAINTAINER_KEY, NOW - 7, reply_to="root"),
        _msg("ans", replies.MAINTAINER_KEY, NOW - 6, reply_to=MINE),
        _msg("deep", OTHER_KEY, NOW - 5, reply_to="ans"),
    ]}
    assert [r.id for r in replies.parse_thread(body, MINE)] == ["ans", "deep"]
