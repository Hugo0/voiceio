"""`voiceio say`: the message format, the queue, the socket and the CLI."""
import threading
import time
from unittest.mock import patch

from voiceio import announce
from voiceio.announce import Message


def test_encode_parse_round_trip():
    msg = Message("tests are green", "build bot", urgent=True)
    cmd = announce.encode(msg)
    assert cmd.startswith("say:")
    assert announce.parse(cmd[len("say:"):]) == msg
    assert msg.spoken == "build bot: tests are green"
    assert Message("hi").spoken == "hi"


def test_parse_rejects_junk_and_caps_length():
    assert announce.parse("not json") is None
    assert announce.parse('["a list"]') is None
    assert announce.parse('{"text": "   "}') is None
    long = announce.parse('{"text": "%s", "from": "x\\n y"}' % ("a" * 5000))
    assert len(long.text) == announce.MAX_TEXT and long.sender == "x y"


def test_announcer_holds_while_busy_then_speaks_in_order():
    spoken, busy = [], threading.Event()
    busy.set()
    done = threading.Event()

    def speak(text):
        spoken.append(text)
        if len(spoken) == 2:
            done.set()

    with patch.object(announce, "cue_and_notify"):
        a = announce.Announcer(speak=speak, busy=busy.is_set)
        a.submit(Message("one", "a"))
        a.submit(Message("two", "b"))
        time.sleep(0.4)
        assert spoken == []          # held while the user dictates
        busy.clear()
        assert done.wait(3)
    assert spoken == ["a: one", "b: two"]


def test_announcer_drops_when_full():
    hold, spoken = threading.Event(), []
    with patch.object(announce, "cue_and_notify"):
        a = announce.Announcer(speak=lambda t: (hold.wait(5), spoken.append(t)), busy=lambda: False)
        results = [a.submit(Message(str(i))) for i in range(announce._QUEUE + 3)]
        hold.set()
        deadline = time.monotonic() + 3   # drain inside the patch: no real notifications
        while a._queue.unfinished_tasks and len(spoken) < sum(results) and time.monotonic() < deadline:
            time.sleep(0.05)
    assert not all(results)


def test_socket_routes_payload_commands(tmp_path, monkeypatch):
    from voiceio.hotkeys import socket_backend as sb
    monkeypatch.setattr(sb, "SOCKET_PATH", tmp_path / "v.sock")
    got = threading.Event()
    payloads = []
    hk = sb.SocketHotkey()
    hk.handle_payload("say", lambda p: (payloads.append(p), got.set()))
    hk.start("x", lambda: None)
    try:
        assert sb.send_command(announce.encode(Message("hello", "agent")))
        assert got.wait(3)
    finally:
        hk.stop()
    assert announce.parse(payloads[0]) == Message("hello", "agent")


def test_cli_speaks_here_without_a_daemon(tmp_path, monkeypatch, capsys):
    import sys
    from voiceio import cli
    from voiceio.hotkeys import socket_backend as sb
    monkeypatch.setattr(sb, "SOCKET_PATH", tmp_path / "absent.sock")
    monkeypatch.setattr(sys, "argv", ["voiceio", "say", "--from", "bot", "hi", "there"])
    with patch.object(announce, "say_here", return_value=True) as here:
        cli.main()
    here.assert_called_once_with(Message("hi there", "bot", False))
