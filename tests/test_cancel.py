"""Cancelling a dictation: nothing typed, nothing kept, no commit chime."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
from fakes import FakeDecoder

from voiceio.app import _State
from voiceio.config import Config
from voiceio.streaming import StreamingSession
from voiceio.typers.base import StreamingTyper, TyperBackend


def _session(typer):
    decoder = FakeDecoder()
    decoder.transcribe.return_value = "final words"
    recorder = MagicMock(sample_rate=16000)
    return StreamingSession(decoder=decoder, typer=typer, recorder=recorder)


def _audio(secs=2.0):
    return np.zeros(int(16000 * secs), dtype=np.float32)


# ── session ──────────────────────────────────────────────────────────────

def test_cancel_clears_the_preedit_and_suppresses_the_final():
    typer = MagicMock(spec=StreamingTyper)
    session = _session(typer)
    session._display.apply("hello there")        # interim preedit
    assert session.cancel()
    typer.clear_preedit.assert_called_once()
    session.stop(_audio())                        # a final pass already running
    typer.commit_text.assert_not_called()


def test_cancel_deletes_interim_words_a_keystroke_typer_typed():
    typer = MagicMock(spec=TyperBackend)
    session = _session(typer)
    session._display.apply("hello there")
    assert session.cancel()
    typer.delete_chars.assert_called_once_with(len("hello there"))


def test_too_late_once_the_final_is_out():
    typer = MagicMock(spec=StreamingTyper)
    session = _session(typer)
    session.stop(_audio())
    typer.commit_text.assert_called_once_with("final words")
    assert not session.cancel()
    typer.clear_preedit.assert_not_called()


def test_cancel_also_suppresses_the_interim_fallback_commit():
    typer = MagicMock(spec=StreamingTyper)
    session = _session(typer)
    session._display.apply("hello")
    session._decoder.transcribe.return_value = ""   # final yields nothing
    session.cancel()
    session.stop(_audio())
    typer.commit_text.assert_not_called()


# ── app ──────────────────────────────────────────────────────────────────

def _make_app(streaming=True, cancel_key="", hotkey_name="socket"):
    cfg = Config()
    cfg.output.streaming = streaming
    cfg.hotkey.cancel_key = cancel_key
    hotkey = MagicMock()
    hotkey.name = hotkey_name
    typer = MagicMock(spec=TyperBackend)
    typer.name = "clipboard"
    with patch("voiceio.app.hotkey_chain.select", return_value=hotkey), \
         patch("voiceio.app.typer_chain.select", return_value=typer), \
         patch("voiceio.app.build_decoder", return_value=MagicMock()), \
         patch("voiceio.app.plat.detect") as detect:
        detect.return_value = MagicMock(display_server="wayland", desktop="gnome")
        from voiceio.app import VoiceIO
        vio = VoiceIO(cfg)
    vio.recorder = MagicMock()
    vio.recorder.get_meter.return_value = {"clip_ratio": 0.0, "peak": 0.0}
    vio._quiet = MagicMock()
    vio._typers = MagicMock()
    vio.engine = MagicMock()
    vio._play_feedback = MagicMock()
    return vio


def test_cancel_while_recording():
    vio = _make_app()
    session = MagicMock()
    vio._state, vio._recording, vio._session = _State.RECORDING, 4, session
    gen = vio._generation
    with patch.object(vio, "_output_thread") as spawn:
        vio.cancel()
    assert vio._state == _State.IDLE and vio._session is None
    session.cancel.assert_called_once()
    session.stop.assert_not_called()             # no final decode
    vio.recorder.stop.assert_called_once()
    vio._typers.end.assert_called_once_with(4)   # IBus source released
    spawn.assert_not_called()                    # no finalizer, no history
    vio.engine.commit.assert_not_called()
    vio._play_feedback.assert_not_called()       # no commit chime
    # Nothing to supersede: an earlier dictation still finalizing keeps its output.
    assert vio._generation == gen


def test_cancel_when_idle_does_nothing():
    vio = _make_app()
    vio.cancel()
    vio.recorder.stop.assert_not_called()
    assert vio._state == _State.IDLE


def _finalizing(vio, tmp_path, session):
    """Start a real finalizer whose session.stop blocks until released."""
    from voiceio import retention
    retention.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    (retention.RECORDINGS_DIR / "clip.wav").write_bytes(b"RIFF")
    vio._retention_extra = lambda audio: {"audio": "clip.wav", "context": None, "model": "x"}
    vio._state = _State.RECORDING
    vio._session = session
    vio.recorder.stop.return_value = _audio()
    vio._do_stop()
    assert vio._state == _State.FINALIZING


def test_cancel_while_finalizing_keeps_nothing(tmp_path):
    vio = _make_app()
    in_stop, release = threading.Event(), threading.Event()
    session = MagicMock()
    session.cancel.return_value = True
    session.stop.side_effect = lambda audio: (in_stop.set(), release.wait(5), "words")[2]
    done = threading.Event()
    vio._typers.end.side_effect = lambda rid: done.set()
    _finalizing(vio, tmp_path, session)
    assert in_stop.wait(5)
    vio.cancel()
    assert vio._state == _State.IDLE
    release.set()
    assert done.wait(5)
    vio.engine.commit.assert_not_called()        # no history
    vio._play_feedback.assert_not_called()
    from voiceio import retention
    assert not (retention.RECORDINGS_DIR / "clip.wav").exists()
    assert not vio._cancelled


def test_cancel_too_late_changes_nothing(tmp_path):
    vio = _make_app()
    in_stop, release = threading.Event(), threading.Event()
    session = MagicMock()
    session.cancel.return_value = False          # final text already typed
    session.stop.side_effect = lambda audio: (in_stop.set(), release.wait(5), "words")[2]
    _finalizing(vio, tmp_path, session)
    assert in_stop.wait(5)
    gen = vio._generation
    vio.cancel()
    assert vio._state == _State.FINALIZING and vio._generation == gen
    release.set()
    for _ in range(200):
        if vio._state == _State.IDLE:
            break
        time.sleep(0.01)
    vio.engine.commit.assert_called_once()


def test_cancel_batch_decode_in_flight(tmp_path):
    from voiceio import retention
    vio = _make_app(streaming=False)
    vio._streaming = False
    retention.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    (retention.RECORDINGS_DIR / "clip.wav").write_bytes(b"RIFF")
    vio._retention_extra = lambda audio: {"audio": "clip.wav"}
    in_decode, release, done = threading.Event(), threading.Event(), threading.Event()
    vio.engine.transcribe.side_effect = lambda audio, bias=None: (
        in_decode.set(), release.wait(5), MagicMock(text="words", abort=False))[2]
    vio._typers.end.side_effect = lambda rid: done.set()
    vio._state = _State.RECORDING
    vio.recorder.stop.return_value = _audio()
    vio._do_stop()
    assert in_decode.wait(5)
    vio.cancel()
    release.set()
    assert done.wait(5)
    vio._typers.type_with_fallback.assert_not_called()
    vio.engine.commit.assert_not_called()
    assert not (retention.RECORDINGS_DIR / "clip.wav").exists()


def test_control_socket_and_cancel_key_wiring():
    vio = _make_app(cancel_key="escape", hotkey_name="evdev")
    vio._socket = MagicMock()
    vio._start_hotkeys()
    vio._socket.handle.assert_any_call("cancel", vio.cancel)
    vio._cancel_hotkey.start.assert_any_call("escape", vio.cancel)


def test_cancel_key_needs_a_key_reading_backend(caplog):
    vio = _make_app(cancel_key="escape", hotkey_name="socket")
    assert vio._cancel_hotkey is None
    assert "cancel_key" in caplog.text


def test_cli_cancel_sends_cancel(monkeypatch):
    import sys

    from voiceio import cli
    sent = []
    monkeypatch.setattr("voiceio.hotkeys.socket_backend.send_command",
                        lambda c: sent.append(c) or True)
    for argv, expected in ((["cancel"], "cancel"), (["toggle"], "toggle"),
                           (["toggle", "--release"], "release")):
        monkeypatch.setattr(sys, "argv", ["voiceio", *argv])
        cli.main()
        assert sent[-1] == expected
