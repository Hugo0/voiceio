"""The crash spool: a dictation's audio survives a killed daemon."""
import os

import numpy as np

from voiceio import spool


def _wait(s):
    s.flush()


def test_recording_is_spooled_then_removed(tmp_path):
    s = spool.Spool(tmp_path)
    s.begin(7, 16000, prebuffer=np.ones(160, np.float32))
    s.write(7, np.full((320, 1), 0.5, np.float32))
    _wait(s)
    path = s.path(7, 16000)
    audio = np.fromfile(path, dtype=np.float32)
    assert len(audio) == 480 and audio[-1] == 0.5
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    s.finish(7)
    _wait(s)
    assert not path.exists()


def test_writes_for_unknown_or_finished_recordings_are_ignored(tmp_path):
    s = spool.Spool(tmp_path)
    s.write(1, np.ones(10, np.float32))
    s.finish(1)
    _wait(s)
    assert list(tmp_path.iterdir()) == []


def test_orphans_are_files_of_dead_processes(tmp_path, monkeypatch):
    long = np.zeros(32000, np.float32)
    (tmp_path / "recording-999999-3-16000.f32").write_bytes(long.tobytes())      # dead
    (tmp_path / f"recording-{os.getpid()}-4-16000.f32").write_bytes(long.tobytes())  # us
    (tmp_path / "recording-999998-1-16000.f32").write_bytes(np.zeros(100, np.float32).tobytes())
    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.setattr(spool, "_alive", lambda pid: pid != 999999 and pid != 999998)
    found = spool.orphans(tmp_path)
    assert [(p.name, len(a), r) for p, a, r in found] == [("recording-999999-3-16000.f32", 32000, 16000)]
    assert not (tmp_path / "recording-999998-1-16000.f32").exists()   # too short: dropped
    assert (tmp_path / f"recording-{os.getpid()}-4-16000.f32").exists()


def test_missing_dir_has_no_orphans(tmp_path):
    assert spool.orphans(tmp_path / "nope") == []


def test_recorder_taps_every_chunk_of_a_recording():
    from unittest.mock import MagicMock
    from voiceio.config import AudioConfig
    from voiceio.recorder import AudioRecorder
    rec = AudioRecorder(AudioConfig())
    rec._stream = MagicMock()
    rec._ring.append(np.ones((80, 1), np.float32))
    got = []
    rec.start(tap=got.append)
    rec._callback(np.full((160, 1), 0.2, np.float32), 160, None, None)
    rec.stop()
    rec._callback(np.full((160, 1), 0.3, np.float32), 160, None, None)   # after stop: not tapped
    assert [len(c) for c in got] == [80, 160]


def test_daemon_recovers_an_orphan_into_history(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from voiceio import app, feedback, history
    orphan = tmp_path / "recording-999999-1-16000.f32"
    orphan.write_bytes(np.zeros(32000, np.float32).tobytes())
    monkeypatch.setattr(spool, "spool_dir", lambda: tmp_path)
    monkeypatch.setattr(spool, "_alive", lambda pid: False)
    saved, told = [], []
    monkeypatch.setattr(history, "append", lambda text, **kw: saved.append((text, kw)))
    monkeypatch.setattr(feedback, "notify", lambda title, body, urgent=False: told.append(body))
    vio = app.VoiceIO.__new__(app.VoiceIO)
    vio.recorder = SimpleNamespace(sample_rate=16000)
    vio.engine = MagicMock()
    vio.engine.transcribe.return_value = SimpleNamespace(text="the words that were cut off", raw="r")
    vio.cfg = SimpleNamespace(history=None)
    vio._recover_orphans()
    assert saved[0][0] == "the words that were cut off"
    assert saved[0][1]["extra"] == {"recovered": True}
    assert "paste-last" in told[0]
    assert not orphan.exists()
