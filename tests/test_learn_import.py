import shutil
import stat
import subprocess

import pytest

from voiceio.learn import importer, review, store

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _tone(path, secs, fmt=()):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={secs}", *fmt, str(path)], check=True)


def test_import_converts_dedupes_and_feeds_the_pipeline(tmp_path):
    src = tmp_path / "phone"
    (src / "nested").mkdir(parents=True)
    _tone(src / "note.ogg", 2)
    _tone(src / "nested" / "memo.wav", 3)
    _tone(src / "blip.wav", 0.2)                 # too short
    (src / "photo.jpg").write_bytes(b"\xff\xd8")  # not audio

    r = importer.import_dir(src, "phone")
    assert (r.added, r.rejected) == (2, 1)
    assert importer.import_dir(src, "phone").duplicate == 2  # content-addressed

    clips = [c for c in store.clips() if c.source == "phone"]
    assert len(clips) == 2 and all(c.path.exists() for c in clips)
    assert all(c.path.is_relative_to(store.learn_dir()) for c in clips)
    assert stat.S_IMODE(clips[0].path.stat().st_mode) == 0o600
    assert stat.S_IMODE(clips[0].path.parent.stat().st_mode) == 0o700

    # No live decode to disagree with: only vocabulary segments get queued.
    lab = store.add_label(clips[0].audio, [store.Segment(0, 2, "plain words")], "teacher:x")
    assert review.queue(clips, {lab.audio: lab}, ["Kalshi"]) == []
