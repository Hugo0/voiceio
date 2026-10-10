"""Import audio recorded elsewhere (phone voice notes, another machine).

Files are decoded with ffmpeg to 16 kHz mono WAV under the learn dir (0700/
0600, never uploaded) and listed in imports.jsonl; from there they flow
through label → review → build like retained dictation. Content-addressed, so
re-importing the same folder adds nothing. Only import your own voice: the
model learns whoever is speaking.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from voiceio import config
from voiceio.core.decoders.base import SAMPLE_RATE
from voiceio.learn.store import IMPORT_PREFIX, _rows, imports_path, learn_dir

AUDIO_SUFFIXES = frozenset({
    ".wav", ".ogg", ".oga", ".opus", ".m4a", ".mp3", ".webm", ".flac", ".aac", ".amr",
})
MIN_SECS = 0.5
MAX_SECS = 30 * 60


@dataclass
class ImportReport:
    added: int = 0
    duplicate: int = 0
    rejected: int = 0
    seconds: float = 0.0


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _decode(src: Path, dest: Path) -> float | None:
    """ffmpeg → 16 kHz mono s16 WAV. Returns duration, or None if unreadable."""
    tmp = dest.with_suffix(".part.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True, timeout=300)
        with wave.open(str(tmp)) as w:
            secs = w.getnframes() / w.getframerate()
    except (subprocess.SubprocessError, OSError, wave.Error):
        tmp.unlink(missing_ok=True)
        return None
    if not MIN_SECS <= secs <= MAX_SECS:
        tmp.unlink(missing_ok=True)
        return None
    config._chmod(tmp, config._SECURE_FILE)
    tmp.rename(dest)
    return secs


def import_dir(src: Path, source: str) -> ImportReport:
    """Import every audio file under `src`, tagged with `source`."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to import audio")
    out_dir = learn_dir() / IMPORT_PREFIX / source
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in (learn_dir(), out_dir.parent, out_dir):
        config._chmod(d, config._SECURE_DIR)
    have = {row["audio"] for row in _rows(imports_path()) if "audio" in row}
    report = ImportReport()
    files = sorted(p for p in Path(src).expanduser().rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    rows = []
    for path in files:
        name = f"{IMPORT_PREFIX}{source}/{_digest(path)}.wav"
        if name in have:
            report.duplicate += 1
            continue
        secs = _decode(path, learn_dir() / name)
        if secs is None:
            report.rejected += 1
            continue
        have.add(name)
        rows.append({"audio": name, "source": source, "duration": round(secs, 2),
                     "ts": path.stat().st_mtime, "imported": time.time()})
        report.added += 1
        report.seconds += secs
    if rows:
        new = not imports_path().exists()
        with open(imports_path(), "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        if new:
            config._chmod(imports_path(), config._SECURE_FILE)
    return report
