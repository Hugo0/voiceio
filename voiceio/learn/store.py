"""The personalization corpus: what was said, what was heard, what is right.

Capture is the existing history + retained recordings (each committed
utterance keeps its audio, the decoder's raw text and the final text). This
module adds the rest under ``$XDG_DATA_HOME/voiceio/learn``:

    labels.jsonl          reference transcripts, newest wins per clip
                          {"audio", "segments": [{"start", "end", "text"}],
                           "source": "teacher:<name>"|"human", "ts"}
    datasets/vNNNN/       frozen train/dev/test splits + manifest.json
    models/<run>/         fine-tuned models (see train.py)
    runs/                 evaluation results

The test split is assigned by a hash of the clip name, so a clip lands in the
same split in every dataset version and is never trained on.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from voiceio import config

SCHEMA = 1
SPLITS = ("train", "dev", "test")


def learn_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".local" / "share"
    return root / "voiceio" / "learn"


def labels_path() -> Path:
    return learn_dir() / "labels.jsonl"


@dataclass(frozen=True)
class Clip:
    """One retained utterance."""
    audio: str              # file name under config.RECORDINGS_DIR
    ts: float
    duration: float
    raw: str                # decoder output (before the text pipeline)
    final: str              # what was committed
    source: str = "desktop"

    @property
    def path(self) -> Path:
        return config.RECORDINGS_DIR / self.audio


def _strip_prefix(text: str) -> str:
    # Older history rows kept the voice prefix in the text.
    return text[len("[voice]"):].lstrip() if text.startswith("[voice]") else text


def clips() -> Iterator[Clip]:
    """Every history entry whose recording still exists."""
    path = config.HISTORY_PATH
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("audio")
            if not name or not (config.RECORDINGS_DIR / name).exists():
                continue
            final = _strip_prefix(row.get("text", ""))
            yield Clip(
                audio=name, ts=row.get("ts", 0.0), duration=row.get("duration") or 0.0,
                raw=_strip_prefix(row.get("raw") or final), final=final,
                source=row.get("source", "desktop"),
            )


# ── labels ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Label:
    """A reference transcript, kept as timed segments: Whisper trains on
    windows of at most 30s, and review edits stay aligned with the audio."""
    audio: str
    segments: tuple[Segment, ...]
    source: str             # "teacher:<model>" or "human"
    ts: float

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def reviewed(self) -> bool:
        return self.source == "human"

    def to_json(self) -> str:
        return json.dumps({"audio": self.audio, "source": self.source, "ts": self.ts,
                           "segments": [asdict(s) for s in self.segments]},
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Label:
        d = json.loads(line)
        return cls(d["audio"], tuple(Segment(**s) for s in d["segments"]),
                   d["source"], d["ts"])


def load_labels() -> dict[str, Label]:
    """Latest label per clip; a human label always beats a teacher label."""
    out: dict[str, Label] = {}
    path = labels_path()
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                lab = Label.from_json(line)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            prev = out.get(lab.audio)
            if prev is None or lab.reviewed or not prev.reviewed:
                out[lab.audio] = lab
    return out


def add_label(audio: str, segments: list[Segment] | tuple[Segment, ...],
              source: str) -> Label:
    lab = Label(audio=audio, segments=tuple(segments), source=source, ts=time.time())
    path = labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config._chmod(path.parent, config._SECURE_DIR)
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        f.write(lab.to_json() + "\n")
    if new:
        config._chmod(path, config._SECURE_FILE)
    return lab


# ── datasets ─────────────────────────────────────────────────────────────

def split_of(audio: str, test_frac: float = 0.1, dev_frac: float = 0.1) -> str:
    """Deterministic split by clip name: stable across dataset versions."""
    h = int(hashlib.sha256(audio.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + dev_frac:
        return "dev"
    return "train"


# Whisper's encoder sees 30s at most; leave room for segment-edge slop.
MAX_WINDOW_SECS = 28.0


def windows(segments: tuple[Segment, ...], max_secs: float = MAX_WINDOW_SECS) -> list[Segment]:
    """Group consecutive segments into training windows of at most `max_secs`.
    A single over-long segment is dropped (its text can't be split safely)."""
    out: list[Segment] = []
    cur: list[Segment] = []
    for seg in segments:
        if seg.end - seg.start > max_secs:
            continue
        if cur and seg.end - cur[0].start > max_secs:
            out.append(Segment(cur[0].start, cur[-1].end, " ".join(s.text.strip() for s in cur)))
            cur = []
        cur.append(seg)
    if cur:
        out.append(Segment(cur[0].start, cur[-1].end, " ".join(s.text.strip() for s in cur)))
    return [w for w in out if w.text.strip()]


def build_dataset(*, reviewed_only: bool = False) -> Path:
    """Freeze labeled clips into a new dataset version. Returns its dir.

    train/dev rows are ≤28s windows {"file_name", "start", "end", "text"} cut
    along label segments; test rows are whole clips (evaluation decodes the
    way dictation does). Audio is referenced by absolute path, never copied.
    """
    labels = load_labels()
    rows = {s: [] for s in SPLITS}
    for clip in clips():
        lab = labels.get(clip.audio)
        if lab is None or (reviewed_only and not lab.reviewed):
            continue
        split = split_of(clip.audio)
        base = {"file_name": str(clip.path), "label_source": lab.source, "source": clip.source}
        if split == "test":
            rows[split].append({**base, "start": 0.0, "end": clip.duration,
                                "text": lab.text, "raw": clip.raw})
            continue
        for w in windows(lab.segments):
            rows[split].append({**base, "start": w.start, "end": w.end, "text": w.text})
    root = learn_dir() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in root.glob("v[0-9]*"))
    version = f"v{int(existing[-1][1:]) + 1:04d}" if existing else "v0001"
    out = root / version
    out.mkdir()
    for split, items in rows.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
    manifest = {
        "schema": SCHEMA, "version": version, "created": time.time(),
        "reviewed_only": reviewed_only,
        "counts": {s: len(v) for s, v in rows.items()},
        "hours": {s: round(sum(r["end"] - r["start"] for r in v) / 3600, 2)
                  for s, v in rows.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def latest_dataset() -> Path | None:
    root = learn_dir() / "datasets"
    versions = sorted(root.glob("v[0-9]*")) if root.exists() else []
    return versions[-1] if versions else None


def read_split(dataset: Path, split: str) -> list[dict]:
    path = dataset / f"{split}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
