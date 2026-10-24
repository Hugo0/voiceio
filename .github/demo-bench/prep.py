"""Builds the benchmark audio from public data only: LibriSpeech test-clean and
the sherpa-onnx model's test_wavs. Writes raw float32 16 kHz files plus
refs.json into <out>/bench-audio/.

usage: prep.py <LibriSpeech/test-clean dir> <test_wavs dir> <out dir>
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
PAUSE = np.zeros(int(0.4 * RATE), dtype=np.float32)  # a dictation-like breath between sentences

ls_dir, wavs_dir, out = (Path(a) for a in sys.argv[1:4])
dst = out / "bench-audio"
dst.mkdir(parents=True, exist_ok=True)

utts = []
for trans in sorted(ls_dir.glob("*/*/*.trans.txt")):
    for line in trans.read_text().splitlines():
        uid, text = line.split(" ", 1)
        path = trans.parent / f"{uid}.flac"
        utts.append({"id": uid, "path": path, "text": text, "secs": sf.info(path).duration, "chapter": trans.parent.name})


def audio(u):
    a, sr = sf.read(u["path"], dtype="float32")
    assert sr == RATE
    return a


refs = []


def write(name, a, text, group):
    (dst / f"{name}.f32").write_bytes(a.astype(np.float32).tobytes())
    refs.append({"name": name, "url": f"bench-audio/{name}.f32", "text": text, "secs": round(len(a) / RATE, 2), "group": group})


rng = random.Random(0)
for u in rng.sample([u for u in utts if 3 <= u["secs"] <= 20], 20):
    write(f"utt-{u['id']}", audio(u), u["text"], "utt")

by_chapter = defaultdict(list)
for u in utts:
    by_chapter[u["chapter"]].append(u)
chapters = sorted(by_chapter)
rng.shuffle(chapters)
made = 0
for ch in chapters:
    parts, texts, secs = [], [], 0.0
    for u in sorted(by_chapter[ch], key=lambda u: u["id"]):
        if secs + u["secs"] > 23:
            continue
        parts += [audio(u), PAUSE]
        texts.append(u["text"])
        secs += u["secs"] + 0.4
        if secs >= 18:
            break
    if 18 <= secs <= 24:
        write(f"cat-{ch}", np.concatenate(parts[:-1]), " ".join(texts), "concat")
        made += 1
    if made == 10:
        break

# Timing clips: ~1 s (time to first preview), 5 s, 15 s, 30 s.
near = lambda s: min(utts, key=lambda u: abs(u["secs"] - s))
u5, u15 = near(5), near(15)
write("t01", audio(u5)[:RATE], "", "timing")
write("t05", audio(u5), u5["text"], "timing")
write("t15", audio(u15), u15["text"], "timing")
long_parts, long_text = [], []
for u in sorted(by_chapter[u15["chapter"]], key=lambda u: u["id"]):
    long_parts += [audio(u), PAUSE]
    long_text.append(u["text"])
    if sum(len(p) for p in long_parts) >= 30 * RATE:
        break
write("t30", np.concatenate(long_parts)[: 30 * RATE], "", "timing")

for w in sorted(wavs_dir.glob("*.wav")):
    a, sr = sf.read(w, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != RATE:
        a = np.interp(np.arange(0, len(a), sr / RATE), np.arange(len(a)), a).astype(np.float32)
    write(f"wav-{w.stem}", a, "", "names")

(dst / "refs.json").write_text(json.dumps(refs, indent=1))
print(len(refs), "clips;", sum(r["secs"] for r in refs), "s")
