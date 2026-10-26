#!/usr/bin/env python3
"""Reproduce modified_beam_search dropping speech for NeMo TDT models.

Builds 20 clips of 24-30 s from LibriSpeech test-clean, each a few utterances
joined by 0.8-2.5 s of low-level noise (-55 dBFS), like pauses in dictation,
and compares greedy_search and modified_beam_search WER against the
LibriSpeech transcripts. Also scores the original single utterances.

    python3 repro_tdt_mbs.py --librispeech LibriSpeech/test-clean \\
        --model sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 [--length-reward 0.5]
"""
import argparse
import glob
import os
import random
import re

import numpy as np
import sherpa_onnx
import soundfile as sf


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


def errors(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1], len(r)


def clips(root):
    files = sorted(glob.glob(f"{root}/*/*/*.flac"))
    refs = {}
    for t in glob.glob(f"{root}/*/*/*.trans.txt"):
        for line in open(t):
            k, v = line.rstrip("\n").split(" ", 1)
            refs[k] = v
    key = lambda f: os.path.basename(f)[:-5]  # noqa: E731
    rng = np.random.default_rng(1)
    random.seed(1)
    concat = []
    for _ in range(20):
        parts, ref, total = [], [], 0
        while total < 16000 * 24:
            f = random.choice(files)
            a, _ = sf.read(f, dtype="float32")
            if len(a) > 16000 * 12:
                continue
            gap = rng.normal(0, 10 ** (-55 / 20), int(16000 * random.uniform(0.8, 2.5)))
            parts += [a, gap.astype("float32")]
            ref.append(refs[key(f)])
            total += len(a) + len(gap)
        concat.append((np.concatenate(parts), " ".join(ref)))
    random.seed(0)
    single = []
    for f in random.sample(files, 20):
        a, _ = sf.read(f, dtype="float32")
        single.append((a, refs[key(f)]))
    return concat, single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--length-reward", type=float, default=None)
    ap.add_argument("--num-threads", type=int, default=4)
    a = ap.parse_args()
    concat, single = clips(a.librispeech)
    f = lambda n: sorted(glob.glob(f"{a.model}/{n}*.onnx"))[0]  # noqa: E731
    for method in ["greedy_search", "modified_beam_search"]:
        kw = dict(encoder=f("encoder"), decoder=f("decoder"), joiner=f("joiner"),
                  tokens=f"{a.model}/tokens.txt", num_threads=a.num_threads,
                  model_type="nemo_transducer", decoding_method=method)
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
        for name, data in [("concatenated", concat), ("single", single)]:
            e = n = dropped = empty = 0
            for audio, ref in data:
                s = rec.create_stream()
                if method == "modified_beam_search" and a.length_reward is not None:
                    s.set_option("length_reward", str(a.length_reward))  # per stream
                s.accept_waveform(16000, audio)
                rec.decode_stream(s)
                hyp = s.result.text
                ce, cn = errors(ref, hyp)
                e, n = e + ce, n + cn
                dropped += len(words(hyp)) < 0.8 * cn
                empty += not hyp.strip()
            print(f"{method:21s} {name:13s} WER {e / n:6.2%}  "
                  f"clips with >20% of words missing: {dropped}/{len(data)}  empty: {empty}")


if __name__ == "__main__":
    main()
