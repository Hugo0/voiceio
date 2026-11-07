#!/usr/bin/env python3
"""Find LibriSpeech test-clean words where a plain-word hotword cannot help a
NeMo model but the same word given as the model's own BPE pieces can.

For k2-fsa/sherpa-onnx#4088. Steps:
  1. decode test-clean with the released sherpa-onnx wheel (greedy, no hotwords)
  2. keep reference words the model got wrong whose split by bpe.vocab (what
     sherpa-onnx does for a plain-word hotword) differs from the model's own
     SentencePiece split
  3. decode those utterances with the CLI built from the PR: no hotwords, plain
     words, and the same words as pieces (one hotwords file holding all of them)
"""
import argparse
import glob
import json
import os
import re
import subprocess

import numpy as np
import sentencepiece as spm
import soundfile as sf


def ssp_encode(word, pieces):
    """Python copy of simple-sentencepiece's Encode (max total score)."""
    s = "▁" + word
    n = len(s)
    route = [(0.0, 0, None)] * (n + 1)
    for i in range(n - 1, -1, -1):
        best, end, piece = float("-inf"), -1, None
        for j in range(i + 1, n + 1):
            p = s[i:j]
            if p in pieces:
                sc = pieces[p] + route[j][0]
                if sc > best or (sc == best and end >= j):
                    best, end, piece = sc, j, p
        route[i] = (0.0 if best == float("-inf") else best, end, piece)
    out, i = [], 0
    while i < n:
        if route[i][1] == -1:
            out.append("<unk>")
            i += 1
        else:
            out.append(route[i][2])
            i = route[i][1]
    return out


def norm(text):
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def align(ref, hyp):
    """Return reference words that were substituted or deleted."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    bad, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                bad.append(ref[i - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            bad.append(ref[i - 1])
            i -= 1
        else:
            j -= 1
    return bad


def cli(exe, model, wavs, extra):
    cmd = [exe, f"--tokens={model}/tokens.txt",
           f"--encoder={model}/encoder.int8.onnx",
           f"--decoder={model}/decoder.int8.onnx",
           f"--joiner={model}/joiner.int8.onnx",
           "--model-type=nemo_transducer",
           "--decoding-method=modified_beam_search"] + extra + wavs
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert len(lines) == len(wavs), p.stderr[-2000:]
    return [json.loads(ln)["text"].strip() for ln in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sp", required=True)
    ap.add_argument("--bpe-vocab", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--score", default="2.0")
    ap.add_argument("--out", default="cands")
    args = ap.parse_args()

    import sherpa_onnx as so

    sp = spm.SentencePieceProcessor(model_file=args.sp)
    pieces = {}
    for ln in open(args.bpe_vocab, encoding="utf-8"):
        t, s = ln.rstrip("\n").split("\t")
        pieces[t] = float(s)

    refs = {}
    for f in glob.glob(f"{args.librispeech}/*/*/*.trans.txt"):
        for ln in open(f):
            k, t = ln.strip().split(" ", 1)
            refs[k] = t
    keys = sorted(refs)

    rec = so.OfflineRecognizer.from_transducer(
        tokens=f"{args.model}/tokens.txt",
        encoder=f"{args.model}/encoder.int8.onnx",
        decoder=f"{args.model}/decoder.int8.onnx",
        joiner=f"{args.model}/joiner.int8.onnx",
        model_type="nemo_transducer", num_threads=4)

    os.makedirs(args.out, exist_ok=True)
    cands = []
    for b in range(0, len(keys), 16):
        batch = keys[b:b + 16]
        ss = []
        for k in batch:
            spk, ch, _ = k.split("-")
            a, r = sf.read(f"{args.librispeech}/{spk}/{ch}/{k}.flac", dtype="float32")
            s = rec.create_stream()
            s.accept_waveform(r, a)
            ss.append((k, s, a, r))
        rec.decode_streams([s for _, s, _, _ in ss])
        for k, s, a, r in ss:
            bad = align(norm(refs[k]), norm(s.result.text))
            for w in set(bad):
                for form in (w.capitalize(), w.lower()):
                    model_split = sp.encode(form, out_type=str)
                    vocab_split = ssp_encode(form, pieces)
                    if model_split != vocab_split:
                        wav = f"{args.out}/{k}.wav"
                        sf.write(wav, a, r, subtype="PCM_16")
                        cands.append(dict(key=k, wav=wav, word=w, form=form,
                                          model=model_split, vocab=vocab_split,
                                          ref=refs[k], greedy=s.result.text.strip()))
                        break
    print(f"{len(cands)} candidate words in {len({c['key'] for c in cands})} utterances")
    for c in cands:
        print(f"{c['key']} {c['form']!r}: model {c['model']} vocab {c['vocab']}")

    plain = sorted({c["form"] for c in cands})
    pretok = sorted({" ".join(c["model"]) for c in cands})
    open("plain.txt", "w").write("\n".join(plain) + "\n")
    open("pretok.txt", "w").write("\n".join(pretok) + "\n")

    wavs = sorted({c["wav"] for c in cands})
    flags = [f"--modeling-unit=bpe", f"--bpe-vocab={args.bpe_vocab}",
             f"--hotwords-score={args.score}"]
    res = {"none": [], "plain": [], "pretok": []}
    for b in range(0, len(wavs), 8):
        chunk = wavs[b:b + 8]
        res["none"] += cli(args.exe, args.model, chunk, [])
        res["plain"] += cli(args.exe, args.model, chunk, flags + ["--hotwords-file=plain.txt"])
        res["pretok"] += cli(args.exe, args.model, chunk, flags + ["--hotwords-file=pretok.txt"])
    by_wav = {w: {m: res[m][i] for m in res} for i, w in enumerate(wavs)}

    hits = []
    for c in cands:
        r = by_wav[c["wav"]]
        ok = {m: c["word"] in norm(r[m]) for m in r}
        tag = "PRETOK-ONLY" if ok["pretok"] and not ok["plain"] and not ok["none"] else ""
        print(f"\n{c['key']} {c['form']!r} {tag}\n  ref   : {c['ref']}")
        for m in r:
            print(f"  {m:6}: {'+' if ok[m] else '-'} {r[m]}")
        if tag:
            hits.append(c)
    print(f"\n{len(hits)} words fixed only by the pretokenized form:")
    for c in hits:
        print(f"  {c['key']} {c['form']!r} model {c['model']} vocab {c['vocab']}")
    json.dump(hits, open("hits.json", "w"), indent=1)


if __name__ == "__main__":
    main()
