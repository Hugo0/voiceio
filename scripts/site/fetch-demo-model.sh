#!/usr/bin/env bash
# Puts the landing-page demo's speech model into site/models/ before
# `npx wrangler deploy`. Not committed (≈137 MB): Workers static assets cap a
# file at 25 MiB, so the encoder is split into parts that
# site/parakeet-worker.js joins again.
# Without the model the demo falls back to Moonshine (site/asr-worker.js).
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME=sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8
URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$NAME.tar.bz2
SHA256=f628312e9fdf8686374cb01a69425c41732529d540860311f16f37cbc32cfe9b
OUT=site/models/parakeet-tdt-110m-int8
TMP=${TMPDIR:-/tmp}/voiceio-demo-model
mkdir -p "$TMP"

if ! echo "$SHA256  $TMP/$NAME.tar.bz2" | sha256sum -c --status 2>/dev/null; then
  curl -fL --retry 3 -o "$TMP/$NAME.tar.bz2" "$URL"
  echo "$SHA256  $TMP/$NAME.tar.bz2" | sha256sum -c
fi
rm -rf "$TMP/x" "$OUT"
mkdir -p "$TMP/x" "$OUT"
tar -xjf "$TMP/$NAME.tar.bz2" -C "$TMP/x"
find "$TMP/x" -type f | sed "s#$TMP/x/##"

python3 - "$TMP/x" "$OUT" <<'PY'
import glob, json, os, sys
src, out = sys.argv[1:]
PART = 24_000_000  # under the 25 MiB per-file cap

def split(path, dest, name):
    size = os.path.getsize(path)
    parts = []
    with open(path, "rb") as f:
        i = 0
        while chunk := f.read(PART):
            pname = name if size <= PART else f"{name}.{i:02d}"
            with open(os.path.join(dest, pname), "wb") as o:
                o.write(chunk)
            parts.append({"name": pname, "size": len(chunk)})
            i += 1
    return {"name": name, "size": size, "parts": parts}

def find(pattern):
    hits = sorted(glob.glob(os.path.join(src, "**", pattern), recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

files = [split(find(pat), out, name) for pat, name in
         (("encoder*.onnx", "encoder.onnx"), ("decoder*.onnx", "decoder.onnx"), ("joiner*.onnx", "joiner.onnx"), ("tokens.txt", "tokens.txt"))]
json.dump({"files": files}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("model:", sum(f["size"] for f in files), "bytes in", sum(len(f["parts"]) for f in files), "files ->", out)
PY
