"""Experimental smaller encoders for the demo (download size is what a first
visit waits on). From the fp32 Parakeet-TDT-110M encoder:
  q4    MatMul weights to 4-bit blocks (MatMulNBits), everything else fp32
  q4c8  q4, plus Conv weights to int8 (ConvInteger)
Writes <out>/<variant>/ like scripts/site/fetch-demo-model.sh does, reusing
the int8 decoder/joiner/tokens from <int8 dir>.

usage: quantize.py <fp32 encoder.onnx> <int8 model dir> <out models dir>
"""
import json
import os
import shutil
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

src, int8_dir, out = sys.argv[1:4]
PART = 24_000_000
meta = {p.key: p.value for p in onnx.load(src, load_external_data=False).metadata_props}


def with_meta(path):
    m = onnx.load(path)
    have = {p.key for p in m.metadata_props}
    for k, v in meta.items():
        if k not in have:
            m.metadata_props.add(key=k, value=v)
    onnx.save(m, path)


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


def package(variant, encoder):
    d = os.path.join(out, variant)
    os.makedirs(d, exist_ok=True)
    files = [split(encoder, d, "encoder.onnx")]
    for f in json.load(open(os.path.join(int8_dir, "manifest.json")))["files"][1:]:
        for p in f["parts"]:
            shutil.copy(os.path.join(int8_dir, p["name"]), d)
        files.append(f)
    json.dump({"files": files}, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(variant, "encoder", os.path.getsize(encoder), "total", sum(f["size"] for f in files))


q = MatMulNBitsQuantizer(onnx.load(src), block_size=32, is_symmetric=True)
q.process()
q.model.save_model_to_file("q4.onnx", use_external_data_format=False)
with_meta("q4.onnx")
package("parakeet-q4", "q4.onnx")

quantize_dynamic("q4.onnx", "q4c8.onnx", op_types_to_quantize=["Conv"], weight_type=QuantType.QUInt8)
with_meta("q4c8.onnx")
package("parakeet-q4c8", "q4c8.onnx")
