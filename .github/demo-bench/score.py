"""Scores bench.mjs output: WER (Whisper English normalizer on both sides) and
a markdown summary. usage: score.py <results.json> <refs.json>"""
import json
import sys

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

norm = EnglishTextNormalizer()
res = json.load(open(sys.argv[1]))
refs = {r["name"]: r for r in json.load(open(sys.argv[2]))}


def wer(decodes, group):
    pairs = [(norm(refs[d["name"]]["text"]), norm(d["text"])) for d in decodes if d["group"] == group]
    pairs = [(r, h) for r, h in pairs if r]
    return (jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) * 100, len(pairs)) if pairs else (None, 0)


mb = lambda b: f"{b / 1e6:.1f}"
out = [f"Runner: {res['cpu']} x{res['nproc']}, {res['chrome']}", ""]
out.append("| model | cpu | net | cold ready s | download s | bytes MB | warm ready s | t1 | t5 | t15 | t30 (ms) | WER utt | WER concat | peak PSS MB | error |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in res["runs"]:
    c = r.get("cold") or {}
    d = {x["name"]: x for x in r.get("decodes", [])}
    t = lambda n: f"{d[n]['ms']:.0f}" if n in d else ""
    wu, nu = wer(r.get("decodes", []), "utt")
    wc, nc = wer(r.get("decodes", []), "concat")
    f = lambda v, n: f"{v:.2f} (n={n})" if v is not None else ""
    out.append("| " + " | ".join([
        r["model"], f"{r['cpu']}x", r["net"],
        f"{c['readyMs'] / 1000:.1f}" if c else "", f"{c['downloadMs'] / 1000:.1f}" if c else "",
        f"{mb(c.get('bytes', 0))} (decl {mb(c.get('declared', 0))})" if c else "",
        f"{r['warm']['readyMs'] / 1000:.1f}" if r.get("warm") else "",
        t("t01"), t("t05"), t("t15"), t("t30"), f(wu, nu), f(wc, nc),
        f"{(r['peakPss'] - r['basePss']) / 1e6:.0f}", (r.get("error") or r.get("cpuError") or r.get("netError") or "")[:80],
    ]) + " |")

out.append("\n### Sample outputs (first full run per model)\n")
seen = set()
for r in res["runs"]:
    if r["model"] in seen or not r.get("decodes") or r["net"] != "none" or r["cpu"] != 1:
        continue
    seen.add(r["model"])
    out.append(f"**{r['model']}** (jsMemory: {r.get('jsMemory')})\n")
    for x in r["decodes"]:
        if x["group"] in ("names", "concat") or x["name"] in ("t05", "t15"):
            out.append(f"- `{x['name']}` ({x['secs']} s, {x['ms']:.0f} ms): {x['text']}")
            if refs[x["name"]]["text"]:
                out.append(f"  - ref: {refs[x['name']]['text'].lower()}")
print("\n".join(out))
