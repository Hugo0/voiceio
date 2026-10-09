"""Score a decoder on the user's held-out clips.

Decodes each test clip the way dictation does (final quality, the user's
vocabulary bias) and compares to the reference label: WER, vocabulary-term
recall (the number personalization is meant to move), hallucinations,
repetition loops, and speed. Results are saved under learn/runs/ so models
can be compared over time.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voiceio.core.decoders.base import Bias, Decoder
from voiceio.learn import metrics
from voiceio.learn.store import learn_dir, read_split
from voiceio.learn.teacher import load_wav


@dataclass
class EvalResult:
    decoder: str
    dataset: str
    clips: int = 0
    audio_secs: float = 0.0
    decode_secs: float = 0.0
    ref_words: int = 0
    errors: int = 0
    terms_spoken: int = 0
    terms_recalled: int = 0
    repetitions: int = 0
    failures: int = 0
    per_clip: list[dict] = field(default_factory=list)

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else 0.0

    @property
    def term_recall(self) -> float:
        return self.terms_recalled / self.terms_spoken if self.terms_spoken else 0.0

    @property
    def rtf(self) -> float:
        return self.audio_secs / self.decode_secs if self.decode_secs else 0.0

    def summary(self) -> str:
        return (f"{self.decoder}: WER {self.wer:.1%}  vocab recall {self.term_recall:.1%} "
                f"({self.terms_recalled}/{self.terms_spoken})  loops {self.repetitions}  "
                f"failures {self.failures}  {self.rtf:.1f}x realtime  [{self.clips} clips]")


def evaluate(decoder: Decoder, dataset: Path, *, bias: Bias = Bias(),
             terms: list[str] = (), limit: int | None = None) -> EvalResult:
    rows = read_split(dataset, "test")[:limit]
    res = EvalResult(decoder=decoder.name, dataset=dataset.name)
    for row in rows:
        audio = load_wav(Path(row["file_name"]))
        t0 = time.monotonic()
        try:
            hyp = decoder.decode(audio, final=True, bias=bias).text
        except Exception:  # noqa: BLE001 — a crash is a result, not an abort
            res.failures += 1
            continue
        dt = time.monotonic() - t0
        a = metrics.align(row["text"], hyp)
        spoken, recalled = metrics.term_hits(row["text"], hyp, list(terms))
        res.clips += 1
        res.audio_secs += len(audio) / 16000
        res.decode_secs += dt
        res.ref_words += a.ref_words
        res.errors += a.errors
        res.terms_spoken += spoken
        res.terms_recalled += recalled
        res.repetitions += metrics.count_repetitions(hyp)
        res.per_clip.append({"file": Path(row["file_name"]).name, "wer": round(a.wer, 4),
                             "terms": [spoken, recalled], "secs": round(dt, 2)})
    return res


def save(result: EvalResult) -> Path:
    runs = learn_dir() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = asdict(result) | {"wer": result.wer, "term_recall": result.term_recall,
                             "rtf": result.rtf}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
