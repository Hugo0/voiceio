"""Score a decoder on the user's held-out clips.

Compares each test clip's decode to its reference label: WER, vocabulary-term
recall (the number personalization is meant to move), repetition loops,
failures and speed. Two modes:

    offline    one final-quality decode per clip — fast, isolates the model
    streaming  replays the clip in real time through StreamingSession exactly
               as dictation runs it (interims, freeze cuts, tail decode at
               stop) and also measures stop→text, the wait a user feels

Results are saved under learn/runs/ so models can be compared over time.
"""
from __future__ import annotations

import json
import statistics
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
    gold: int = 0           # clips whose reference a human verified
    mode: str = "offline"
    waits: list[float] = field(default_factory=list)   # streaming: stop→text
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

    def wait_summary(self) -> str:
        if not self.waits:
            return ""
        p90 = statistics.quantiles(self.waits, n=10)[-1] if len(self.waits) > 1 else self.waits[0]
        return (f"stop→text median {statistics.median(self.waits):.2f}s  p90 {p90:.2f}s  "
                f"max {max(self.waits):.2f}s")

    def summary(self) -> str:
        wait = f"\n  {self.wait_summary()}" if self.waits else ""
        return self._scores() + wait

    def _scores(self) -> str:
        return (f"{self.decoder}: WER {self.wer:.1%}  vocab recall {self.term_recall:.1%} "
                f"({self.terms_recalled}/{self.terms_spoken})  loops {self.repetitions}  "
                f"failures {self.failures}  {self.rtf:.1f}x realtime  "
                f"[{self.clips} clips, {self.gold} human-verified]")


def evaluate(decoder: Decoder, dataset: Path, *, bias: Bias = Bias(),
             terms: list[str] = (), limit: int | None = None,
             streaming: bool = False, freeze_secs: float = 8.0,
             speed: float = 1.0) -> EvalResult:
    rows = read_split(dataset, "test")[:limit]
    res = EvalResult(decoder=decoder.name, dataset=dataset.name,
                     mode="streaming" if streaming else "offline")
    for row in rows:
        audio = load_wav(Path(row["file_name"]))
        t0 = time.monotonic()
        try:
            if streaming:
                hyp, wait = _replay(decoder, audio, bias, freeze_secs, speed)
                res.waits.append(wait)
            else:
                hyp = decoder.decode(audio, final=True, bias=bias).text
        except Exception:  # noqa: BLE001 — a crash is a result, not an abort
            res.failures += 1
            continue
        dt = time.monotonic() - t0
        a = metrics.align(row["text"], hyp)
        spoken, recalled = metrics.term_hits(row["text"], hyp, list(terms))
        res.clips += 1
        res.gold += row.get("label_source") == "human"
        res.audio_secs += len(audio) / 16000
        res.decode_secs += dt
        res.ref_words += a.ref_words
        res.errors += a.errors
        res.terms_spoken += spoken
        res.terms_recalled += recalled
        res.repetitions += metrics.count_repetitions(hyp)
        clip = {"file": Path(row["file_name"]).name, "wer": round(a.wer, 4),
                "terms": [spoken, recalled], "secs": round(dt, 2)}
        if streaming:
            clip["wait"] = round(res.waits[-1], 2)
        res.per_clip.append(clip)
    return res


_CHUNK = 1600  # 100 ms, the size real capture delivers


def _replay(decoder: Decoder, audio, bias: Bias, freeze_secs: float,
            speed: float) -> tuple[str, float]:
    """Feed `audio` through a StreamingSession as if spoken; press stop at the
    end. Returns (final raw text, seconds from stop to final text)."""
    from voiceio.audio_source import PushAudioSource
    from voiceio.core.pipeline import Pipeline
    from voiceio.streaming import StreamingSession
    from voiceio.typers.collecting import CollectingTyper

    source = PushAudioSource(16000)
    session = StreamingSession(decoder, CollectingTyper(), source, pipeline=Pipeline([]),
                               freeze_secs=freeze_secs, bias=bias)
    session.start()
    for i in range(0, len(audio), _CHUNK):
        source.feed(audio[i:i + _CHUNK])
        time.sleep(_CHUNK / 16000 / speed)
    source.close()
    t0 = time.monotonic()
    text = session.stop(source.get_audio_so_far())
    return text, time.monotonic() - t0


def save(result: EvalResult) -> Path:
    runs = learn_dir() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = asdict(result) | {"wer": result.wer, "term_recall": result.term_recall,
                             "rtf": result.rtf}
    if result.waits:
        data["stop_to_text_median"] = statistics.median(result.waits)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def latest(model: str, dataset: str, mode: str = "offline") -> dict | None:
    """The newest saved run of `model` (a faster-whisper name or path) on
    `dataset`, or None."""
    runs = learn_dir() / "runs"
    want = f"faster-whisper:{model}"
    for path in sorted(runs.glob("eval-*.json"), reverse=True) if runs.exists() else []:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (data.get("decoder", "").removesuffix("+draft") == want
                and data.get("dataset") == dataset and data.get("mode", "offline") == mode):
            return data
    return None


# A fine-tune may trade a hair of WER for vocabulary; it may not regress either.
WER_SLACK = 0.002


def promotion_verdict(candidate: dict, current: dict) -> tuple[bool, str]:
    """Is `candidate` at least as good as `current` on the same clips?"""
    rows = [("WER", current["wer"], candidate["wer"], "{:.1%}"),
            ("vocab recall", current["term_recall"], candidate["term_recall"], "{:.1%}"),
            ("loops", current["repetitions"], candidate["repetitions"], "{}")]
    table = "\n".join(f"  {name:<13} {fmt.format(a):>8} → {fmt.format(b):<8}"
                      for name, a, b, fmt in rows)
    ok = (candidate["wer"] <= current["wer"] + WER_SLACK
          and candidate["term_recall"] >= current["term_recall"]
          and candidate["repetitions"] <= current["repetitions"])
    return ok, table
