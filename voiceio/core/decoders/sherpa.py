"""sherpa-onnx transducer decoder: NVIDIA Parakeet-TDT (and other NeMo transducers).

sherpa-onnx's hotwords are an Aho-Corasick graph with no token cap — the way
past Whisper's ~35-term ceiling. They need `modified_beam_search`, which lost
speech on NeMo TDT models until the `length_reward` fix (k2-fsa/sherpa-onnx
#4084); `features()` detects it, and without it this decodes greedily with no
biasing rather than lose words.

One recognizer per model load: the encoder alone is ~650 MB, so hotwords are
never baked into the recognizer — each decode passes them to its own stream.
sherpa-onnx re-tokenizes hotwords with bpe.vocab, which disagrees with how the
model actually splits most proper nouns (`Hetzner` -> `▁H et zn er`, the
model writes `▁Het zn er`), so the graph never matches them. A build with the
pre-tokenized hotwords patch (`HOTWORDS_PRETOKENIZED`) takes the model's own
BPE pieces instead; otherwise sherpa's tokenization is the fallback.

Only proper nouns become hotwords: lowercase mined jargon caused false
insertions of those words on real dictation (measured, see CLAUDE.md).

Model directory layout (as published by sherpa-onnx):
    encoder*.onnx  decoder*.onnx  joiner*.onnx  tokens.txt
"""
from __future__ import annotations

import functools
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceio.core.decoders.base import (
    SAMPLE_RATE,
    Bias,
    BiasMode,
    Capabilities,
    DecodeError,
    DecodeResult,
)

log = logging.getLogger(__name__)

MBS = "modified_beam_search"
GREEDY = "greedy_search"
# The encoder breaks down on very long inputs (multi-minute audio crashed or
# dropped text); every decode is cut into pieces no longer than this.
MAX_CHUNK_SECS = 30.0
# Where in each 30 s window to look for the quietest spot to cut.
_CUT_SEARCH_FROM_SECS = 22.0
# Measured on 84 held-out dictations (parakeet-tdt-0.6b-v2 int8): beam 4,
# length reward 0.5, proper-noun hotwords at score 2.0 — 6.7% WER, 78% vocab
# recall. More paths cost time and found nothing more.
MAX_ACTIVE_PATHS = 4
LENGTH_REWARD = 0.5
HOTWORDS_SCORE = 2.0


@dataclass(frozen=True)
class SherpaFeatures:
    installed: bool = False
    # The NeMo TDT modified_beam_search fix (k2-fsa/sherpa-onnx#4084). Without
    # it, beam search — and so hotwords — drops speech on Parakeet.
    length_reward: bool = False
    # Hotwords may be given as tokens from tokens.txt (voiceio's patched build).
    pretokenized_hotwords: bool = False


@functools.cache
def features() -> SherpaFeatures:
    """What the installed sherpa-onnx can do (cached; never raises)."""
    try:
        import sherpa_onnx
    except Exception:  # noqa: BLE001 — absent or a broken native lib: not installed
        return SherpaFeatures()
    config = getattr(sherpa_onnx, "OfflineRecognizerConfig", None)
    length_reward = config is not None and hasattr(config, "length_reward")
    if not length_reward:
        import inspect
        try:
            params = inspect.signature(sherpa_onnx.OfflineRecognizer.from_transducer).parameters
            length_reward = "length_reward" in params
        except (AttributeError, TypeError, ValueError):
            pass
    return SherpaFeatures(True, length_reward,
                          bool(getattr(sherpa_onnx, "HOTWORDS_PRETOKENIZED", False)))


def default_threads() -> int:
    """Half the logical cores (≈ the physical ones), 2..6: one recognizer
    serves every pass, so it doesn't split them across lanes like Whisper."""
    return max(2, min(6, (os.cpu_count() or 8) // 2))


def _find(model_dir: Path, stem: str) -> str:
    hits = sorted(model_dir.glob(f"{stem}*.onnx"), key=lambda p: ("int8" not in p.name, p.name))
    if not hits:
        raise DecodeError(f"no {stem}*.onnx in {model_dir}")
    return str(hits[0])


def _special(piece: str) -> bool:
    return piece.startswith("<") and piece.endswith(">")


def is_proper_noun(term: str) -> bool:
    return any(c.isupper() for c in term)


def _usable(term: str) -> bool:
    # "/" separates hotwords in a stream's list; a word starting with ":" is
    # read as a boost score.
    return "/" not in term and not any(w.startswith(":") for w in term.split())


class Tokenizer:
    """The model's own BPE split of a term, from tokens.txt.

    sentencepiece BPE: start from characters (a word-start "▁" on each word)
    and keep merging the adjacent pair whose merged piece ranks best — piece
    ids are merge ranks. Checked against the model's output on every term of
    the vocabulary it was built for.
    """

    def __init__(self, tokens_txt: Path):
        self.pieces: dict[str, int] = {}
        for line in tokens_txt.read_text(encoding="utf-8").splitlines():
            piece, _, idx = line.rpartition(" ")
            if piece and idx.lstrip("-").isdigit():
                self.pieces[piece] = int(idx)
        self._merges = {p: i for p, i in self.pieces.items() if not _special(p)}

    def _word(self, word: str) -> list[str]:
        syms = list("▁" + word)
        while True:
            best = None
            for k in range(len(syms) - 1):
                rank = self._merges.get(syms[k] + syms[k + 1])
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, k)
            if best is None:
                return syms
            k = best[1]
            syms[k:k + 2] = [syms[k] + syms[k + 1]]

    def encode(self, term: str) -> list[str] | None:
        """Pieces of `term`, or None when a character has no piece."""
        out = [p for w in term.split() for p in self._word(w)]
        return out if out and all(p in self.pieces for p in out) else None

    def bpe_vocab(self) -> str:
        """bpe.vocab for sherpa's own hotword encoder: score = -rank."""
        return "".join(f"{p}\t{-i}\n" for p, i in sorted(self._merges.items(), key=lambda x: x[1]))


def _bpe_vocab_file(model_dir: Path, tokenizer: Tokenizer) -> str:
    """bpe.vocab next to the model (release tarballs don't ship one), or in
    a temp file when the model directory is read-only."""
    path = model_dir / "bpe.vocab"
    if path.is_file():
        return str(path)
    text = tokenizer.bpe_vocab()
    try:
        tmp = path.with_name(f".bpe.vocab.{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return str(path)
    except OSError:
        f = tempfile.NamedTemporaryFile("w", suffix=".bpe.vocab", delete=False, encoding="utf-8")
        with f:
            f.write(text)
        return f.name


def split_long(audio: np.ndarray, max_secs: float = MAX_CHUNK_SECS) -> list[np.ndarray]:
    """Cut audio into pieces of at most `max_secs`, each at the quietest spot
    of its last few seconds (the freeze cut's rule, so mostly a pause)."""
    from voiceio.streaming import _find_freeze_cut
    limit = int(max_secs * SAMPLE_RATE)
    search_from = min(_CUT_SEARCH_FROM_SECS, max_secs * 0.75)
    out, rest = [], audio
    while len(rest) > limit:
        cut = _find_freeze_cut(rest[:limit], SAMPLE_RATE, search_from, force=True) or limit
        out.append(rest[:cut])
        rest = rest[cut:]
    out.append(rest)
    return out


class SherpaTransducer:
    """Offline transducer decode via sherpa-onnx.

    Re-entrant: sherpa's recognizer is const and ONNX Runtime sessions are
    thread-safe, so a decode never queues behind another — stop's final pass
    runs alongside an in-flight interim instead of waiting for it (`draft`
    is this decoder, which lets the session decode past a pending freeze).
    """

    def __init__(self, model_dir: str | Path, *, threads: int = 4,
                 method: str = "", hotwords_score: float = HOTWORDS_SCORE,
                 max_active_paths: int = MAX_ACTIVE_PATHS,
                 length_reward: float = LENGTH_REWARD,
                 model_type: str = "nemo_transducer", label: str | None = None):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise DecodeError("sherpa-onnx not installed: pip install 'python-voiceio[sherpa]'") from e
        feats = features()
        self._dir = Path(model_dir).expanduser()
        if not method:
            method = MBS if feats.length_reward else GREEDY
        elif method == MBS and not feats.length_reward:
            log.warning("This sherpa-onnx lacks the NeMo TDT beam-search fix: modified_beam_search "
                        "may drop speech. Install the patched build: voiceio models sherpa")
        self._method = method
        mbs = method == MBS
        self.name = f"sherpa:{label or self._dir.name}" + ("" if mbs else f"@{method}")
        self.capabilities = Capabilities(bias=BiasMode.UNBOUNDED if mbs else BiasMode.NONE)
        self.draft = self
        self._pretokenized = mbs and feats.pretokenized_hotwords
        self._tokenizer = Tokenizer(self._dir / "tokens.txt") if mbs else None
        self._hotwords_cache: tuple[str | None, str] = (None, "")
        self._busy = 0
        self._idle = threading.Condition()
        kwargs = dict(
            encoder=_find(self._dir, "encoder"), decoder=_find(self._dir, "decoder"),
            joiner=_find(self._dir, "joiner"), tokens=str(self._dir / "tokens.txt"),
            num_threads=threads, decoding_method=method, model_type=model_type,
        )
        if mbs:
            kwargs.update(max_active_paths=max_active_paths, hotwords_score=hotwords_score,
                          modeling_unit="bpe",
                          bpe_vocab=_bpe_vocab_file(self._dir, self._tokenizer))
            if feats.length_reward:
                kwargs.update(length_reward=length_reward)
        # Upstream reads length_reward per stream (k2-fsa/sherpa-onnx#4084 after
        # review); builds without that read ignore the option.
        self._stream_reward = repr(float(length_reward)) if mbs and length_reward else ""
        t0 = time.monotonic()
        try:
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise DecodeError(f"sherpa-onnx could not load {self._dir}: {e}") from e
        log.info("Loaded %s (%s, %d threads, %s hotwords) in %.1fs", self.name, method, threads,
                 "pre-tokenized" if self._pretokenized else "sherpa-tokenized" if mbs else "no",
                 time.monotonic() - t0)

    def stream_hotwords(self, hotwords: str | None) -> str:
        """The per-stream hotword list for `Bias.hotwords` (", "-joined):
        proper nouns only, "/"-separated, as model tokens when supported."""
        if not hotwords or self._tokenizer is None:
            return ""
        if self._hotwords_cache[0] == hotwords:
            return self._hotwords_cache[1]
        terms = [t for t in dict.fromkeys(t.strip() for t in hotwords.split(","))
                 if t and is_proper_noun(t) and _usable(t)]
        out = []
        for term in terms:
            pieces = self._tokenizer.encode(term)
            if pieces is None:
                continue  # a character the model can't write
            out.append(" ".join(pieces) if self._pretokenized else term)
        rendered = "/".join(out)
        self._hotwords_cache = (hotwords, rendered)
        return rendered

    def decode(self, audio: np.ndarray, *, final: bool = False,
               bias: Bias = Bias()) -> DecodeResult:
        t0 = time.monotonic()
        hotwords = self.stream_hotwords(bias.hotwords) if self.capabilities.bias else ""
        audio = np.asarray(audio, dtype=np.float32)
        texts = []
        with self._idle:
            self._busy += 1
        try:
            for chunk in split_long(audio) if len(audio) else ():
                stream = (self._rec.create_stream(hotwords) if hotwords
                          else self._rec.create_stream())
                if self._stream_reward and hasattr(stream, "set_option"):
                    stream.set_option("length_reward", self._stream_reward)
                stream.accept_waveform(SAMPLE_RATE, chunk)
                self._rec.decode_stream(stream)
                if text := stream.result.text.strip():
                    texts.append(text)
        except Exception as e:  # noqa: BLE001
            raise DecodeError(f"sherpa-onnx decode failed: {e}") from e
        finally:
            with self._idle:
                self._busy -= 1
                self._idle.notify_all()
        return DecodeResult(" ".join(texts), (), time.monotonic() - t0)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no decode is running (benchmarks: an abandoned interim
        must not steal CPU from the next run)."""
        with self._idle:
            return self._idle.wait_for(lambda: self._busy == 0, timeout)

    def healthy(self) -> bool:
        return True

    def recover(self) -> None:
        pass

    def close(self) -> None:
        self._rec = None
