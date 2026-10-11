"""On-screen vocabulary: bias one dictation toward the names you can see.

Most misrecognitions are proper nouns and jargon, and the word you are about
to say is often already on screen: in the focused window's title or in the
text you just selected. This reads those two sources when a recording starts,
picks out the terms Whisper is likely to miss (rare words, CamelCase,
acronyms, multiword names) and hands them to `core.bias` for this one
recording, inside a small share of the hotword budget.

Capture happens off the hotkey path, under a hard deadline, and degrades to
"no terms" on any failure. Nothing captured here is logged or written to
disk; only a term count is logged.

Not used: AT-SPI (the focused widget's text). It needs PyGObject, which the
venv can't import, so a system-Python helper; a one-shot walk of the
accessibility tree took 120-900 ms on GNOME, and Chromium/Electron apps expose
their text only while a screen reader is on.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Capture deadline. Long enough for two ~20 ms subprocess reads under load,
# short enough that the terms are in place before the first interim pass.
CAPTURE_TIMEOUT_SECS = 0.15
# Only the head of a large selection is scanned: extraction stays ~10 ms.
MAX_SCAN_CHARS = 8000
MAX_TERMS = 8

# Zipf thresholds (see wordfreq.py: 2.0 ≈ top 30K words). A capitalized word
# is a name, and Whisper misses names it has seen far more often than generic
# rare words (Kosh 2.1, Wayland 2.7), so names get a higher bar.
_ZIPF_LOWER = 2.0
_ZIPF_NAME = 3.0
_ZIPF_MIXED = 3.5
_ZIPF_COMMON = 4.0

_TOKEN = re.compile(r"[\w'’.\-/@#:+]+")
_WORD = re.compile(r"^[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’\-]*[A-Za-zÀ-ɏ]$")
_MIXED = re.compile(r"[a-zà-ÿ][A-ZÀ-Þ]|[A-ZÀ-Þ]{2}[a-zà-ÿ]")


@dataclass(frozen=True)
class ScreenContext:
    title: str = ""
    selection: str = ""
    secs: float = 0.0

    def sources(self) -> list[str]:
        return [n for n, v in (("title", self.title), ("selection", self.selection)) if v]


def _kind(word: str, zipf: Callable[[str], float]) -> int:
    """How much a word looks like something Whisper will miss: 3 = CamelCase
    or rare name, 2 = acronym, 1 = rare lowercase word, 0 = not a term."""
    if not _WORD.match(word) or len(word) < 3:
        return 0
    if _MIXED.search(word):
        return 3 if len(word) <= 20 and zipf(word) < _ZIPF_MIXED else 0
    if word.isupper():
        return 2 if len(word) <= 6 and zipf(word) < _ZIPF_NAME else 0
    if word[0].isupper():
        return 3 if zipf(word) < _ZIPF_NAME else 0
    return 1 if len(word) >= 4 and zipf(word) < _ZIPF_LOWER else 0


def _words(text: str) -> list[list[str]]:
    """Runs of plain words; an identifier, URL or number ends a run, and its
    letter parts (voiceio.dev → voiceio) stand alone."""
    runs: list[list[str]] = [[]]
    for line in text.splitlines():
        runs.append([])
        for tok in _TOKEN.findall(line):
            stripped = tok.strip("'’.-:")
            if _WORD.match(stripped):
                runs[-1].append(stripped)
                if tok.rstrip("'’") != stripped:   # trailing punctuation ends the run
                    runs.append([])
                continue
            runs.append([])
            for part in re.split(r"[^A-Za-zÀ-ɏ'’]+", tok):
                if _WORD.match(part):
                    runs[-1].append(part)
                    runs.append([])
    return [r for r in runs if r]


def _name_end(run: list[str], kinds: list[int], i: int,
              zipf: Callable[[str], float]) -> int:
    """End of a multiword name starting at run[i] (2-3 capitalized words,
    one of them rare), or i when there is none. Very common words don't
    start or end a name: "Hey Kosh" is a greeting, not a name."""
    j = i
    while j < len(run) and j - i < 3 and run[j][0].isupper() and not run[j].isupper():
        j += 1
    if j - i < 2 or zipf(run[i]) >= _ZIPF_COMMON or run[i] == run[i + 1]:
        return i
    while j - i >= 2 and zipf(run[j - 1]) >= _ZIPF_COMMON:
        j -= 1
    return j if j - i >= 2 and any(k == 3 for k in kinds[i:j]) else i


def extract_terms(
    texts: Iterable[str],
    *,
    language: str = "en",
    exclude: Iterable[str] = (),
    prefer: Iterable[str] = (),
    max_terms: int = MAX_TERMS,
) -> list[str]:
    """Candidate hotwords from on-screen text, best first.

    `texts` in priority order (title before selection). `exclude` holds terms
    already biased (case-insensitive); `prefer` the user's full vocabulary,
    whose long tail never fits the budget — seeing one of them on screen is
    the strongest signal there is.
    """
    from functools import lru_cache

    from voiceio.wordfreq import _zipf

    zipf = lru_cache(maxsize=4096)(lambda w: _zipf(w, language))
    excluded = {t.lower() for t in exclude}
    preferred = {t.lower() for t in prefer}
    # lowercased term → [score, first seen, spelling as first seen]
    scores: dict[str, list] = {}

    def add(term: str, score: float) -> None:
        key = term.lower()
        if key in excluded:
            return
        if key in scores:
            # Repeats add a little, so the term the page is about outranks a
            # one-off, without letting a long page flood the list.
            scores[key][0] += 0.5
            return
        scores[key] = [score + (3 if key in preferred else 0), len(scores), term]

    texts = list(texts)
    for rank, text in enumerate(texts):
        weight = (len(texts) - rank) * 0.1   # earlier source wins ties
        for run in _words((text or "")[:MAX_SCAN_CHARS]):
            kinds = [_kind(w, zipf) for w in run]
            i = 0
            while i < len(run):
                j = _name_end(run, kinds, i, zipf)
                if j > i + 1:
                    add(" ".join(run[i:j]), 3 + weight)
                    i = j
                    continue
                if kinds[i]:
                    add(run[i], kinds[i] + weight)
                i += 1

    ranked = sorted(scores.values(), key=lambda v: (-v[0], v[1]))
    out: list[str] = []
    seen_words: set[str] = set()
    for _, _, term in ranked:
        parts = term.lower().split()
        if seen_words.intersection(parts):
            continue   # a name and its parts cost tokens twice for one word
        out.append(term)
        seen_words.update(parts)
        if len(out) >= max_terms:
            break
    return out


def warm(language: str = "en") -> None:
    """Load the word frequency tables now, not on the first dictation."""
    extract_terms(["Warmup"], language=language)


# ── capture ────────────────────────────────────────────────────────────────


def _read_primary(timeout: float) -> str:
    """The PRIMARY selection (highlighted text). xclip through XWayland first:
    on GNOME, wl-paste can take keyboard focus for a frame, which drops an
    IBus preedit — never acceptable at the start of a dictation."""
    from voiceio.platform import detect
    p = detect()
    if os.environ.get("DISPLAY") and shutil.which("xclip"):
        argv = ["xclip", "-o", "-selection", "primary"]
    elif p.is_wayland and p.desktop != "gnome" and shutil.which("wl-paste"):
        argv = ["wl-paste", "-p", "--no-newline"]
    else:
        return ""
    try:
        out = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout[:MAX_SCAN_CHARS * 4].decode("utf-8", "replace")


def _read_title(_timeout: float) -> str:
    from voiceio import retention
    return retention.active_window_title() or ""


def capture(timeout: float = CAPTURE_TIMEOUT_SECS, *,
            readers: dict[str, Callable[[float], str]] | None = None) -> ScreenContext:
    """Read the window title and selection in parallel; whatever isn't back
    by `timeout` is dropped (its thread finishes on its own). Never raises."""
    readers = readers or {"title": _read_title, "selection": _read_primary}
    t0 = time.monotonic()
    results: dict[str, str] = {}

    def run(name: str, fn: Callable[[float], str]) -> None:
        try:
            results[name] = fn(timeout) or ""
        except Exception:  # noqa: BLE001 — context is a bonus
            log.debug("screen context: %s failed", name, exc_info=True)

    threads = [threading.Thread(target=run, args=item, daemon=True,
                                name=f"screen-{item[0]}") for item in readers.items()]
    for t in threads:
        t.start()
    deadline = t0 + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    got = dict(results)   # snapshot: a late reader must not change it
    return ScreenContext(title=got.get("title", ""), selection=got.get("selection", ""),
                         secs=time.monotonic() - t0)


def describe() -> str:
    """For `voiceio doctor`: which sources this session can read."""
    from voiceio.focus import focused_window
    from voiceio.platform import detect
    p = detect()
    sources = []
    if os.environ.get("DISPLAY") and shutil.which("xclip"):
        sources.append("selection (xclip)")
    elif p.is_wayland and p.desktop != "gnome" and shutil.which("wl-paste"):
        sources.append("selection (wl-paste)")
    if focused_window(xwayland=True) is not None:
        sources.append("window title")
    elif p.is_wayland and p.desktop in ("gnome", "kde"):
        sources.append("window title of X11 apps only")
    return ", ".join(sources) or "no source available"
