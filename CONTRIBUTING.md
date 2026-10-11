# Contributing to voiceio

## Quick start

```bash
git clone https://github.com/Hugo0/voiceio.git
cd voiceio
uv sync --extra desktop --extra server --extra dev
uv run voiceio setup   # bootstraps CLI symlinks into ~/.local/bin/
uv run pytest tests/ -x -q
```

Extras: `desktop` (mic, hotkeys, tray), `server`, `tts`, `sherpa`, `train`,
`dev`. The base install is headless: anything that imports sounddevice, evdev,
pynput or pystray must do so lazily or behind a desktop-only module, and CI
proves `.[server]` imports without them.

## Architecture

```
voiceio/
├── core/            # Display-free dictation core, shared by every front end
│   ├── engine.py    #   Engine: one decoder + vocabulary, corrections, pipeline
│   ├── pipeline.py  #   Text pipeline as composable stages (disfluency → cleanup → numbers
│   │                #   → commands → corrections → postcorrect → llm → prefix)
│   ├── bias.py      #   BiasProvider: vocabulary/hotwords/prompt per request
│   └── decoders/    #   Decoder protocol (base), model catalog, interim/final lanes,
│                    #   faster_whisper/ (subprocess worker, crash recovery), sherpa-onnx, whisper.cpp
├── learn/           # Personalization: store (corpus), importer, teacher (label), review,
│                    #   augment, metrics, evaluate (offline + streaming), train (LoRA →
│                    #   CTranslate2), clean
├── cli/             # CLI: __init__ (entry point, run/toggle/setup), doctor, servicecmd,
│                    #   uninstall, models, vocab, corrections, learn, history, configcmd
├── server/          # Headless HTTP/WebSocket server (python -m voiceio.server)
├── app.py           # Desktop daemon state machine (IDLE/RECORDING/FINALIZING/ERROR)
├── watchdog.py      # Health thread: worker recovery, audio stream, tray, resume detection
├── streaming.py     # Streaming session: interim/freeze/final decode scheduling
├── display.py       # TypedText: what the session showed, preedit or keystroke edits
├── audio_source.py  # Where streaming gets samples (microphone, socket, file)
├── recorder.py      # Microphone capture with 1s pre-buffer ring (desktop extra)
├── idle_audio.py    # When mic + chime streams stay open between dictations ([audio] keep_warm)
├── power.py         # on_battery() from sysfs (idle audio policy, nightly learning)
├── vad.py           # Voice Activity Detection (Silero ONNX / RMS fallback)
├── vocabulary.py    # vocabulary.txt, ranked to fit the hotword token budget
├── context_terms.py # On-screen vocabulary: terms from window title + selection, per dictation
├── vocab_stats.py   # Per-term usage stats that drive the ranking
├── tokens.py        # Exact Whisper token counting
├── prompt.py        # initial_prompt from recent transcript context
├── postprocess.py   # Rule-based cleanup and filler removal (stages live in core/pipeline.py)
├── numbers.py       # Spoken number → digit conversion ("twenty five" → "25")
├── commands.py      # Voice commands: "new line", "scratch that", "correct that"
├── corrections.py   # Corrections dictionary, plus rule_warnings() and the evidence audit
├── suggest.py       # Offline: unusual words in history (`vocab suggest`)
├── postcorrect.py   # Constrained final-pass LLM rewrite of misheard words (cloud, opt-in)
├── consent.py       # Explicit consent record for cloud LLM calls
├── privacy.py       # describe_transcript(): what a log may say about a transcript
├── retention.py     # Per-utterance audio + context storage, size caps
├── history.py       # Transcription history (JSONL)
├── llm.py           # Optional local LLM cleanup via Ollama
├── llm_api.py       # OpenAI-compatible chat client (OpenRouter/OpenAI/etc.)
├── config.py        # Config schema, TOML loading, data paths, 0600/0700 permissions
├── platform.py      # OS/DE/IM detection, distro-aware pkg_install(), system python lookup
├── health.py        # Diagnostic probes for all backends + features (`voiceio doctor`)
├── service.py       # systemd user units + CLI symlink management
├── feedback.py      # Sound cues (persistent output stream) + notifications
├── hints.py         # Contextual CLI hints (silenceable, frequency-limited)
├── demo.py          # Guided tour (voiceio demo)
├── clipboard_read.py # Clipboard / primary-selection read/write
├── focus.py         # Focused window's app id + title (Hyprland, sway, niri, X11)
├── wordfreq.py      # Word frequency lookup, edit distance
├── pidlock.py       # PID file locking
├── backends.py      # ProbeResult and shared backend types
├── setup/           # `voiceio setup`: wizard steps over a shared SetupState + non-interactive mode
├── hotkeys/         # evdev, pynput, Unix socket backends + chain resolution
├── typers/          # ibus, ydotool, wtype, xdotool, clipboard, pynput, fcitx bridge + chain;
│                    #   manager.py: TyperManager (selection, IDLE-gated upgrade/fallback)
├── ibus/            # IBus engine process (GLib main loop + socket listener);
│                    #   session.py: IBusSession (input sources, engine lifecycle)
│                    #   install.py: launcher, component XML, IBus env, GNOME source
├── tray/            # Animated tray icon (AppIndicator3 / pystray fallback)
├── tts/             # Text-to-speech engines (piper, espeak, edge-tts) + chain
├── data/            # Shipped static files (udev uaccess rule)
├── models/          # Bundled Silero VAD (ONNX)
└── sounds/          # WAV audio cues
```

Packaging lives at the root: `pyproject.toml`, and `flake.nix` + `nix/` (package,
NixOS module, Home Manager module).

## Key patterns

**State machine** — `app.py` uses `_State` enum protected by `_hotkey_lock`. Two counters: the generation (bumped on stop) decides who may still write output, so a slow finalizer never stomps a newer recording; the recording id (bumped on start) pairs each IBus claim with its release.

**One engine, many front ends** — the desktop daemon, the server, `voiceio eval` and `voiceio learn` all compose `core.engine.Engine` (decoder + bias + pipeline) instead of wiring those pieces themselves, so they cannot drift apart.

**Decoders** — every speech runtime implements the `core.decoders.base.Decoder` protocol and declares its `BiasMode` (hotwords, prompt, unbounded). `catalog.py` lists the models an adapter can run, with their licenses. voiceio never ships weights.

**Chain & probe** — Every backend implements `probe() → ProbeResult`. `chain.select()` picks the first working one. Runtime failures trigger automatic re-probe and fallback. A probe's `fix_cmd` is what `voiceio doctor --fix` runs; privileged ones are a single `sh -c` shown in a consent prompt (see `typers/ydotool.py`, `hotkeys/evdev.py`).

**Adding a backend** — Create `voiceio/typers/my_backend.py`, implement `TyperBackend` (or `StreamingTyper` for preedit; a backend that synthesizes key events subclasses `KeystrokeTyper` and implements `_type`/`_delete`, so it waits for held modifiers first), register in `__init__.py`, add to `chain.py`, add probe test.

**Hotkey deduplication** — evdev and socket both fire for the same keypress. `on_hotkey()` uses lock + 0.3s debounce.

**Streaming** — IBus path uses preedit (underlined preview) + commit. Fallback path uses word-level append with char-level diff on final.

**Incremental finalization** — during recording, the un-frozen audio tail is cut, beam-decoded and frozen, so interim and stop-time passes only decode the remaining tail. Two cuts: at every speech pause the recorder's VAD reports (≥`audio.silence_duration` of non-speech) once ≥4s of tail precede it, and, for speech with no such pause, once the tail exceeds `output.streaming_freeze_secs` (default 8s), at the quietest interior 300ms window (searched relative to the tail's own RMS — mic-gain independent). Pause cuts keep the stop-time tail to the last phrase or so, and land in real silence rather than a between-words dip. Long dictations finalize in O(tail) instead of O(recording) (~10x faster stop for multi-minute notes). Frozen text conditions the tail decode via per-call `context` prompt. Freeze passes never touch the display; decode failures never advance the boundary. With two decode lanes, interims run on the draft lane and freezes/finals on the final lane; if a freeze is still decoding at stop, the tail after its cut decodes in parallel on the draft lane.

**Post-processing pipeline** — `core.pipeline.Pipeline` is the single pipeline, built with `Pipeline.from_config(..., profile=...)`. Stages return text plus *effects*; side effects (writing flags, "scratch that") apply only on the final pass. Used by streaming, batch and server alike.

**LLM integration** — Optional Ollama-based grammar/spelling cleanup. Runs on final pass only (never during streaming). `llm.py` has shared helpers (install, start, pull, diagnose) used by setup, doctor, and app.

**Tray icon** — Pre-rendered PNG frames in freedesktop icon theme. AppIndicator3 subprocess under system Python (avoids GTK/venv conflicts). Phase-matched transitions between states. App works fine without it.

## Code style

- `ruff check voiceio/` (runs in CI)
- Python 3.11+, type hints on public APIs
- [Conventional Commits](https://www.conventionalcommits.org/) (feat/fix/refactor/docs/test/ci/chore)
- DRY: reuse existing utilities and patterns before writing new code
- Only validate at system boundaries, trust internal code
- Comments only where logic isn't self-evident
- Never log what the user said — at any level: the daemon writes DEBUG to
  `~/.local/state/voiceio/voiceio.log`, and under systemd INFO is the journal.
  Use `privacy.describe_transcript()` (a word count) instead.

## Testing

```bash
uv run pytest tests/ -x -q
```

- Mock external deps (audio, subprocesses, /dev/input)
- Use `spec=TyperBackend` on MagicMock (Python 3.11 protocol quirk)
- Test race conditions with `threading.Event` + timeouts
- `tests/conftest.py` sandboxes `HOME` and every XDG dir before voiceio is imported: a test can never touch your real config, history or recordings
- Speed and accuracy changes need numbers from real speech, not just green tests: `voiceio learn eval` (one decode per held-out clip) and `voiceio learn eval --streaming` (replays clips through live dictation, also times stop→text). Compare before/after on the same dataset version

## Pull requests

**Low quality spam PRs without a linked issue might be be closed.** Open an issue first, discuss the approach, get a thumbs-up, then submit code. This applies to AI-generated and human contributions equally.

- Link the issue in your PR description (`Fixes #123`)
- All tests must pass (`uv run pytest tests/ -x -q`)
- Add tests for new backends or bug fixes
- One logical change per PR

## Model choice: why Whisper, and what would change it

Short version: **Parakeet is faster and we still don't use it.** If you're about
to open "switch to Parakeet / Canary / Moonshine, it's Nx faster" — read this
first, then open the issue anyway if you have data that contradicts it.

Our errors are almost entirely proper nouns and jargon (`HyperNEAT` →
`hyperneed`, `Kalshi` → `kalchi`). So the axis that matters is **contextual
biasing**, not raw WER or speed. Whisper's only lever is `hotwords`, and it is
small and hard-capped: faster-whisper truncates hotwords at 223 tokens
(`max_length // 2 - 1`), they share one 448-token budget with the transcription
*output*, and proper nouns cost 5-7 tokens each — so ~35 terms, permanently.
That ceiling is architectural. There is no upstream fix coming, and it's why
`voiceio/vocabulary.py` ranks terms instead of just listing them.

**Measured on this project's own retained audio (2026-07-17, Ryzen 9 7940HS,
CPU int8)** — reproduce before trusting:

| | speed | notes |
|---|---|---|
| whisper-small (current) | 4.8x realtime | hotwords work; ~35-term ceiling |
| Parakeet-TDT-0.6b-v3 int8, greedy | **12.7x** | clean, but **greedy ignores hotwords** |
| Parakeet-TDT, `modified_beam_search` | — | **lost/truncated speech on ~8/20 real clips** |

[sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html) hotwords
are the prize: an Aho-Corasick automaton with **no token cap**, which would
delete the ~35-term ceiling entirely. But they require `modified_beam_search`
(greedy silently ignores them), and MBS on NeMo TDT is broken — see
[#3267](https://github.com/k2-fsa/sherpa-onnx/issues/3267). We reproduced it on
sherpa-onnx **1.13.4**: 3/20 clips returned empty on real speech, ~8/20 lost or
truncated it (greedy 47 words → MBS 0). A genuinely silent clip came back empty
from both, so that's real speech loss, not VAD.

So biasing and model choice are **one decision**, and today it costs 40% speech
loss. Without biasing, Parakeet recovers *fewer* of our vocabulary terms than
whisper-small does with hotwords (3 vs 5 over 20 clips). Speed is not the
bottleneck — whisper-small already runs 4.8x realtime on an 18s median
utterance.

**The trigger to revisit:** when
[PR #3657](https://github.com/k2-fsa/sherpa-onnx/pull/3657) ("Fix
modified_beam_search hallucination/empty text for NeMo TDT") merges, Parakeet
becomes faster *and* unbounded-biasing, and this decision flips. Re-run the
spike before migrating: decode ~20 real clips with greedy and MBS, compare word
counts per clip, and only proceed if MBS matches greedy. Note Parakeet is
offline-only (no true streaming) and crashes on multi-minute audio — chunk at
30s, which the freeze path already does.

## Platform notes

- **IBus** is the only reliable streaming text injection on Wayland. Keystroke simulation drops characters on rapid corrections.
- **Wayland caps a single message at 4096 bytes.** GNOME Shell relays preedit and commit strings to the focused app as one `zwp_text_input_v3` event, so an oversized string makes the compositor tear down that app's connection — it dies mid-dictation. Everything handed to IBus goes through `voiceio/ibus/textlimit.py`; never call `commit_text`/`update_preedit_text` with unbounded text.
- **evdev** needs read access to `/dev/input/event*`: the shipped udev `uaccess` rule (`voiceio/data/60-voiceio-uaccess.rules`, must sort before `73-seat-late.rules`) or, as a fallback, the `input` group. Multiple keyboard devices each get their own reader thread.
- **Sound** uses persistent `sounddevice.OutputStream`. WAV files padded with ~100ms silence.
- **Tray on non-Ubuntu** needs AppIndicator3 packages + GNOME Shell extension.
- **PyPI** package is `python-voiceio`, CLI is `voiceio`. Use `config.PYPI_NAME`.

## Releasing

```bash
# Bump version in pyproject.toml + voiceio/__init__.py
git commit -am "chore: bump version to X.Y.Z"
git tag vX.Y.Z
git push && git push --tags
# CI auto-publishes to PyPI + creates GitHub Release
```
