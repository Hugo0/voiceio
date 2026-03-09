# Contributing to voiceio

## Quick start

```bash
git clone https://github.com/Hugo0/voiceio.git
cd voiceio
pip install -e ".[dev]"
pytest tests/ -x -q
```

## Architecture

```
voiceio/
├── app.py              # Core engine: state machine, hotkey handler, self-healing
├── cli.py              # CLI entry point (setup, doctor, test, toggle, service)
├── config.py           # Config schema + TOML loading
├── platform.py         # Platform detection (OS, display server, desktop)
├── recorder.py         # Audio capture with 1s pre-buffer ring
├── transcriber.py      # Whisper subprocess wrapper with crash recovery
├── streaming.py        # Real-time transcription with word-level corrections
├── hotkeys/            # Hotkey detection backends
│   ├── chain.py        #   Chain resolution (platform → ordered backend list)
│   ├── evdev.py        #   Linux evdev (direct /dev/input)
│   ├── pynput_backend.py   X11/macOS
│   └── socket_backend.py   Unix socket (GNOME shortcut → voiceio-toggle)
├── typers/             # Text injection backends
│   ├── chain.py        #   Chain resolution
│   ├── base.py         #   TyperBackend + StreamingTyper protocols
│   ├── ibus.py         #   IBus input method (preedit + commit, preferred)
│   ├── ydotool.py      #   Wayland keystroke simulation
│   ├── wtype.py        #   Wayland (Sway/Hyprland)
│   ├── xdotool.py      #   X11 keystroke simulation
│   ├── clipboard.py    #   Clipboard paste (universal fallback)
│   └── pynput_type.py  #   macOS
├── ibus/               # IBus engine process (separate GLib main loop)
│   └── engine.py       #   Engine + Unix socket listener
├── sounds/             # WAV audio cues (start/stop/commit)
├── feedback.py         # Sound playback + desktop notifications
├── service.py          # Systemd service management
├── wizard.py           # Interactive setup wizard
└── worker.py           # Whisper worker subprocess
```

## Key patterns

### Chain & probe

Every backend implements `probe() → ProbeResult(ok, reason, fix_hint)`. Chains are platform-specific ordered lists. `chain.select()` picks the first working backend. If a backend fails at runtime, `_type_with_fallback()` re-probes and switches automatically.

### Adding a backend

1. Create `voiceio/typers/my_backend.py` (or `hotkeys/`)
2. Implement the `TyperBackend` protocol (or `StreamingTyper` for preedit support)
3. Register in `__init__.py`
4. Add to the appropriate chains in `chain.py`
5. Add a probe test in `tests/test_backend_probes.py`

### Dual hotkey deduplication

Both evdev and socket backends fire for the same keypress (~30-200ms apart). `on_hotkey()` uses a blocking lock + 0.3s timestamp debounce, with the timestamp updated *after* the handler completes to prevent phantom recordings from threads waiting behind the lock.

### Streaming text injection

- **IBus path**: preedit (underlined preview) updated each cycle, committed on stop
- **Fallback path**: word-level append-only with fuzzy punctuation matching, char-level diff on final commit

## Code style

- **Formatter/linter**: `ruff check voiceio/` (runs in CI)
- **Python**: 3.11+, type hints on public APIs
- **Imports**: stdlib → third-party → local, separated by blank lines
- **No over-engineering**: don't add abstractions for one-time operations. Three similar lines > premature helper function
- **No unnecessary error handling**: trust internal code, only validate at system boundaries
- **Comments**: only where logic isn't self-evident. No docstrings on obvious methods

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add wtype backend for Sway/Hyprland
fix: prevent phantom recording from dual hotkey race
refactor: extract chain resolution into shared module
docs: update setup instructions for Wayland
test: add concurrent hotkey deduplication test
ci: add trusted publishing workflow
chore: bump version to 0.3.0
```

- Keep messages short (< 72 chars for subject line)
- Use body for *why*, not *what*
- One logical change per commit

## Testing

```bash
pytest tests/ -x -q          # run all tests
pytest tests/test_streaming.py -x -q  # run specific file
```

- Mock external dependencies (audio, subprocesses, /dev/input)
- Use `spec=TyperBackend` on MagicMock to prevent false protocol matches (Python 3.11 quirk)
- Test chain resolution for all platform combinations
- Race conditions: use `threading.Thread` + `threading.Event` with timeouts

## Platform notes

- **Wayland/GNOME**: IBus is the only reliable text injection method for streaming. Keystroke simulation (ydotool/wtype) drops characters on rapid corrections.
- **evdev**: requires `input` group membership. Multiple keyboard devices are common (3+ on laptops) — each gets its own reader thread.
- **PipeWire**: `pw-play` clips the tail of very short WAV files. All sounds need ~150ms silence padding.
- **pipx installs**: the PyPI package is `python-voiceio`, the CLI command is `voiceio`. Use `config.PYPI_NAME` constant for the package name.

## Releasing

Releases are automated via GitHub Actions + PyPI Trusted Publishing:

1. Bump version in `pyproject.toml` and `voiceio/__init__.py`
2. Commit: `chore: bump version to X.Y.Z`
3. Push to main
4. Create a GitHub Release with tag `vX.Y.Z`
5. The `publish.yml` workflow builds and publishes to PyPI automatically
