# Contributing to voiceio

## Quick start

```bash
git clone https://github.com/Hugo0/voiceio.git
cd voiceio
uv pip install -e ".[linux,dev]"
uv run voiceio setup   # bootstraps CLI symlinks into ~/.local/bin/
uv run pytest tests/ -x -q
```

## Architecture

```
voiceio/
├── app.py           # State machine (IDLE/RECORDING/FINALIZING/ERROR), health watchdog
├── cli.py           # CLI: setup, doctor, test, toggle, service, logs, uninstall
├── config.py        # Config schema + TOML loading
├── platform.py      # OS/DE detection, distro-aware pkg_install()
├── recorder.py      # Audio capture with 1s pre-buffer ring
├── transcriber.py   # Whisper subprocess with crash recovery
├── streaming.py     # Real-time transcription with word-level corrections
├── health.py        # Diagnostic probes for all backends + tray
├── feedback.py      # Sound playback (persistent sounddevice stream) + notifications
├── service.py       # Systemd service + CLI symlink management
├── wizard.py        # Interactive setup wizard
├── worker.py        # Whisper worker subprocess
├── hotkeys/         # evdev, pynput, Unix socket backends + chain resolution
├── typers/          # ibus, ydotool, wtype, xdotool, clipboard, pynput + chain
├── ibus/            # IBus engine process (GLib main loop + socket listener)
├── tray/            # Animated tray icon (AppIndicator3 / pystray fallback)
└── sounds/          # WAV audio cues
```

## Key patterns

**State machine** — `app.py` uses `_State` enum protected by `_hotkey_lock`. Generation counter prevents stale finalizers from stomping newer recordings.

**Chain & probe** — Every backend implements `probe() → ProbeResult`. `chain.select()` picks the first working one. Runtime failures trigger automatic re-probe and fallback.

**Adding a backend** — Create `voiceio/typers/my_backend.py`, implement `TyperBackend` (or `StreamingTyper` for preedit), register in `__init__.py`, add to `chain.py`, add probe test.

**Hotkey deduplication** — evdev and socket both fire for the same keypress. `on_hotkey()` uses lock + 0.3s debounce.

**Streaming** — IBus path uses preedit (underlined preview) + commit. Fallback path uses word-level append with char-level diff on final.

**Tray icon** — Pre-rendered PNG frames in freedesktop icon theme. AppIndicator3 subprocess under system Python (avoids GTK/venv conflicts). Phase-matched transitions between states. App works fine without it.

## Code style

- `ruff check voiceio/` (runs in CI)
- Python 3.11+, type hints on public APIs
- [Conventional Commits](https://www.conventionalcommits.org/) (feat/fix/refactor/docs/test/ci/chore)
- DRY: reuse existing utilities and patterns before writing new code
- Only validate at system boundaries, trust internal code
- Comments only where logic isn't self-evident

## Testing

```bash
uv run pytest tests/ -x -q
```

- Mock external deps (audio, subprocesses, /dev/input)
- Use `spec=TyperBackend` on MagicMock (Python 3.11 protocol quirk)
- Test race conditions with `threading.Event` + timeouts

## Pull requests

**Low quality spam PRs without a linked issue might be be closed.** Open an issue first, discuss the approach, get a thumbs-up, then submit code. This applies to AI-generated and human contributions equally.

- Link the issue in your PR description (`Fixes #123`)
- All tests must pass (`uv run pytest tests/ -x -q`)
- Add tests for new backends or bug fixes
- One logical change per PR

## Platform notes

- **IBus** is the only reliable streaming text injection on Wayland. Keystroke simulation drops characters on rapid corrections.
- **evdev** requires `input` group. Multiple keyboard devices each get their own reader thread.
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
