# voiceio

[![CI](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml/badge.svg)](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![Python](https://img.shields.io/pypi/pyversions/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/pepy/dt/python-voiceio)](https://pepy.tech/projects/python-voiceio)

Speak → text, locally, instantly.

## Quick start

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt install pipx ibus gir1.2-ibus-1.0 python3-gi portaudio19-dev

# 2. Install voiceio
pipx install python-voiceio

# 3. Run the setup wizard
voiceio setup
```

That's it. Press **Ctrl+Alt+V** (or your chosen hotkey) to start dictating.

<details>
<summary><strong>Fedora</strong></summary>

```bash
sudo dnf install pipx ibus python3-gobject portaudio-devel
pipx install python-voiceio
voiceio setup
```
</details>

<details>
<summary><strong>Arch Linux</strong></summary>

```bash
sudo pacman -S python-pipx ibus python-gobject portaudio
pipx install python-voiceio
voiceio setup
```
</details>

<details>
<summary><strong>Windows</strong></summary>

```powershell
# Option A: Install with pip (requires Python 3.11+)
pip install python-voiceio
voiceio setup

# Option B: Download the installer from GitHub Releases (no Python needed)
# https://github.com/Hugo0/voiceio/releases
# Also available as a portable .zip if you prefer no installation.
```

Windows uses pynput for hotkeys and text injection. No extra system dependencies required.
</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
pipx install python-voiceio
voiceio setup
```
</details>

<details>
<summary><strong>Build from source</strong></summary>

If you want the source code locally to hack on or customize for personal use. PRs are welcome!

```bash
git clone https://github.com/Hugo0/voiceio
cd voiceio
uv pip install -e ".[linux,dev]"

# Bootstrap CLI commands onto PATH (creates ~/.local/bin/voiceio)
uv run voiceio setup
```

> **Note:** Source installs live inside a virtualenv, so `voiceio` isn't on PATH until setup creates symlinks in `~/.local/bin/`. If `voiceio` isn't found after setup, restart your terminal or run `export PATH="$HOME/.local/bin:$PATH"`.
</details>

> You can also install with `uv tool install python-voiceio` or `pip install python-voiceio`.

## How it works

```
hotkey → mic capture → whisper (local) → text at cursor
          pre-buffered   streaming        IBus / clipboard
```

Press your hotkey to start recording (1s pre-buffer catches the first syllable). Text streams into the focused app as an underlined preview. Press again to commit. Transcription runs locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), text is injected through IBus (any GTK/Qt app) with clipboard fallback for terminals.

## Features

- **Streaming**: text appears as you speak, not after you stop
- **Works everywhere**: IBus input method for GUI apps, clipboard for terminals
- **Wayland + X11**: evdev hotkeys work on both, no root required
- **Pre-buffer**: never miss the first syllable
- **Voice commands**: "new line", "comma", "scratch that", punctuation by name
- **Autocorrect**: LLM-powered review of recurring Whisper mistakes (`voiceio correct`)
- **Text-to-speech**: hear selected text spoken back (Piper, eSpeak, Edge TTS)
- **Smart post-processing**: numbers ("twenty five" → "25"), punctuation, capitalization
- **Auto-healing**: falls back to the next working backend if one fails
- **Autostart**: optional systemd service, restarts on crash
- **Self-diagnosing**: `voiceio doctor` checks everything, `--fix` repairs it

## Models

| Model | Size | Speed | Accuracy | Good for |
|-------|------|-------|----------|----------|
| `tiny` | 75 MB | ~10x realtime | Basic | Quick notes, low-end hardware |
| `base` | 150 MB | ~7x realtime | Good | Daily use (default) |
| `small` | 500 MB | ~4x realtime | Better | Longer dictation |
| `medium` | 1.5 GB | ~2x realtime | Great | Accuracy-sensitive work |
| `large-v3` | 3 GB | ~1x realtime | Best | Maximum quality, GPU recommended |

Models download automatically on first use. Switch anytime: `voiceio --model small`.

## Commands

```
voiceio                  Start the daemon
voiceio setup            Interactive setup wizard
voiceio doctor           Health check (--fix to auto-repair)
voiceio test             Test microphone + live transcription
voiceio demo             Interactive guided tour of all features
voiceio toggle           Toggle recording on a running daemon
voiceio correct          Review and fix recurring transcription errors
voiceio history          View transcription history
voiceio update           Update to latest version
voiceio service install  Autostart on login (systemd / Windows Startup)
voiceio logs             View recent logs
voiceio uninstall        Remove all system integrations
```

## Configuration

`voiceio setup` handles everything interactively. To tweak later, edit the config file or override at runtime:

- Linux/macOS: `~/.config/voiceio/config.toml`
- Windows: `%LOCALAPPDATA%\voiceio\config\config.toml`

```bash
voiceio --model large-v3 --language auto -v
```

See [config.example.toml](config.example.toml) for all options.

## Troubleshooting

```bash
voiceio doctor           # see what's working
voiceio doctor --fix     # auto-fix issues
voiceio logs             # check debug output
```

| Problem | Fix |
|---------|-----|
| No text appears | `voiceio doctor --fix` - usually a missing IBus component or GNOME input source |
| Hotkey doesn't work on Wayland | `sudo usermod -aG input $USER` then log out and back in |
| Transcription too slow | Use a smaller model: `voiceio --model tiny` |
| Want to start fresh | `voiceio uninstall` then `voiceio setup` |
| Windows: antivirus blocks hotkeys | pynput uses global keyboard hooks — add an exception for voiceio |
| Windows: no sound feedback | Check `voiceio logs` for audio device info |
| macOS issues | Experimental — consider [aquavoice.com](https://aquavoice.com/) or contribute a PR |

## Platform support

| Platform | Status | Text injection | Hotkeys | Streaming preview |
|----------|--------|---------------|---------|-------------------|
| Ubuntu / Debian (GNOME, Wayland) | **Tested daily** | IBus | evdev / GNOME shortcut | Yes |
| Ubuntu / Debian (GNOME, X11) | Supported | IBus | evdev / pynput | Yes |
| Fedora (GNOME) | Supported | IBus | evdev / GNOME shortcut | Yes |
| Arch Linux | Supported | IBus | evdev | Yes |
| KDE / Sway / Hyprland | Should work | IBus / ydotool / wtype | evdev | Yes |
| Windows 10/11 | Experimental | pynput / clipboard | pynput | Type-and-correct (no preedit) |
| macOS | Experimental | pynput / clipboard | pynput | Type-and-correct (no preedit) |

voiceio auto-detects your platform and picks the best available backends. Run `voiceio doctor` to see what's working on your system.

## Uninstall

```bash
voiceio uninstall        # removes service, IBus, shortcuts, symlinks
pipx uninstall python-voiceio   # removes the package
```

## Roadmap

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) and [open issues](https://github.com/Hugo0/voiceio/issues).

**Now**
- [ ] macOS polish (IMKit for native preedit, Accessibility API for text injection)

**Soon**
- [ ] Per-app context awareness (detect focused app, adapt formatting/behavior)
- [ ] File/audio transcription mode (`voiceio transcribe recording.mp3`)

**Backlog**
- [ ] Multiple engine backends (whisper.cpp for Vulkan/AMD, VOSK for low-end hardware)
- [ ] Echo cancellation (filter system audio for meeting use)
- [ ] Wake word activation ("Hey voiceio")
**Done**
- [x] Text-to-speech output (Piper/eSpeak/Edge TTS — completes the "io")
- [x] LLM auto-audit dictionary (`voiceio correct --auto` — scan history with LLM, interactive correction)
- [x] LLM post-processing via Ollama (grammar cleanup, spelling fixes on final pass)
- [x] Corrections dictionary — auto-replace misheard words, "correct that" voice command
- [x] Transcription history — searchable log of everything you've dictated
- [x] Number-to-digit conversion ("three hundred forty two" → "342")
- [x] VAD-based silence filtering (Silero VAD, prevents Whisper hallucinations)
- [x] Voice commands — "new line", "new paragraph", "scratch that", punctuation by name
- [x] Custom vocabulary / personal dictionary (bias Whisper via `initial_prompt`)
- [x] Smart punctuation & capitalization post-processing
- [x] Windows support
- [x] System tray icon with animated states
- [x] Auto-stop on silence

## License

MIT