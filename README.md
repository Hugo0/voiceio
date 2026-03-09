# voiceio

[![CI](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml/badge.svg)](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voiceio)](https://pypi.org/project/voiceio/)
[![Python](https://img.shields.io/pypi/pyversions/voiceio)](https://pypi.org/project/voiceio/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Push-to-talk voice-to-text for Linux and macOS, on any app. Press a hotkey, speak, press again - text appears at your cursor.

100% local and open source. No API keys, no cloud, no telemetry. Use and modify at your will.

<!-- demo video -->
<p align="center">
  <a href="https://www.tella.tv/video/YOUR_VIDEO_ID">
    <img src="https://github.com/Hugo0/voiceio/raw/main/assets/demo-thumbnail.png" alt="voiceio demo" width="600">
  </a>
  <br>
  <em>Click to watch the demo</em>
</p>

## Quick start

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt install pipx ibus gir1.2-ibus-1.0 python3-gi portaudio19-dev

# 2. Install voiceio
pipx install voiceio

# 3. Run the setup wizard
voiceio setup
```

That's it. Press **Ctrl+Alt+V** (or your chosen hotkey) to start dictating.

<details>
<summary><strong>Fedora</strong></summary>

```bash
sudo dnf install pipx ibus python3-gobject portaudio-devel
pipx install voiceio
voiceio setup
```
</details>

<details>
<summary><strong>Arch Linux</strong></summary>

```bash
sudo pacman -S python-pipx ibus python-gobject portaudio
pipx install voiceio
voiceio setup
```
</details>

<details>
<summary><strong>Build from source</strong></summary>

If you want the source code locally to hack on or customize for personal use. PRs are welcome!

```bash
git clone https://github.com/Hugo0/voiceio
cd voiceio
pip install -e ".[linux,dev]"
voiceio setup
```
</details>

> You can also install with `uv tool install voiceio` or `pip install voiceio`.

## How it works

```
hotkey → mic capture → whisper (local) → text at cursor
          pre-buffered   streaming        IBus / clipboard
```

1. Press your hotkey: voiceio starts recording (with a 1-second pre-buffer, so it catches the beginning even if you start speaking before pressing)
2. Speak naturally: text streams into the focused app in real-time as an underlined preview
3. Press the hotkey again: the final transcription replaces the preview and is committed

Transcription runs locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Text is injected through IBus (works in any GTK/Qt app: browsers, Telegram, editors) with an automatic clipboard fallback for terminals.

## Features

- **Streaming**: text appears as you speak, not after you stop
- **Works everywhere**: IBus input method for GUI apps, clipboard for terminals
- **Wayland + X11**: evdev hotkeys work on both, no root required
- **Pre-buffer**: never miss the first syllable
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
voiceio toggle           Toggle recording on a running daemon
voiceio service install  Autostart on login via systemd
voiceio logs             View recent logs
voiceio uninstall        Remove all system integrations
```

## Configuration

`voiceio setup` handles everything interactively. To tweak later, edit `~/.config/voiceio/config.toml` or override at runtime:

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
| Doesn't work on MacOS | I haven't added proper support for apple yet. either use https://aquavoice.com/ or make a PR |

## Platform support

| Platform | Status | Text injection | Hotkeys | Streaming preview |
|----------|--------|---------------|---------|-------------------|
| Ubuntu / Debian (GNOME, Wayland) | **Tested daily** | IBus | evdev / GNOME shortcut | Yes |
| Ubuntu / Debian (GNOME, X11) | Supported | IBus | evdev / pynput | Yes |
| Fedora (GNOME) | Supported | IBus | evdev / GNOME shortcut | Yes |
| Arch Linux | Supported | IBus | evdev | Yes |
| KDE / Sway / Hyprland | Should work | IBus / ydotool / wtype | evdev | Yes |
| macOS | Experimental | pynput / clipboard | pynput | Type-and-correct (no preedit) |

voiceio auto-detects your platform and picks the best available backends. Run `voiceio doctor` to see what's working on your system.

## Uninstall

```bash
voiceio uninstall        # removes service, IBus, shortcuts, symlinks
pipx uninstall voiceio   # removes the package
```

## TODO

**Launch**
- [ ] Publish to PyPI
- [ ] Record demo video + thumbnail
- [ ] Test clean install on a fresh VM/container
- [ ] GitHub repo: description, topics, social preview image
- [ ] Bump version to 0.2.0

**Code quality**
- [ ] IBus activation on non-GNOME desktops (KDE, Sway, Hyprland), currently GNOME-only via gsettings
- [ ] `voiceio doctor --json` for machine-readable output
- [ ] Shell completions (`voiceio completion bash/zsh/fish`)
- [ ] Refactor wizard.py (882 lines) into smaller, testable modules
- [ ] Socket protocol versioning (e.g. `v1:preedit:text`)
- [ ] Configurable log file path

## Wishlist

Contributions welcome! Open an issue to discuss before starting.

**High impact**
- [ ] **Text-to-speech (voice output)**: select text, press a hotkey, hear it spoken aloud. Completes the "io" in voiceio. Use a local TTS engine (Piper, Coqui, espeak-ng), same philosophy: no cloud, no API keys
- [ ] **Wake word**: "Hey voiceio" hands-free activation (no hotkey needed). Use a small always-on keyword model (e.g. openWakeWord, Porcupine)
- [ ] **Custom vocabulary / hot words**: user-defined word list for names, jargon, technical terms that Whisper gets wrong. Boost via `initial_prompt` or fine-tuned logit bias
- [ ] **Per-app profiles**: different language/model/output settings per application (e.g. formal writing in docs, casual in chat)
- [ ] **Voice commands**: "select all", "new line", "undo that", "delete last sentence". Parse transcribed text for command patterns before injecting
- [ ] **Punctuation & formatting commands**: "period", "comma", "new paragraph", "capitalize that"
- [ ] **Auto-punctuation model**: post-process Whisper output with a small punctuation/capitalization model for cleaner text

**Platform expansion**
- [ ] **macOS Input Method (IMKit)**: native streaming preedit on macOS, matching IBus quality on Linux
- [ ] **Windows support**: Text Services Framework (TSF) for text injection, global hotkeys via win32api
- [ ] **Flatpak / Snap packaging**: sandboxed distribution for Linux
- [ ] **AUR package**: community package for Arch Linux

**UX polish**
- [ ] **System tray icon with recording animation**: pulsing/colored icon showing recording state, quick menu for model/language switching
- [ ] **Desktop notifications with transcribed text**: show what was typed, with an undo button
- [ ] **Confidence indicator**: visual hint when Whisper is uncertain (maybe highlight low-confidence words)
- [ ] **Recording timeout**: auto-stop after N seconds of silence or max duration, preventing forgotten recordings
- [ ] **Sound themes**: bundled sound packs (subtle, mechanical, sci-fi, none)
- [ ] **First-run onboarding overlay**: lightweight "press Ctrl+Alt+V to start" hint on first launch

**Power features**
- [ ] **Multi-language in one session**: auto-detect language switches mid-dictation (Whisper supports this but needs tuning)
- [ ] **Speaker diarization**: "Person 1: ... Person 2: ..." for meeting notes (via pyannote or whisperX)
- [ ] **LLM post-processing**: pipe transcription through a local LLM (Ollama) for grammar correction, summarization, or reformatting
- [ ] **Clipboard history**: keep last N transcriptions, quick-paste from history
- [ ] **Transcription log / journal**: searchable history of everything you've dictated, with timestamps
- [ ] **API / webhook**: expose a local API so other tools can trigger recording or receive transcriptions
- [ ] **Browser extension**: inject text into web apps that don't work with IBus (e.g. some Electron apps)

**Developer experience**
- [ ] **Plugin system**: hooks for pre/post processing (e.g. custom formatters, translators, text transforms)
- [ ] **Alternative STT backends**: support Whisper.cpp, Deepgram, AssemblyAI, OpenAI Whisper API as optional backends
- [ ] **GPU acceleration docs**: CUDA/ROCm setup guide for faster transcription on large models

## License

MIT
