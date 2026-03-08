# voiceio

Push-to-talk voice-to-text for Linux. Hold a hotkey, speak, release — text appears at your cursor.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for fast, local, offline transcription. No API keys, no cloud, no telemetry.

## Install

```bash
# System dependencies
sudo apt install xdotool xclip

# Install voiceio
pip install .

# Or with tray icon support
pip install ".[tray]"
```

## Usage

```bash
# Run with defaults (hold Right Super key to record, base model)
voiceio

# Use a larger model for better accuracy
voiceio --model large-v3

# Auto-detect language
voiceio --language auto

# Verbose output
voiceio -v
```

## Configuration

Copy the example config and edit:

```bash
mkdir -p ~/.config/voiceio
cp config.example.toml ~/.config/voiceio/config.toml
```

See `config.example.toml` for all options.

## Run as a systemd service

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/voiceio.service << 'EOF'
[Unit]
Description=voiceio push-to-talk

[Service]
ExecStart=%h/.local/bin/voiceio
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user enable --now voiceio
```

## How it works

1. **pynput** listens for your configured hotkey (global, works in any app)
2. **sounddevice** captures microphone audio while the key is held
3. **faster-whisper** transcribes the audio locally using a Whisper model
4. **xdotool** types the result at your cursor position

Latency is typically 1-2 seconds after releasing the key, depending on the model and audio length.

## Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny | 75 MB | Fastest | Basic |
| base | 150 MB | Fast | Good |
| small | 500 MB | Moderate | Better |
| medium | 1.5 GB | Slower | Great |
| large-v3 | 3 GB | Slowest | Best |

Models are downloaded automatically on first use.

## Known limitations

- **X11 only**: pynput and xdotool require X11. On Wayland, they work through XWayland but may not capture hotkeys from all windows.
- **Clipboard method**: Using `output.method = "xclip"` overwrites your clipboard.

## License

MIT
