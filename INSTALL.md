# Installing voiceio (guide for AI agents)

Terse, imperative, copy-pasteable. This is the install runbook for an AI agent
setting up **voiceio** (local voice dictation) on a user's **Linux** machine.
Windows/macOS are best-effort and out of scope here. NixOS: use the flake (see README).

> Contributing to the codebase instead? See [CONTRIBUTING.md](CONTRIBUTING.md)
> (also exposed as `AGENTS.md`).

## 0. Preconditions

- Linux with a working microphone and a graphical session (Wayland or X11).
- `sudo` available for the one system-package step.
- Python 3.11+.

## 1a. Shortcut: distro package (Debian/Ubuntu, Arch)

A package installs voiceio and its system dependencies in one step; skip
sections 1 and 2 and go to 3. Upgrade it with the package manager, not pipx.

```bash
# Debian / Ubuntu 22.04+ (amd64): the .deb from the latest GitHub release
curl -fLO "$(curl -fsS https://api.github.com/repos/Hugo0/voiceio/releases/latest \
  | grep -o 'https://[^"]*voiceio_[^"]*_amd64\.deb' | head -n1)"
sudo apt install -y ./voiceio_*_amd64.deb

# Arch (AUR)
yay -S --noconfirm python-voiceio
```

Verify with `voiceio --version`. Don't mix this with a pipx install: if
`pipx list` shows python-voiceio, `pipx uninstall python-voiceio` first.

## 1. Install system dependencies (distro-specific)

voiceio needs a C toolchain (the `evdev` hotkey backend has **no prebuilt
wheel** — it compiles on install), PortAudio (mic capture), and IBus + its
GObject bindings (text injection). Run the line for the user's distro:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y pipx build-essential python3-dev portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi

# Fedora
sudo dnf install -y pipx gcc gcc-c++ make python3-devel portaudio-devel ibus ibus-libs python3-gobject

# Arch
sudo pacman -S --noconfirm python-pipx base-devel portaudio ibus python-gobject
```

Not sure of the exact set? After installing voiceio you can always ask it:

```bash
voiceio doctor          # prints "Missing system dependencies:" + one install command
```

## 2. Install voiceio

```bash
pipx install 'python-voiceio[desktop]'   # the base package alone is headless (no mic/hotkeys)
pipx ensurepath          # puts ~/.local/bin on PATH; open a new shell after
```

Verify:

```bash
voiceio --version        # expect: voiceio X.Y.Z
```

If `voiceio: command not found`, PATH isn't updated yet: run
`export PATH="$HOME/.local/bin:$PATH"` or start a new shell.

## 3. Configure — non-interactive

Use the non-interactive path (no TTY, no prompts). It prints machine-readable
`[voiceio-setup] step=... status=...` lines and exits non-zero on failure.

```bash
# Accept all recommended defaults (model=small, hotkey=ctrl+alt+v, streaming on)
voiceio setup --defaults
```

Override any subset with JSON (unspecified keys keep their defaults):

```bash
voiceio setup --answers '{"model":"base","language":"en","hotkey":"ctrl+alt+v","install_service":true}'
```

Add `--yes` to let setup run the system-package install command itself if a
dependency is still missing (otherwise it only prints the command).

Accepted `--answers` keys: `model` (tiny|base|small|distil-large-v3|large-v3-turbo),
`language`, `hotkey`, `backend` (auto|evdev|socket|pynput), `streaming`,
`tts_enabled`, `tts_engine`, `tts_speed`, `llm_enabled`, `llm_model`,
`cloud_api_key`, `cloud_base_url`, `cloud_model`,
`sound_enabled`, `notify_clipboard`, `commands_enabled`, `punctuation_cleanup`,
`number_conversion`, `voice_input_prefix`, `tray_enabled`, `install_service`,
`download_model`, `install_system_deps`.

Success looks like a final line:

```
[voiceio-setup] step=done status=ok hotkey=ctrl+alt+v config=/home/<user>/.config/voiceio/config.toml
```

Re-running setup is **safe**: it merges over the existing `config.toml` and
never wipes hand-edited keys or a saved `cloud.api_key`.

## 4. Install the autostart service

`voiceio setup --defaults` already installs it (unless `install_service:false`).
To manage it explicitly:

```bash
voiceio service install     # systemd --user unit, starts on login, restarts on crash
voiceio service status
```

## 5. Verify end-to-end

```bash
voiceio doctor              # exit 0 = healthy; prints per-backend ✓/✗ and any missing deps
voiceio service status      # expect "running"
```

Expected: `doctor` exits 0, shows a working typer backend (ibus preferred) and a
hotkey backend. A microphone + streaming check is available via `voiceio doctor --live`
(needs a TTY and live audio — skip in headless/agent contexts).

Tell the user: press the hotkey (default **Ctrl+Alt+V**) in any app, speak, then
press it again to commit.

## 6. Troubleshooting (decision tree, keyed on real output)

- **`voiceio: command not found`** → PATH missing `~/.local/bin`. Run
  `pipx ensurepath` then open a new shell, or `export PATH="$HOME/.local/bin:$PATH"`.
- **pip/pipx error building `evdev` / `error: command 'gcc' failed`** → C
  toolchain/headers missing. Install `build-essential python3-dev` (apt) /
  `gcc make python3-devel` (dnf) / `base-devel` (pacman), then reinstall.
- **`PortAudioError` / `OSError: ... portaudio`, or doctor: "no microphone"** →
  install `portaudio19-dev` (apt) / `portaudio-devel` (dnf) / `portaudio`
  (pacman). Confirm a mic exists.
- **doctor: "no text injection backend"** → install `ibus gir1.2-ibus-1.0
  python3-gi` (apt equivalents per distro), then re-run `voiceio setup --defaults`.
- **`[voiceio-setup] step=preflight status=missing`** → run the printed `cmd=`,
  or re-run setup with `--yes`.
- **`[voiceio-setup] step=... status=error reason='...'`** → non-zero exit; read
  the `reason`. `step=validate` = bad `--answers` JSON/keys; `step=model` =
  model download failed (network); `step=system` = no mic (exit 5) or no typer
  backend (exit 6). On `step=system` failures a valid `config.toml` is still
  written (look for `step=config status=written`) — fix the hardware/packages
  and re-run setup; heavy steps (model download, service install) are skipped
  until the system is usable.
- **Hotkey does nothing (GNOME/Wayland, socket backend)** → the DE shortcut may
  not have registered. Check Settings → Keyboard → Custom Shortcuts; command
  must be `voiceio-toggle`. Or enable evdev hotkeys: `voiceio doctor --fix`
  installs `/etc/udev/rules.d/60-voiceio-uaccess.rules` (asks for sudo, works
  without re-login). Fallback: `sudo usermod -aG input $USER` (log out/in).
  Then re-run setup.
- **`sounddevice is not installed` / desktop extra missing** → reinstall with
  `pipx install --force 'python-voiceio[desktop]'`.
- **Nothing types in a terminal** → expected; terminals use the clipboard
  fallback, GUI apps use IBus.

## 7. Where the user's data lives (say this out loud)

Everything stays local: config in `~/.config/voiceio/`, history + retained audio
+ logs in `~/.local/state/voiceio/`. Plain files the user owns. Zero telemetry.
Two optional, off-by-default features send *text* (never audio) to a cloud LLM
the user configures themselves.
