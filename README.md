# voiceio

[![CI](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml/badge.svg)](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![Python](https://img.shields.io/pypi/pyversions/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Voice dictation for Linux that learns you.** Local, private, open source, and it fine-tunes itself on your own voice.

[![voiceio: dictation that learns you](https://voiceio.dev/og.png)](https://voiceio.dev)

**[Try it in your browser at voiceio.dev](https://voiceio.dev)**: no install, the model runs in the tab.

- **Types anywhere.** Press your hotkey, speak, and the words stream into the focused app with a live underlined preview (IBus), on Wayland and X11: GNOME, KDE, Hyprland, sway, i3.
- **Runs on your machine.** Whisper, or NVIDIA's Parakeet (text lands ~0.7 s after you stop talking instead of ~4.4 s, with every name you use as a hotword), decodes your speech locally. No account, no server, no telemetry; logs never contain your words.
- **Learns your words.** Keep your recordings (opt-in, on your disk) and voiceio fine-tunes its model on them while the machine is idle. It switches only when the new model wins on clips it never trained on. On the author's voice: 12.1% → 4.9% word errors after one night on a laptop CPU.

## Install

**With a coding agent.** Paste this into Claude Code, Codex or any agent with a shell:

```text
Install voiceio on this machine following https://voiceio.dev/llms.txt. Ask me which hotkey I want and whether to keep my recordings on this computer so voiceio can learn my voice. Finish with voiceio doctor and fix whatever it reports.
```

**By hand:**

```bash
sudo apt install pipx build-essential python3-dev portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi   # Debian/Ubuntu
sudo dnf install pipx gcc gcc-c++ make python3-devel portaudio-devel ibus ibus-libs python3-gobject   # Fedora
sudo pacman -S python-pipx base-devel portaudio ibus python-gobject                                   # Arch

pipx install 'python-voiceio[desktop]'
voiceio setup        # model, hotkey, autostart
```

**Or as a Debian/Ubuntu package** (new; bundles everything, nothing to compile):

```bash
# Debian / Ubuntu: download voiceio_*.deb from the latest release, then
sudo apt install ./voiceio_*.deb

voiceio setup
```

Then press your hotkey, speak, and press it again (or hold it while you speak). `voiceio doctor` shows what works, `voiceio doctor --fix` repairs what it can, and `voiceio doctor --report` shares an anonymous install report on the [feedback board](https://voiceio.dev/#feedback) (it shows you the text and asks first). NixOS, other extras and installing from source: [docs/linux.md](docs/linux.md). Agent runbook: [INSTALL.md](INSTALL.md).

## Make it yours

```bash
voiceio vocab add Kalshi              # a name, recognized from the next dictation
voiceio corrections add kalchi Kalshi # fix whatever still comes out wrong
voiceio learn schedule on             # learn from your recordings while the machine is idle
voiceio learn status                  # what it learned and how each model scores
```

Scheduled learning runs daily or weekly at a time you pick, on mains power and only while you're away from the computer: it stops within seconds when you come back and resumes later where it was. You get a notification with a button to stop it. It never switches models unless the new one wins, and `voiceio learn rollback` undoes a switch. Details: [docs/learning.md](docs/learning.md).

## Let your agents talk to you

```bash
voiceio say --from build-bot "tests are green"     # chime, notification, then it's read aloud
voiceio say --from deploy --urgent "I need your approval"
```

Running several coding agents? Have each one call `voiceio say` when it needs you, so you hear *which* one. It never talks over you: while you dictate, announcements wait (speech into an open mic would get typed). They are read in your [tts] voice, offline.

## Configure

Everything lives in `~/.config/voiceio/config.toml`; [config.example.toml](config.example.toml) documents every option. `voiceio config set section.key value` edits it in place.

## Docs

| | |
|---|---|
| [Learning your voice](docs/learning.md) | Vocabulary, corrections, labeling, fine-tuning, evaluation, schedule |
| [Choosing a model](docs/models.md) | Whisper sizes, Parakeet, whisper.cpp servers, licenses |
| [Commands](docs/commands.md) | Every `voiceio` command |
| [Your data](docs/data.md) | What is stored where, and what is off by default |
| [Phone dictation](docs/phone.md) | Run voiceio headless on a home server |
| [Linux support and troubleshooting](docs/linux.md) | Desktops, backends, NixOS, fixes |
| [Comparisons](https://voiceio.dev/vs/) | voiceio vs Voxtype, Wispr Flow, Superwhisper, Dragon and others |

## Roadmap

- **Parakeet by default for everyone**: it needs our sherpa-onnx fix ([#4084](https://github.com/k2-fsa/sherpa-onnx/pull/4084)) in an upstream release; until then `voiceio models sherpa && voiceio models use parakeet-tdt-0.6b-v2` installs a patched build ([why](CONTRIBUTING.md#model-choice-parakeet-where-it-can-be-biased-whisper-otherwise)).
- **GPU on AMD and Intel**: let setup run a Vulkan whisper.cpp server for you.
- **Typing without IBus on GNOME and KDE**: a libei backend.
- **Distro packages**: AUR (packaged, not yet published) and .rpm.
- **One model everywhere**: sync your fine-tune across your machines.

## Contributing

Open an issue before a large PR. [CONTRIBUTING.md](CONTRIBUTING.md) covers the architecture, conventions and the reasoning behind the model choice. Install reports and ideas are welcome on the [feedback board](https://voiceio.dev/#feedback).

## Credits

- Speech recognition: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), with OpenAI's [Whisper](https://github.com/openai/whisper) models; [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with NVIDIA's [Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). The in-browser demo runs [Moonshine](https://github.com/usefulsensors/moonshine) through [transformers.js](https://github.com/huggingface/transformers.js).
- Ideas borrowed with thanks: modifier release, paste keys and media pausing from [Voxtype](https://github.com/peteonrails/voxtype); the landing page's look from Romario Kavin's [KOLlateral](https://github.com/RomarioKavin1/kollateral).

## License

MIT. Model weights are not part of voiceio and carry their own licenses (see [docs/models.md](docs/models.md)).
