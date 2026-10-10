# voiceio

[![CI](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml/badge.svg)](https://github.com/Hugo0/voiceio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![Python](https://img.shields.io/pypi/pyversions/python-voiceio)](https://pypi.org/project/python-voiceio/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Voice dictation that learns you.** ([website](https://hugo0.github.io/voiceio/)) Local, open source, private — and it fine-tunes itself on your own voice, on your own machine.

Off-the-shelf speech models stumble on exactly the words you say most: your colleagues' names, your project, your jargon. voiceio keeps (with your permission) the audio of what you dictate, has a bigger model transcribe it overnight, and trains a personal model from it — nothing uploaded, no GPU required. On the author's dictation, one overnight run on a laptop CPU cut word errors from **12.1% to 5.0%** on held-out clips and recognized more of the author's vocabulary, at the same speed. Every model is scored on clips it never trained on, and it only replaces the current one when it wins.

https://github.com/user-attachments/assets/9cf5d1ac-b4bb-4cf8-b775-7a66dc16b376

Press a hotkey, speak, and the text streams into whatever app has focus. Press again to commit.

- **Learns you.** `voiceio learn` fine-tunes a model on your own voice; a vocabulary steers it toward new names right away; corrections fix the rest. See [Make it yours](#make-it-yours).
- **Local.** Speech is decoded on-device ([faster-whisper](https://github.com/SYSTRAN/faster-whisper) by default). Audio never leaves the machine and there is no telemetry. Two optional, off-by-default features send *text* (never audio) to an LLM endpoint you configure: final-transcript polish (`[postcorrect]`) and an optional local Ollama cleanup (`[llm]`).
- **Yours.** History, retained audio, corrections, vocabulary and fine-tuned models are plain local files. See [Your data](#your-data).
- **Linux-first.** Wayland and X11, GNOME, KDE, Hyprland, sway and i3. Backends are picked by probing what works. See [Linux support](#linux-support).


> **Upgrading from 0.x?** See [UPGRADING.md](UPGRADING.md) — 1.0 renames a few commands and moves desktop dependencies to the `[desktop]` extra.

## Quickstart

```bash
# 1. System packages: a C toolchain (evdev has no wheel), PortAudio, IBus + GObject bindings
sudo apt install pipx build-essential python3-dev portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi   # Debian/Ubuntu
sudo dnf install pipx gcc gcc-c++ make python3-devel portaudio-devel ibus ibus-libs python3-gobject   # Fedora
sudo pacman -S python-pipx base-devel portaudio ibus python-gobject                                   # Arch

# 2. voiceio with the desktop extra (microphone, hotkeys, tray)
pipx install 'python-voiceio[desktop]'

# 3. Guided setup: model download, hotkey, autostart
voiceio setup
```

Then press **Ctrl+Alt+V**, speak, and press it again. `voiceio doctor` tells you what is working and `voiceio doctor --fix` repairs what it can.

NixOS users: see [NixOS](#nixos). Agents installing voiceio non-interactively: see [INSTALL.md](INSTALL.md).

<details>
<summary><strong>Extras</strong></summary>

The base package (`pip install python-voiceio`) is headless transcription only: no microphone, hotkeys or display needed.

| Extra | Adds |
|---|---|
| `desktop` | The dictation app: microphone (sounddevice), hotkeys (evdev, pynput), tray icon |
| `server` | HTTP/WebSocket transcription server: `python -m voiceio.server` |
| `tts` | Text-to-speech: piper (local), edge-tts (cloud) |
| `sherpa` | sherpa-onnx runtime for non-Whisper models (experimental) |
| `train` | `voiceio learn train`: LoRA fine-tuning (torch, transformers, peft) |

</details>

<details>
<summary><strong>From source</strong></summary>

```bash
git clone https://github.com/Hugo0/voiceio && cd voiceio
uv sync --extra desktop --extra dev
uv run voiceio setup      # also links voiceio into ~/.local/bin
```

</details>

## Commands

> **Note:** the CLI is being restructured for v1.0. This list is the target layout. Check it against `voiceio --help` before release.

| Command | What it does |
|---|---|
| `voiceio run` | Start the dictation daemon in the foreground (the systemd unit runs this) |
| `voiceio toggle` | Start/stop recording on the running daemon. Bind it to a key if you don't use evdev |
| `voiceio setup` | Guided setup (`--defaults` / `--answers '{json}'` for scripts) |
| `voiceio doctor [--fix]` | Check every backend; `--fix` repairs what it can, asking before anything that needs sudo |
| `voiceio service install\|uninstall\|start\|stop\|restart\|status\|logs` | systemd user unit: start on login, restart on crash; `logs -f` follows |
| `voiceio models list\|bench\|use` | See, compare and pick speech models ([Choose a model](#choose-a-model)) |
| `voiceio vocab` | Manage the vocabulary that steers the decoder |
| `voiceio corrections` | Manage find/replace rules for words it still gets wrong |
| `voiceio learn …` | Personalize: label → review → augment → build → eval → train → promote |
| `voiceio history` | Search what you have dictated |
| `voiceio config` | Show or set config values |
| `voiceio uninstall` | Remove the service, IBus component, shortcuts and symlinks (asks before deleting your data) |

## Your data

Everything voiceio keeps is in three directories (they follow `XDG_CONFIG_HOME`, `XDG_STATE_HOME` and `XDG_DATA_HOME`). Files are created `0600`, directories `0700`. Nothing is uploaded. Recordings, decode traces and window titles are **off unless you turn them on** — `voiceio setup` asks whether to keep recordings so voiceio can learn your voice. Logs record sizes and timings, never your words.

| Path | Holds | Retention |
|---|---|---|
| **`~/.config/voiceio/`** | *Settings and what it has learned about your words* | |
| `config.toml` | All settings | — |
| `vocabulary.txt` | Your terms, one per line, used as decoder hotwords | Never pruned; rarely used terms rank lower |
| `vocab_stats.json` | Per-term usage counts used to rank the vocabulary | Rebuilt as you dictate |
| `corrections.json` | Find/replace rules (`post grass` → `PostgreSQL`) | — |
| `flagged.txt` | Words flagged with the "correct that" voice command, for review | — |
| `snapshots/` | Copies of corrections + vocabulary from before 1.0 (safe to delete) | — |
| `consent.json` | Whether you allowed cloud LLM calls (text only) | — |
| `hints.json` | Which CLI tips you have already seen | — |
| **`~/.local/state/voiceio/`** | *What you said* | |
| `history.jsonl` | Every committed transcript: time, raw and final text (and the focused-window title with `[data] capture_context`) | `[history] enabled`, `max_entries`, `max_age_days` (0 = keep forever) |
| `recordings/*.wav` | The audio of each utterance, for `voiceio learn` — **opt-in** | `[data] retain_audio` (default off), `max_audio_mb` (default 4096), `min_free_gb` (default 5) |
| `streaming_trace.jsonl`, `postcorrect_pairs.jsonl` | Per-pass decode text and timings, LLM before/after pairs — **opt-in**, for debugging | `[data] capture_intermediates` (default off); each capped at 64 MB |
| `voiceio.log`, `worker.log`, `ibus-engine-stderr.log` | Logs | `voiceio.log` rotates at 2 MB × 3 |
| **`~/.local/share/voiceio/learn/`** | *Personalization corpus* (only exists once you run `voiceio learn`) | Never pruned; delete freely |
| `labels.jsonl` | Reference transcripts from the teacher model or your review | — |
| `augment/` | Synthetic TTS clips of your hard words | — |
| `datasets/vNNNN/` | Frozen train/dev/test splits | — |
| `exports/` | Datasets packed for training on another machine | — |
| `models/<run>/` | Your fine-tuned models, plus `promotions.jsonl` | — |
| `runs/` | Evaluation results | — |

Set `[data] capture_context = false` to stop recording window titles. `voiceio uninstall` offers to delete the config and state directories; delete `~/.local/share/voiceio` yourself.

## Make it yours

Most dictation errors are proper nouns and jargon: `PostgreSQL` heard as `post grass`, a colleague's name spelled three ways. There are three tools for that, from cheapest to most powerful.

**1. Vocabulary.** Put your names and terms in `~/.config/voiceio/vocabulary.txt` (or use `voiceio vocab`). They are passed to Whisper as *hotwords* on every decode, ranked by how often you use them. One honest limit: Whisper's hotword channel is capped at 223 tokens, and a proper noun costs 5 to 7 tokens, so only about **35 terms** steer each decode. The rest are kept and ranked, but they do nothing until they rank high enough. This ceiling comes from the model, not from voiceio. [CONTRIBUTING.md](CONTRIBUTING.md#model-choice-why-whisper-and-what-would-change-it) explains why we still use Whisper.

**2. Corrections.** Find/replace rules for what still comes out wrong (`voiceio corrections add WRONG RIGHT`, or say "correct that" right after a mistake and review with `voiceio corrections flagged`). `voiceio vocab suggest` lists unusual words from your history — offline — so you can add the real names to your vocabulary and turn the mishearings into corrections.

**3. Fine-tuning with `voiceio learn`.** This is how you get past the 35-term ceiling: teach the model your vocabulary directly, from your own retained audio.

```bash
voiceio learn import DIR --source phone   # optional: add your own recordings from elsewhere
voiceio learn label      # a stronger teacher model (large-v3-turbo) transcribes your clips
voiceio learn review     # fix the teacher where it matters: your names and terms
voiceio learn review --test   # verify the held-out clips, so eval scores against the truth: it asks only
                              #   where the teacher and your live decode disagree, quickest clips first
voiceio learn augment    # synthesize extra examples of rare terms with local TTS
voiceio learn build      # freeze a versioned train/dev/test dataset (the test split never trains)
voiceio learn eval       # score the current model on your held-out clips: WER and term recall
                         #   --streaming replays them through live dictation and times stop→text
voiceio learn train      # LoRA fine-tune (needs the `train` extra; decoder-only on CPU, --full on a GPU)
voiceio learn promote <run>   # dictate with the fine-tune; `voiceio learn rollback` undoes it
voiceio learn maintain   # later: label what's new, and once 50+ clips piled up, retrain and compare
voiceio learn schedule on   # …or let it run `maintain` every night, and notify you when a model wins
                            #   (--auto-promote: switch to a winner by itself; `learn rollback` undoes it)
```

Labeling, training and offline eval run in an idle-priority cgroup, so they never slow dictation down. Everything stays on your machine unless you choose `train --remote`.

`voiceio learn status` shows where you are, each model's scores, and suggests `promote` only when a fine-tune wins on everything. `promote` itself refuses a model that hasn't, and reports how many clips got better or worse with a confidence interval — a few dozen test clips can move an average by luck.

**Is it actually helping?** Once some clips are labeled, measure instead of guessing:

```bash
voiceio learn eval --pipeline   # what each text stage (numbers, commands, corrections, cloud
                                # postcorrect…) did to your past dictations, and its commonest edits
voiceio corrections audit       # rules your own audio contradicts, and ones that look risky
voiceio learn eval --streaming  # accuracy *and* stop→text, replaying clips through live dictation
```

On the author's data these found a correction that kept turning "composable" into "comparable", a number rule writing "no one" as "no 1", and a cloud pass that never fixed a single vocabulary term.

## Choose a model

```bash
voiceio models list           # known models, your fine-tunes, and which is active
voiceio models bench          # score candidates on your held-out clips (after `voiceio learn build`)
voiceio models use medium     # switch
```

| Model | Runtime | License | Notes |
|---|---|---|---|
| `small` | faster-whisper | MIT | Default. About 5× realtime on a laptop CPU |
| `medium` | faster-whisper | MIT | Better on names, about 2.5× slower |
| `large-v3-turbo` | faster-whisper | MIT | Best Whisper for dictation; wants a GPU. Also the default `learn` teacher |
| `distil-large-v3` | faster-whisper | MIT | English only |
| `sherpa:parakeet-tdt-0.6b-v3` | sherpa-onnx | CC-BY-4.0 | Experimental. Fast, but no vocabulary biasing yet |
| `whispercpp:http://…` | whisper.cpp server | depends on the model | Experimental. E.g. a Vulkan build on an AMD iGPU |

Pick on your own audio, not leaderboards: for dictation full of names, how well a model can be *biased* matters more than its benchmark WER.

**Licenses.** voiceio ships no model weights. Each model is downloaded from its own source on first use, under its own license. The defaults are permissive (MIT). A fine-tune you make with `voiceio learn` inherits the license of the base model it was trained from.

## Dictate from your phone

`python -m voiceio.server` (the `server` extra) serves the same engine over HTTP and WebSocket, for a phone or any other client: `POST /transcribe` with an audio file, or stream PCM over `/ws/transcribe` for live partials. It binds `127.0.0.1` by default; put it behind your VPN or a reverse proxy and set `VOICE_TOKEN` (bearer token) and `VOICE_ALLOWED_ORIGINS` when it is reachable from elsewhere. `VOICE_MODEL`, `VOICE_LANES`, `VOICE_HOST`/`VOICE_PORT` override the rest.

The server reads its own machine's voiceio config, so it gets your vocabulary and corrections once you copy them over:

```bash
scripts/sync_personalization.sh myserver           # vocabulary + corrections
scripts/sync_personalization.sh myserver --model   # …and your promoted fine-tune
```

To fine-tune on what you dictate from the phone too, opt in on the server with `[data] retain_audio = true` in its `config.toml`, then bring the clips home: `scripts/pull_phone_dictation.sh myserver` (an `rsync` over SSH followed by `voiceio learn import`). Only your own voice should go in: the model learns whoever is speaking.

## Linux support

| | Text injection | Streaming preview | Hotkey | Status |
|---|---|---|---|---|
| GNOME (Wayland) | IBus, ydotool fallback | Yes (IBus preedit) | evdev, or a GNOME shortcut set up by `setup` | Tested daily |
| GNOME (X11) | IBus, xdotool fallback | Yes | evdev, pynput | Supported |
| KDE Plasma (Wayland) | IBus, ydotool fallback | Yes with IBus | evdev | Should work |
| Hyprland, incl. Omarchy | IBus, else wtype → ydotool | Yes with IBus; with fcitx5 (Omarchy's default), words are typed as you speak, without the in-place preview | evdev, or `bind = CTRL ALT, V, exec, voiceio toggle` | Should work |
| sway | IBus, else wtype → ydotool | As Hyprland | evdev, or `bindsym` + `voiceio toggle` | Should work |
| i3 / other X11 | IBus, xdotool | Yes with IBus | pynput, evdev | Should work |
| NixOS | as above | as above | evdev via the flake's udev rule | Flake provided, unbuilt ([NixOS](#nixos)) |

**Input methods.** IBus is the only way to get a live, underlined preview that is corrected in place on Wayland. voiceio installs its own IBus engine and only activates it while you dictate. If fcitx5 is your input method, voiceio leaves it alone and types the final text with wtype or ydotool. Terminals that ignore input methods get the clipboard.

**Hotkeys without the `input` group.** The evdev backend reads keyboards from `/dev/input`. `voiceio doctor --fix` installs a udev rule, `/etc/udev/rules.d/60-voiceio-uaccess.rules`, that grants the logged-in user access to keyboard devices. It works immediately, without logging out. The old route still works as a fallback: `sudo usermod -aG input $USER`, then log out and back in. You can also skip evdev and bind `voiceio toggle` in your compositor.

**ydotool** needs write access to `/dev/uinput`; `voiceio doctor --fix` installs a uaccess rule for that too.

### NixOS

The repository is a flake. **It has not been built yet**; please report fixes.

```nix
# flake inputs: voiceio.url = "github:Hugo0/voiceio";
# NixOS: package, uinput, and the keyboard uaccess rule
imports = [ voiceio.nixosModules.default ];
programs.voiceio.enable = true;

# Home Manager: config.toml from Nix + a systemd user service
imports = [ voiceio.homeManagerModules.default ];
services.voiceio = {
  enable = true;
  settings.model.name = "small";
};
```

For the streaming preview, enable IBus as your input method: `i18n.inputMethod = { enable = true; type = "ibus"; };`.

### Windows and macOS

There is a best-effort pynput path for both (hotkeys and typing, no live preview). It is untested and unmaintained, with no installers or CI. Patches are welcome; bug reports will likely sit.

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing is typed | `voiceio doctor --fix`; usually the IBus component or the GNOME input source |
| Hotkey does nothing on Wayland | `voiceio doctor --fix` installs the keyboard udev rule, or bind `voiceio toggle` in your compositor |
| `sounddevice is not installed` | You installed without the desktop extra: `pipx install --force 'python-voiceio[desktop]'` |
| evdev fails to build | Install a C toolchain and Python headers (see Quickstart), then reinstall |
| Too slow | `voiceio doctor` shows your real stop→text. Turn off cloud postcorrect; use `voiceio models use small` (or a GPU with `large-v3-turbo`); or narrow the beam of the passes you keep: on the author's 44 clips `[model] beam_size = 2` matched the default 5 (WER 4.9% vs 5.4%, vocabulary 28/32 vs 29/32) while decoding faster; greedy (`1`) lost vocabulary (26/32). Measure on yours: `voiceio learn eval --beam 2` |
| Words come out wrong in a pattern | `voiceio learn eval --pipeline` and `voiceio corrections audit` show which stage or rule did it |
| Start over | `voiceio uninstall`, then `voiceio setup` |

Logs: `journalctl --user -u voiceio` or `~/.local/state/voiceio/voiceio.log`.

## Roadmap

- **Unbounded vocabulary** — move to a decoder whose biasing has no token cap (see [model choice](CONTRIBUTING.md#model-choice-why-whisper-and-what-would-change-it)).
- **One model everywhere** — sync your fine-tune to your phone server and laptops, peer to peer.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture, conventions and the reasoning behind the model choice. Please open an issue before a large PR.

## License

MIT. Model weights are not part of voiceio and carry their own licenses (see [Choose a model](#choose-a-model)).
