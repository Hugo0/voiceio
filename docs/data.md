# Your data

Everything voiceio keeps is in three directories (they follow `XDG_CONFIG_HOME`, `XDG_STATE_HOME` and `XDG_DATA_HOME`). Files are created `0600`, directories `0700`. Nothing is uploaded, except the anonymous weekly count below if you turn it on. Recordings, decode traces and window titles are **off unless you turn them on** — `voiceio setup` asks whether to keep recordings so voiceio can learn your voice. Logs record sizes and timings, never your words.

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
| `ping.json` | When the anonymous weekly count was last sent (only with `[telemetry] ping = true`) | One timestamp |
| **`~/.local/share/voiceio/learn/`** | *Personalization corpus* (only exists once you run `voiceio learn`) | Never pruned; delete freely |
| `labels.jsonl` | Reference transcripts from the teacher model or your review | — |
| `augment/` | Synthetic TTS clips of your hard words | — |
| `datasets/vNNNN/` | Frozen train/dev/test splits | — |
| `exports/` | Datasets packed for training on another machine | — |
| `models/<run>/` | Your fine-tuned models, plus `promotions.jsonl` | — |
| `runs/` | Evaluation results | — |

The microphone is open only while voiceio is warm: always with `[audio] keep_warm = "always"`, on mains power or within `warm_minutes` (default 5) of your last dictation on battery with `"auto"` (the default), and only while dictating with `"never"`. While open it feeds a 1-second in-memory pre-buffer that is discarded unless you start a dictation; after a cold open there is no pre-buffer, so the first syllable may be clipped.

Set `[data] capture_context = false` to stop recording window titles. `voiceio uninstall` offers to delete the config and state directories; delete `~/.local/share/voiceio` yourself.

## The anonymous weekly count (opt-in)

*Not switched on yet: until its server, ping.voiceio.dev, is live, voiceio never asks and never sends it.*

`voiceio setup` (or, for an install that was never asked, `voiceio doctor`) asks once whether to send an anonymous weekly count to voiceio.dev's own server (`https://ping.voiceio.dev/ping`), so we know how many people use voiceio. It is off unless you say yes. When on, the running daemon sends at most one request every 7 days with exactly five fields, for example `{"v": "1.18.7", "distro": "ubuntu 26.04", "desktop": "gnome", "session": "wayland", "decoder": "parakeet"}`: the voiceio version, distro, desktop, session type and the kind of speech model (never its name or path). There is no install ID, hostname or username, and nothing about what you say. The server adds one to a per-week counter for those five values and keeps nothing else: no IP address, no headers, no time finer than the week. The totals are public at `https://ping.voiceio.dev/ping/stats`. `voiceio doctor` shows whether it is on and the exact request; turn it off with `voiceio config set telemetry.ping false`. The server code is in `workers/ping/`.
