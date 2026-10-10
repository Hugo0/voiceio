# Your data

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
