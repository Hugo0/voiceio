# Upgrading to voiceio 1.0

1.0 is a breaking release. Most setups keep working without changes; these are
the things to know.

## Install

Desktop dependencies (microphone, hotkeys, tray) moved to an extra:

```bash
pipx install --force 'python-voiceio[desktop]'      # or: uv tool install 'python-voiceio[desktop]'
```

`[linux]` still works as an alias. A headless server needs only `[server]`.

## Commands

| Before | Now |
|---|---|
| `voiceio` (start the daemon) | `voiceio run` — bare `voiceio` prints help |
| `voiceio correct "wrong" "right"` | `voiceio corrections add wrong right` |
| `voiceio correct --list / --remove / --flagged` | `voiceio corrections list / remove / flagged` |
| `voiceio correct --vocab` | `voiceio vocab show` |
| `voiceio correct --auto` | removed — see below |
| `voiceio test` | `voiceio doctor --live` |
| `voiceio logs` | `voiceio service logs [-f]` |
| `voiceio eval` | `voiceio models bench` / `voiceio learn eval` |
| `voiceio update` | your package manager |
| `voiceio-doctor`, `voiceio-setup`, `voiceio-test` | `voiceio doctor`, `voiceio setup`, `voiceio doctor --live` |

`voiceio-toggle` still exists, because GNOME shortcuts created before 1.0 call
it. Re-running `voiceio setup` switches the shortcut to `voiceio toggle`.

Run `voiceio service install` once to rewrite the systemd unit for
`voiceio run`. Until then, the old unit (bare `voiceio`) keeps working.

## Cloud mining is gone

The weekly `voiceio-correct.timer` asked a cloud LLM to mine your history for
correction rules. Measured over months, it produced 387 rules of which 4 ever
fired, and it added vocabulary without review. It is removed. `voiceio run`
deletes the old timer. In its place:

- `voiceio vocab suggest`: unusual words from your history, offline
- `voiceio learn`: fine-tune recognition on your own voice

## Defaults that changed

- Two decoder processes (`model.decode_lanes = 2`), so stopping never waits
  on a preview. That costs about 300 MB extra for `small`. Set it to `1` on
  low-memory machines.
- Long dictations are finalized in 8 s chunks (was 15 s).
- `[postcorrect]` (the optional cloud polish) times out after 3 s (was 8 s)
  and rejects edits that change numbers, add formatting, or delete content
  words.
- Transcript text is no longer written to the log at INFO level.

## Removed options

`output.min_recording_secs`, `output.cancel_window_secs` and
`autocorrect.mine_corrections` are ignored if present.

## Test data in your history

If you ever ran voiceio's test suite before 1.0, it may have written
"Hello world" rows into your history and traces. `voiceio doctor` tells you
whether it did. `voiceio learn clean --apply` removes them, with backups.
