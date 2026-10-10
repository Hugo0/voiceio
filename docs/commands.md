# Commands

> **Note:** the CLI is being restructured for v1.0. This list is the target layout. Check it against `voiceio --help` before release.

| Command | What it does |
|---|---|
| `voiceio run` | Start the dictation daemon in the foreground (the systemd unit runs this) |
| `voiceio toggle [--press\|--release]` | Start/stop recording on the running daemon. Bind it to a key if you don't use evdev; bind `--press` and `--release` (Hyprland `bindr`, sway `--release`) for push-to-talk |
| `voiceio cancel` | Abandon the dictation in progress: nothing is typed, saved to history or kept as audio. Bind it to a key, or set `[hotkey] cancel_key`
| `voiceio setup` | Guided setup (`--defaults` / `--answers '{json}'` for scripts) |
| `voiceio doctor [--fix]` | Check every backend; `--fix` repairs what it can, asking before anything that needs sudo |
| `voiceio service install\|uninstall\|start\|stop\|restart\|status\|logs` | systemd user unit: start on login, restart on crash; `logs -f` follows |
| `voiceio models list\|bench\|use` | See, compare and pick speech models ([Choosing a model](models.md)) |
| `voiceio vocab` | Manage the vocabulary that steers the decoder |
| `voiceio corrections` | Manage find/replace rules for words it still gets wrong |
| `voiceio learn …` | Personalize: label → review → augment → build → eval → train → promote |
| `voiceio history` | Search what you have dictated |
| `voiceio paste-last` | Type the last dictation again into the focused app (through the running daemon's typer) |
| `voiceio copy-last` | Copy the last dictation to the clipboard |
| `voiceio config` | Show or set config values |
| `voiceio uninstall` | Remove the service, IBus component, shortcuts and symlinks (asks before deleting your data) |
