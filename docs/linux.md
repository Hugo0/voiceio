# Linux support and troubleshooting

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

**Tap or hold.** With `[hotkey] mode = "auto"` (the default) a quick tap starts recording and the next tap stops it; holding the key for `hold_ms` (1 s) or longer is push-to-talk; anything shorter counts as a tap, so a relaxed press never stops a dictation on release, and recording stops `release_tail_ms` (250 ms) after you let go. `"hold"` is always push-to-talk, `"toggle"` never. Releases are seen by evdev and pynput. A compositor shortcut running `voiceio toggle` fires on key down only, so it toggles, unless you bind the release too: Hyprland `bind = CTRL ALT, V, exec, voiceio toggle --press` plus `bindr = CTRL ALT, V, exec, voiceio toggle --release`; sway `bindsym Ctrl+Alt+v exec voiceio toggle --press` plus `bindsym --release Ctrl+Alt+v exec voiceio toggle --release`.

**ydotool** needs write access to `/dev/uinput`; `voiceio doctor --fix` installs a uaccess rule for that too.

### Distro packages

The `.deb` (attached to every [GitHub release](https://github.com/Hugo0/voiceio/releases) from 1.11.0) and the AUR package `python-voiceio` (coming soon) install the same thing: voiceio with every extra (desktop, TTS) and its own Python in `/usr/lib/voiceio`, plus `/usr/bin/voiceio`. Nothing is compiled on your machine and a distro `python3` upgrade can't break it. The distro's `python3` is still used for the IBus engine and the tray icon, which need its GObject bindings.

```bash
sudo apt install ./voiceio_*.deb      # Debian 12+ / Ubuntu 22.04+, amd64
yay -S python-voiceio                 # Arch, once it's on the AUR
voiceio setup                         # as your user, as with pipx
```

`voiceio setup` writes the same per-user files as a pipx install (config, the IBus component, the systemd user service); the service runs `/usr/lib/voiceio/bin/voiceio`, a path that stays put across upgrades. After an upgrade, `systemctl --user restart voiceio` picks up the new version. To remove: `voiceio uninstall`, then `sudo apt remove voiceio` or `sudo pacman -R python-voiceio`. GPU (CUDA) decoding isn't set up by the packages; use pipx for that. How they are built: [packaging/README.md](../packaging/README.md).

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
| Pastes nothing (or `^V`) in a terminal | Automatic on Hyprland, sway, niri and X11: a focused terminal gets `ctrl+shift+v` (`output.terminal_paste_keys`). GNOME and KDE on Wayland don't say which app has focus: `voiceio config set output.paste_keys ctrl+shift+v` |
| Words come out wrong in a pattern | `voiceio learn eval --pipeline` and `voiceio corrections audit` show which stage or rule did it |
| Start over | `voiceio uninstall`, then `voiceio setup` |

Logs: `journalctl --user -u voiceio` or `~/.local/state/voiceio/voiceio.log`.
