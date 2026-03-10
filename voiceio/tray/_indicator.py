#!/usr/bin/env python3
"""Standalone AppIndicator tray process.

Runs under *system* Python (which has gi + AyatanaAppIndicator3).
Spawned as a subprocess by the main voiceio process.

Left-click opens menu with "Toggle Recording" as the first item.
Middle-click also toggles (via secondary-activate).

Protocol:
  stdin  (parent -> child): "recording\n", "idle\n", "quit\n"
  stdout (child -> parent): "toggle\n" when user clicks toggle

Args:
  --theme-dir PATH       icon theme directory (hicolor structure)
  --idle-icon NAME       icon name for idle state
  --rec-icons NAME,...   comma-separated icon names for recording frames
  --interval MS          animation interval in milliseconds
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme-dir", required=True)
    parser.add_argument("--idle-icons", required=True, help="comma-separated idle icons")
    parser.add_argument("--rec-icons", required=True, help="comma-separated")
    parser.add_argument("--proc-icons", default="", help="comma-separated processing icons")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    idle_icon_names = args.idle_icons.split(",")
    rec_icon_names = args.rec_icons.split(",")
    proc_icon_names = args.proc_icons.split(",") if args.proc_icons else rec_icon_names
    idle_interval = args.interval * 4  # ~500ms — slow, calm cycle

    import gi
    gi.require_version("Gtk", "3.0")

    try:
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3 as AppIndicator3
    except (ValueError, ImportError):
        gi.require_version("AppIndicator3", "0.1")
        from gi.repository import AppIndicator3

    from gi.repository import GLib, Gtk

    indicator = AppIndicator3.Indicator.new_with_path(
        "voiceio",
        idle_icon_names[0],
        AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        args.theme_dir,
    )
    indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)

    recording = [False]

    def _emit_toggle(_=None):
        sys.stdout.write("toggle\n")
        sys.stdout.flush()

    # Build menu
    menu = Gtk.Menu()

    toggle_item = Gtk.MenuItem(label="Start recording")
    toggle_item.connect("activate", _emit_toggle)
    menu.append(toggle_item)

    sep = Gtk.SeparatorMenuItem()
    menu.append(sep)

    quit_item = Gtk.MenuItem(label="Quit voiceio")
    quit_item.connect("activate", lambda _: Gtk.main_quit())
    menu.append(quit_item)

    menu.show_all()
    indicator.set_menu(menu)
    indicator.set_secondary_activate_target(toggle_item)

    # Animation state
    anim_frame = [0]
    anim_source = [None]
    anim_icons = [rec_icon_names]  # current icon set for animation
    anim_interval = [args.interval]

    def _animate() -> bool:
        icons = anim_icons[0]
        frame = anim_frame[0] % len(icons)
        indicator.set_icon_full(icons[frame], "animating")
        anim_frame[0] = frame + 1
        return True

    def _clear_animation():
        """Stop any running animation timer."""
        if anim_source[0] is not None:
            GLib.source_remove(anim_source[0])
            anim_source[0] = None

    def _start_animation(icons, interval, label):
        _clear_animation()
        # Preserve frame index for phase-matched transitions —
        # all states share the same phase per frame, so the wave
        # position stays continuous across state changes.
        anim_icons[0] = icons
        anim_interval[0] = interval
        anim_frame[0] = anim_frame[0] % len(icons)
        _animate()
        anim_source[0] = GLib.timeout_add(interval, _animate)
        toggle_item.set_label(label)

    def _go_idle():
        recording[0] = False
        _start_animation(idle_icon_names, idle_interval, "Start recording")

    def _on_command(cmd: str) -> None:
        cmd = cmd.strip()
        if cmd == "recording":
            recording[0] = True
            _start_animation(rec_icon_names, args.interval, "Stop recording")
        elif cmd == "processing":
            _start_animation(proc_icon_names, args.interval // 2, "Processing...")
        elif cmd == "idle":
            _go_idle()
        elif cmd == "error":
            _go_idle()
            toggle_item.set_label("Error - click to retry")
        elif cmd == "quit":
            Gtk.main_quit()

    def _stdin_reader():
        try:
            for line in sys.stdin:
                GLib.idle_add(_on_command, line)
        except (EOFError, ValueError):
            pass
        GLib.idle_add(Gtk.main_quit)

    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(Gtk.main_quit))

    # Start with idle animation
    _start_animation(idle_icon_names, idle_interval, "Start recording")

    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    Gtk.main()


if __name__ == "__main__":
    main()
