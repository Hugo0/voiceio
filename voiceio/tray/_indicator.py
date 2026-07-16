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

    from gi.repository import Gio, GLib, Gtk

    recording = [False]

    def _emit(cmd):
        def _handler(_=None):
            sys.stdout.write(f"{cmd}\n")
            sys.stdout.flush()
        return _handler

    # Build menu
    menu = Gtk.Menu()

    toggle_item = Gtk.MenuItem(label="Start recording")
    toggle_item.connect("activate", _emit("toggle"))
    menu.append(toggle_item)

    menu.append(Gtk.SeparatorMenuItem())

    correct_item = Gtk.MenuItem(label="Review corrections...")
    correct_item.connect("activate", _emit("menu:correct"))
    menu.append(correct_item)

    history_item = Gtk.MenuItem(label="View history...")
    history_item.connect("activate", _emit("menu:history"))
    menu.append(history_item)

    menu.append(Gtk.SeparatorMenuItem())

    demo_item = Gtk.MenuItem(label="Demo...")
    demo_item.connect("activate", _emit("menu:demo"))
    menu.append(demo_item)

    doctor_item = Gtk.MenuItem(label="Doctor...")
    doctor_item.connect("activate", _emit("menu:doctor"))
    menu.append(doctor_item)

    logs_item = Gtk.MenuItem(label="View logs...")
    logs_item.connect("activate", _emit("menu:logs"))
    menu.append(logs_item)

    menu.append(Gtk.SeparatorMenuItem())

    quit_item = Gtk.MenuItem(label="Quit voiceio")
    quit_item.connect("activate", lambda _: Gtk.main_quit())
    menu.append(quit_item)

    menu.show_all()

    # Animation state
    anim_frame = [0]
    anim_source = [None]
    anim_icons = [rec_icon_names]  # current icon set for animation
    anim_interval = [args.interval]

    indicator = [None]
    ind_serial = [0]

    def _make_indicator() -> None:
        """(Re)create the AppIndicator and register it with the watcher.

        A fresh id per rebuild gives a fresh D-Bus object path, so a
        rebuilt indicator never collides with the half-dead export of
        the previous one.
        """
        old = indicator[0]
        if old is not None:
            old.set_status(AppIndicator3.IndicatorStatus.PASSIVE)
        ind_serial[0] += 1
        ind_id = "voiceio" if ind_serial[0] == 1 else f"voiceio-{ind_serial[0]}"
        ind = AppIndicator3.Indicator.new_with_path(
            ind_id,
            anim_icons[0][anim_frame[0] % len(anim_icons[0])],
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
            args.theme_dir,
        )
        ind.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        ind.set_menu(menu)
        ind.set_secondary_activate_target(toggle_item)
        indicator[0] = ind

    def _animate() -> bool:
        icons = anim_icons[0]
        frame = anim_frame[0] % len(icons)
        indicator[0].set_icon_full(icons[frame], "animating")
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

    _make_indicator()

    # Start with idle animation
    _start_animation(idle_icon_names, idle_interval, "Start recording")

    # Registration watchdog: GNOME Shell's appindicator extension can drop
    # items when it restarts (lock screen, extension reload) — its watcher
    # re-seeks existing items and one D-Bus hiccup during that scan loses
    # the icon for good while this process keeps running. Poll the watcher
    # and rebuild the indicator if our item is gone.
    session_bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

    def _registration_ok() -> bool:
        try:
            reply = session_bus.call_sync(
                "org.kde.StatusNotifierWatcher",
                "/StatusNotifierWatcher",
                "org.freedesktop.DBus.Properties",
                "Get",
                GLib.Variant("(ss)", ("org.kde.StatusNotifierWatcher",
                                      "RegisteredStatusNotifierItems")),
                GLib.VariantType.new("(v)"),
                Gio.DBusCallFlags.NONE,
                2000,
                None,
            )
        except GLib.Error:
            return True  # no watcher (extension off / shell starting) — nothing to fix
        items = reply.unpack()[0]
        return any("voiceio" in item for item in items)

    def _check_registration() -> bool:
        if not _registration_ok():
            _make_indicator()
        return True

    GLib.timeout_add_seconds(60, _check_registration)

    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    Gtk.main()


if __name__ == "__main__":
    main()
