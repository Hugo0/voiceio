#!/usr/bin/env python3
"""VoiceIO IBus engine: receives commands via Unix socket, injects text via IBus.

Run as a standalone process:
    python3 -m voiceio.ibus.engine

Architecture:
    - GLib main loop drives the IBus engine (required by IBus).
    - Socket listener thread receives commands from voiceio daemon.
    - Commands are dispatched to the engine via GLib.idle_add() for thread safety.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import socket
import sys
import threading
from pathlib import Path

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, GObject, IBus

from voiceio.ibus import (
    READY_PATH,
    SOCKET_PATH,
    acquire_singleton_lock,
)
from voiceio.ibus.pending import PendingBuffer
from voiceio.ibus.textlimit import split_utf8, tail_utf8

log = logging.getLogger(__name__)

# Singleton lock fd, held for the process lifetime (see main()). Module-level so
# it is never garbage-collected — closing the fd would release the lock.
_lock_fd: int | None = None
ENGINE_NAME = "voiceio"
COMPONENT_NAME = "org.voiceio.ibus"
# A datagram longer than the buffer we hand recvfrom() is truncated silently,
# which would inject half a dictation with no error anywhere. Sized above the
# largest datagram a sender can produce (SO_SNDBUF, ~208K by default) so that
# cannot happen.
RECV_BYTES = 262144
# Between commit chunks: long enough for the compositor to flush the previous
# one to the app, short enough that a 10KB note lands in a blink.
COMMIT_GAP_MS = 40


class VoiceIOEngine(IBus.Engine):
    """IBus engine that receives text injection commands via socket."""

    __gtype_name__ = "VoiceIOEngine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._focused = False
        log.info("VoiceIOEngine instance created (path=%s)", kwargs.get("object_path"))

    def do_focus_in(self):
        # Focus transitions also discard any visible preedit (IBus default
        # mode), so log them — a flickering preedit is usually focus churn.
        log.debug("focus_in")
        self._focused = True

    def do_focus_out(self):
        log.debug("focus_out (preedit discarded by client if visible)")
        self._focused = False

    def do_process_key_event(self, keyval, keycode, state):
        # CRITICAL: Always pass all keys through. Never intercept typing.
        # If this ever returns True (or raises), ALL keyboard input dies system-wide.
        try:
            return False
        except Exception:
            return False

    def preedit(self, text: str) -> None:
        """Show text as preedit (underlined preview), tail-first once long."""
        if not text:
            self.hide_preedit_text()
            return
        shown = tail_utf8(text)
        ibus_text = IBus.Text.new_from_string(shown)
        ibus_text.append_attribute(
            IBus.AttrType.UNDERLINE,
            IBus.AttrUnderline.SINGLE,
            0,
            len(shown),
        )
        self.update_preedit_text(ibus_text, len(shown), True)

    def commit(self, text: str) -> None:
        """Clear preedit and commit final text, in size-capped chunks.

        Each chunk after the first goes out on a later main-loop turn. The
        compositor holds a commit as pending until it sends `done` from its
        own idle handler, and a second commit before that *replaces* the
        first — committing chunks back to back delivered only the last one
        (a long dictation pasted as its final ~2KB).
        """
        self.hide_preedit_text()
        chunks = split_utf8(text)
        if not chunks:
            return
        self.commit_text(IBus.Text.new_from_string(chunks[0]))
        rest = iter(chunks[1:])

        def next_chunk() -> bool:
            chunk = next(rest, None)
            if chunk is None:
                return False
            self.commit_text(IBus.Text.new_from_string(chunk))
            return True

        if len(chunks) > 1:
            GLib.timeout_add(COMMIT_GAP_MS, next_chunk)

    def clear(self) -> None:
        """Clear preedit without committing."""
        self.hide_preedit_text()


class VoiceIOEngineFactory(IBus.Factory):
    """Custom factory that creates engine instances with proper D-Bus object paths."""

    __gtype_name__ = "VoiceIOEngineFactory"
    _engine_count = 0

    def __init__(self, bus):
        self._bus = bus
        super().__init__(
            object_path=IBus.PATH_FACTORY,
            connection=bus.get_connection(),
        )
        log.info("VoiceIOEngineFactory created")

    def do_create_engine(self, engine_name):
        global _engine
        VoiceIOEngineFactory._engine_count += 1
        obj_path = f"/org/freedesktop/IBus/Engine/{VoiceIOEngineFactory._engine_count}"
        log.info("Creating engine '%s' at %s", engine_name, obj_path)
        # A fresh engine instance means a new focus/window context. Any commands
        # buffered against the previous (non-existent) instance are stale — drop
        # them so they don't replay into whatever window is focused now.
        if len(_pending):
            log.info("Dropping %d buffered command(s) on engine creation", len(_pending))
            _pending.clear()
        engine = VoiceIOEngine(
            engine_name=engine_name,
            object_path=obj_path,
            connection=self._bus.get_connection(),
        )
        _engine = engine
        # Signal readiness to the voiceio daemon
        try:
            READY_PATH.write_text(str(os.getpid()))
            log.info("Engine ready signal written to %s", READY_PATH)
        except OSError:
            pass
        return engine


# Global engine reference (set when factory creates the engine)
_engine: VoiceIOEngine | None = None
# Commands that arrived before the engine instance existed. Stale entries are
# dropped and the whole buffer is cleared on engine creation (see PendingBuffer).
_pending = PendingBuffer()


# Inode of the socket file this process bound. A replaced engine exiting late
# under load used to unlink its *successor's* socket, leaving the new engine
# alive but unreachable (then reaped as a zombie, in a loop).
_bound_ino: int | None = None


def _unlink_socket_if_ours() -> None:
    try:
        if _bound_ino is not None and SOCKET_PATH.stat().st_ino == _bound_ino:
            SOCKET_PATH.unlink()
    except OSError:
        pass


# Set when the engine shuts down. The listener must not poll
# mainloop.is_running(): it starts before mainloop.run(), and on a loaded CPU
# it won that race, saw "not running", exited and unlinked its own socket —
# an engine alive but deaf, which the daemon then reaped as a zombie.
_stopping = threading.Event()


def _socket_listener(mainloop: GLib.MainLoop, stopping: threading.Event = _stopping) -> None:
    """Listen for commands on Unix DGRAM socket. Runs in a thread."""
    global _bound_ino
    try:
        SOCKET_PATH.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(str(SOCKET_PATH))
        _bound_ino = SOCKET_PATH.stat().st_ino
    except OSError:
        # Without a socket the engine is deaf: exit so the watchdog respawns it.
        log.exception("Socket listener could not bind — stopping the engine")
        GLib.idle_add(mainloop.quit)
        return
    sock.settimeout(1.0)
    log.info("Socket listener started at %s", SOCKET_PATH)

    while not stopping.is_set():
        try:
            data, addr = sock.recvfrom(RECV_BYTES)
        except socket.timeout:
            continue
        except OSError:
            # A deaf engine is worse than a dead one: quit, so the daemon's
            # watchdog sees the exit and respawns a working engine.
            log.exception("Socket listener error — stopping the engine")
            GLib.idle_add(mainloop.quit)
            break

        msg = data.decode("utf-8", errors="replace")
        kind, _, body = msg.partition(":")
        log.debug("Received: %s%s", kind, f" ({len(body)} chars)" if body else "")

        if msg == "ping":
            # Respond to probe: send pong back
            if addr:
                try:
                    sock.sendto(b"pong", addr)
                except OSError:
                    pass
            continue

        if msg == "focus?":
            # Quick focus query — no need to go through GLib
            if addr and _engine is not None:
                reply = b"focused" if _engine._focused else b"unfocused"
                try:
                    sock.sendto(reply, addr)
                except OSError:
                    pass
            continue

        # Dispatch to engine on GLib main thread
        GLib.idle_add(_handle_command, msg)

    sock.close()
    _unlink_socket_if_ours()


def _flush_pending() -> None:
    """Replay commands that arrived before the engine was ready, dropping any
    that have gone stale."""
    for msg in _pending.drain_fresh():
        _dispatch(msg)


def _dispatch(msg: str) -> None:
    """Execute a single command on the engine."""
    try:
        if msg.startswith("preedit:"):
            _engine.preedit(msg[8:])
        elif msg.startswith("commit:"):
            _engine.commit(msg[7:])
        elif msg == "clear":
            _engine.clear()
        else:
            log.warning("Unknown command: %s", msg.partition(":")[0][:40])
    except Exception:
        log.exception("Error dispatching command: %s", msg.partition(":")[0])


def _handle_command(msg: str) -> bool:
    """Handle a command on the GLib main thread. Returns False to remove from idle."""
    if _engine is None:
        log.debug("Engine not ready, buffering command: %s", msg.partition(":")[0])
        _pending.add(msg)
        return False

    # Flush any buffered commands first (stale ones are dropped)
    if len(_pending):
        log.info("Engine ready, flushing %d buffered commands", len(_pending))
        _flush_pending()

    _dispatch(msg)
    return False  # run once, don't repeat


def main() -> None:
    # Log to file so we can debug when IBus spawns us
    log_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus-engine.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                str(log_path), maxBytes=1_000_000, backupCount=1,
            ),
        ],
    )

    # Refuse to double-run, race-free. ibus-daemon exec-spawns its own copy of
    # this engine (component XML) when something activates the voiceio source
    # (GNOME per-window source memory / Super+Space — voiceio is first in
    # mru-sources). Without an atomic gate, two engines both pass a ping check
    # then both unlink+rebind SOCKET_PATH in _socket_listener; the loser is
    # left on an orphaned inode — a "zombie" the watchdog kills and respawns.
    # The lock fd is held for our whole lifetime (released on process death).
    global _lock_fd
    _lock_fd = acquire_singleton_lock()
    if _lock_fd is None:
        log.info("Another VoiceIO engine holds the lock — exiting duplicate")
        return

    IBus.init()
    bus = IBus.Bus()

    if not bus.is_connected():
        log.error("Cannot connect to IBus daemon. Is IBus running?")
        sys.exit(1)

    # Register GTypes
    GObject.type_register(VoiceIOEngine)
    GObject.type_register(VoiceIOEngineFactory)

    # Create custom factory (registers on D-Bus at IBus.PATH_FACTORY)
    VoiceIOEngineFactory(bus)  # registers on D-Bus at IBus.PATH_FACTORY

    # Register component so IBus knows about our engine
    component = IBus.Component.new(
        COMPONENT_NAME,
        "VoiceIO voice input",
        "1.0",
        "MIT",
        "voiceio",
        "",
        "",
        "voiceio",
    )
    engine_desc = IBus.EngineDesc.new(
        ENGINE_NAME,
        "VoiceIO",
        "Voice-to-text input",
        "other",
        "MIT",
        "voiceio",
        "",
        "us",
    )
    component.add_engine(engine_desc)
    bus.register_component(component)

    log.info("VoiceIO IBus engine registered with custom factory")
    bus.request_name(COMPONENT_NAME, 0)

    mainloop = GLib.MainLoop()

    # Start socket listener in background thread
    listener = threading.Thread(
        target=_socket_listener, args=(mainloop,), daemon=True,
    )
    listener.start()

    try:
        mainloop.run()
    except KeyboardInterrupt:
        pass
    finally:
        _stopping.set()
        _unlink_socket_if_ours()
        READY_PATH.unlink(missing_ok=True)
        log.info("VoiceIO IBus engine stopped")


if __name__ == "__main__":
    main()
