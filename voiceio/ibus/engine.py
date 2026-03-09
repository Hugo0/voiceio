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

from voiceio.ibus import READY_PATH, SOCKET_PATH

log = logging.getLogger(__name__)
ENGINE_NAME = "voiceio"
COMPONENT_NAME = "org.voiceio.ibus"


class VoiceIOEngine(IBus.Engine):
    """IBus engine that receives text injection commands via socket."""

    __gtype_name__ = "VoiceIOEngine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._focused = False
        log.info("VoiceIOEngine instance created (path=%s)", kwargs.get("object_path"))

    def do_focus_in(self):
        self._focused = True

    def do_focus_out(self):
        self._focused = False

    def do_process_key_event(self, keyval, keycode, state):
        # CRITICAL: Always pass all keys through. Never intercept typing.
        # If this ever returns True (or raises), ALL keyboard input dies system-wide.
        try:
            return False
        except Exception:
            return False

    def preedit(self, text: str) -> None:
        """Show text as preedit (underlined preview)."""
        if not text:
            self.hide_preedit_text()
            return
        ibus_text = IBus.Text.new_from_string(text)
        ibus_text.append_attribute(
            IBus.AttrType.UNDERLINE,
            IBus.AttrUnderline.SINGLE,
            0,
            len(text),
        )
        self.update_preedit_text(ibus_text, len(text), True)

    def commit(self, text: str) -> None:
        """Clear preedit and commit final text."""
        self.hide_preedit_text()
        if text:
            self.commit_text(IBus.Text.new_from_string(text))

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
_pending_commands: list[str] = []


def _socket_listener(mainloop: GLib.MainLoop) -> None:
    """Listen for commands on Unix DGRAM socket. Runs in a thread."""
    SOCKET_PATH.unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(SOCKET_PATH))
    sock.settimeout(1.0)
    log.info("Socket listener started at %s", SOCKET_PATH)

    while mainloop.is_running():
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue
        except OSError:
            break

        msg = data.decode("utf-8", errors="replace")
        log.debug("Received: %s", msg[:80])

        if msg == "ping":
            # Respond to probe: send pong back
            if addr:
                try:
                    sock.sendto(b"pong", addr)
                except OSError:
                    pass
            continue

        # Dispatch to engine on GLib main thread
        GLib.idle_add(_handle_command, msg)

    sock.close()
    SOCKET_PATH.unlink(missing_ok=True)


def _flush_pending() -> None:
    """Replay any commands that arrived before the engine was ready."""
    while _pending_commands:
        _dispatch(_pending_commands.pop(0))


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
            log.warning("Unknown command: %s", msg[:40])
    except Exception:
        log.exception("Error dispatching command: %s", msg[:40])


def _handle_command(msg: str) -> bool:
    """Handle a command on the GLib main thread. Returns False to remove from idle."""
    if _engine is None:
        log.debug("Engine not ready, buffering command: %s", msg[:40])
        _pending_commands.append(msg)
        return False

    # Flush any buffered commands first
    if _pending_commands:
        log.info("Engine ready, flushing %d buffered commands", len(_pending_commands))
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
        SOCKET_PATH.unlink(missing_ok=True)
        READY_PATH.unlink(missing_ok=True)
        log.info("VoiceIO IBus engine stopped")


if __name__ == "__main__":
    main()
