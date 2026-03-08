from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from pynput import keyboard

from voiceio import config
from voiceio.recorder import AudioRecorder
from voiceio.transcriber import Transcriber
from voiceio.typer import type_text

log = logging.getLogger("voiceio")

# Map common key names to pynput Key attributes
_SPECIAL_KEYS = {
    "Super_r": keyboard.Key.cmd_r,
    "Super_l": keyboard.Key.cmd_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "shift_r": keyboard.Key.shift_r,
    "shift_l": keyboard.Key.shift_l,
    "caps_lock": keyboard.Key.caps_lock,
    "scroll_lock": keyboard.Key.scroll_lock,
    "pause": keyboard.Key.pause,
    "insert": keyboard.Key.insert,
    "print_screen": keyboard.Key.print_screen,
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
}


def _resolve_hotkey(key_name: str):
    if key_name in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[key_name]
    # Single character key
    if len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)
    # Try as a pynput Key attribute
    try:
        return getattr(keyboard.Key, key_name)
    except AttributeError:
        log.error("Unknown key '%s', falling back to Super_r", key_name)
        return keyboard.Key.cmd_r


class VoiceIO:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.hotkey = _resolve_hotkey(cfg.hotkey.key)
        self.recorder = AudioRecorder(cfg.audio)
        self.transcriber = Transcriber(cfg.model)
        self._processing = False

    def on_press(self, key) -> None:
        if key == self.hotkey and not self.recorder.is_recording and not self._processing:
            self.recorder.start()
            try:
                from voiceio.tray import set_recording
                set_recording(True)
            except Exception:
                pass

    def on_release(self, key) -> None:
        if key == self.hotkey and self.recorder.is_recording:
            audio = self.recorder.stop()
            try:
                from voiceio.tray import set_recording
                set_recording(False)
            except Exception:
                pass
            if audio is not None:
                # Process in a thread so we don't block the listener
                threading.Thread(
                    target=self._process, args=(audio,), daemon=True
                ).start()

    def _process(self, audio) -> None:
        self._processing = True
        try:
            text = self.transcriber.transcribe(audio)
            if text:
                type_text(text, method=self.cfg.output.method)
        except Exception:
            log.exception("Transcription/typing failed")
        finally:
            self._processing = False

    def run(self) -> None:
        if self.cfg.tray.enabled:
            try:
                from voiceio.tray import start as start_tray
                start_tray(quit_callback=lambda: signal.raise_signal(signal.SIGINT))
            except Exception:
                log.warning("Failed to start tray icon", exc_info=True)

        log.info("voiceio ready — hold [%s] to record", self.cfg.hotkey.key)

        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

        log.info("voiceio stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voiceio",
        description="Push-to-talk voice-to-text for Linux",
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to config.toml"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override model name (e.g. large-v3)"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Override language (e.g. es, auto)"
    )
    args = parser.parse_args()

    cfg = config.load(path=args.config and __import__("pathlib").Path(args.config))

    if args.verbose:
        cfg.daemon.log_level = "DEBUG"
    if args.model:
        cfg.model.name = args.model
    if args.language:
        cfg.model.language = args.language

    logging.basicConfig(
        level=getattr(logging, cfg.daemon.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Graceful shutdown
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    vio = VoiceIO(cfg)
    vio.run()


if __name__ == "__main__":
    main()
