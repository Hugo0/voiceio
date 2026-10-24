"""`voiceio say`: let an agent or a script speak to you, and say who it is.

A chime, a desktop notification titled with the sender, then the sender's
name and the text read aloud. The daemon queues announcements and holds them
while you dictate: speech played into an open mic would be typed back out,
and talking over you is rude anyway. Without a daemon, `voiceio say` speaks
in its own process instead (slower: it loads the voice first).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

MAX_TEXT = 600          # characters read aloud; an announcement, not an essay
_QUEUE = 8              # waiting announcements beyond this are dropped
_HOLD_SECS = 600        # wait at most this long for a dictation to finish


@dataclass(frozen=True)
class Message:
    text: str
    sender: str = ""
    urgent: bool = False

    @property
    def spoken(self) -> str:
        return f"{self.sender}: {self.text}" if self.sender else self.text


def encode(msg: Message) -> str:
    """The control-socket command for `msg`."""
    return "say:" + json.dumps({"text": msg.text, "from": msg.sender, "urgent": msg.urgent})


def parse(payload: str) -> Message | None:
    try:
        d = json.loads(payload)
        text = " ".join(str(d.get("text", "")).split())[:MAX_TEXT]
        sender = " ".join(str(d.get("from", "")).split())[:60]
    except (ValueError, AttributeError):
        return None
    return Message(text, sender, bool(d.get("urgent"))) if text else None


def cue_and_notify(msg: Message) -> None:
    """The chime and the notification: who is talking, even with no voice."""
    from voiceio import feedback
    if feedback.open_cue_stream():
        (feedback.play_alert_sound if msg.urgent else feedback.play_commit_sound)()
        time.sleep(0.4)   # let the chime land before the voice starts
    feedback.notify(msg.sender or "voiceio", msg.text, urgent=msg.urgent)


class Announcer:
    """One announcement at a time, in order, on its own thread."""

    def __init__(self, speak: Callable[[str], None], busy: Callable[[], bool],
                 hold_secs: float = _HOLD_SECS) -> None:
        self._speak = speak
        self._busy = busy
        self._hold_secs = hold_secs
        self._queue: queue.Queue[Message] = queue.Queue(maxsize=_QUEUE)
        threading.Thread(target=self._run, name="announcer", daemon=True).start()

    def submit(self, msg: Message) -> bool:
        try:
            self._queue.put_nowait(msg)
            return True
        except queue.Full:
            log.warning("say: %d announcements waiting, dropped one from %s",
                        _QUEUE, msg.sender or "?")
            return False

    def _run(self) -> None:
        while True:
            msg = self._queue.get()
            deadline = time.monotonic() + self._hold_secs
            while self._busy() and time.monotonic() < deadline:
                time.sleep(0.25)
            try:
                cue_and_notify(msg)
                self._speak(msg.spoken)
            except Exception:
                log.exception("say: announcement from %s failed", msg.sender or "?")


def say_here(msg: Message) -> bool:
    """No daemon: chime, notify and speak in this process. False if no TTS
    engine works here (the chime and notification still happened)."""
    from voiceio import config, tts
    from voiceio.tts.player import TTSPlayer
    cue_and_notify(msg)
    cfg = config.load().tts
    engine = tts.select(cfg, allow_network=False)
    if engine is None:
        return False
    audio, rate = engine.synthesize(msg.spoken, cfg.voice, cfg.speed)
    TTSPlayer().play(audio, rate)
    return True
