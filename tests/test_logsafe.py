"""Logs record lengths and outcomes, never the words.

A transcript is the user's speech. It belongs in the files they own and can
delete (history.jsonl, retained audio) — not in a log, which on a desktop
lands in ~/.local/state/voiceio/ and under systemd lands in the journal, on
disk, for every utterance ever spoken.
"""
import logging

from voiceio.logsafe import summary


def test_summary_reports_length_not_content():
    assert summary("Hello world.") == "12 chars"
    assert "Hello" not in summary("Hello world.")


def test_summary_of_nothing_said():
    assert summary("") == "(silence)"


def test_transcriber_logs_the_size_not_the_sentence(caplog):
    """The line this replaced put every dictated sentence in the journal."""
    secret = "my bank password is hunter two"
    with caplog.at_level(logging.DEBUG, logger="voiceio.transcriber"):
        logging.getLogger("voiceio.transcriber").info(
            "Transcribed %.1fs audio in %.1fs (%.1fx realtime): %s",
            10.0, 1.0, 10.0, summary(secret),
        )
    assert secret not in caplog.text
    assert "30 chars" in caplog.text
