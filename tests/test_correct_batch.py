"""Tests for the non-interactive `voiceio correct --auto --batch` mode."""
from __future__ import annotations

import time
from unittest.mock import patch

from voiceio.autocorrect import ReviewResult, SuspiciousWord
from voiceio.autocorrect_state import load_state
from voiceio.cli import _cmd_correct_auto
from voiceio.corrections import CorrectionDict
from voiceio.service import _correct_service_unit, _correct_timer_unit


def _entries():
    return [{"ts": time.time(), "text": "check the mantekka transfers"}]


def _suspicious():
    return [SuspiciousWord(word="mantekka", count=3, reason="rare word",
                           contexts=["check the mantekka transfers"])]


class TestBatchMode:
    def test_no_api_key_bails_quietly(self, tmp_path, capsys):
        cd = CorrectionDict(path=tmp_path / "c.json")
        with patch("voiceio.llm_api.resolve_api_key", return_value=""), \
             patch("voiceio.history.read", return_value=_entries()):
            _cmd_correct_auto(cd, batch=True)
        assert "skipping batch" in capsys.readouterr().out.lower()

    def test_pending_review_notifies_and_keeps_cursor(self, tmp_path):
        cd = CorrectionDict(path=tmp_path / "c.json")
        review = ReviewResult()
        review.ask_user.append({"wrong": "mantekka", "right": "Manteca",
                                "reason": "ambiguous"})
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.feedback.notify") as notify:
            _cmd_correct_auto(cd, batch=True)
        assert notify.called
        assert "1 suggestion" in notify.call_args[0][1]
        # Cursor must NOT advance — pending items stay re-proposable
        assert not load_state().last_scan_ts

    def test_clean_run_advances_cursor(self, tmp_path):
        cd = CorrectionDict(path=tmp_path / "c.json")
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words", return_value=[]), \
             patch("voiceio.feedback.notify") as notify:
            _cmd_correct_auto(cd, batch=True)
        assert not notify.called
        assert load_state().last_scan_ts is not None

    def test_vocabulary_added_without_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voiceio.config.CONFIG_DIR", tmp_path)
        cd = CorrectionDict(path=tmp_path / "c.json")
        review = ReviewResult()
        review.vocabulary.append("Grafana")
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.feedback.notify"):
            _cmd_correct_auto(cd, batch=True)  # must not block on input()
        assert "Grafana" in (tmp_path / "vocabulary.txt").read_text()


class TestTimerUnits:
    def test_service_unit_runs_batch(self):
        unit = _correct_service_unit("/usr/bin/voiceio")
        assert "correct --auto --batch" in unit
        assert "Type=oneshot" in unit

    def test_timer_is_weekly_persistent(self):
        unit = _correct_timer_unit()
        assert "OnCalendar=weekly" in unit
        assert "Persistent=true" in unit


class TestProtectLanguages:
    def test_spanish_word_blocked(self):
        from voiceio.autocorrect import gate_correction
        # "harina" (Spanish: flour) is a non-word in English but must not be
        # rewritten for a bilingual user
        assert gate_correction("harina", "hearing",
                               protect_languages=["es"]) is not None
        # without protection it passes (English-only user)
        assert gate_correction("harina", "hearing") is None

    def test_english_nonword_still_passes(self):
        from voiceio.autocorrect import gate_correction
        assert gate_correction("ngroc", "ngrok",
                               vocabulary={"ngrok"},
                               protect_languages=["es", "ca"]) is None
