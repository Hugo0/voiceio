"""Tests for the non-interactive `voiceio correct --auto --batch` mode."""
from __future__ import annotations

import time
from unittest.mock import patch

from voiceio.autocorrect import AdjudicationResult, ReviewResult, SuspiciousWord
from voiceio.autocorrect_state import load_state
from voiceio.cli import _cmd_correct_auto
from voiceio.corrections import CorrectionDict
from voiceio.service import (
    _correct_service_unit,
    _correct_timer_unit,
    _service_unit,
)


def _entries():
    return [{"ts": time.time(), "text": "check the mantekka transfers"}]


def _suspicious():
    return [SuspiciousWord(word="mantekka", count=3, reason="rare word",
                           contexts=["check the mantekka transfers"])]


def _cfg_with_mining():
    """Config with the retired correction mining explicitly opted back in."""
    from voiceio.config import load
    cfg = load()
    cfg.autocorrect.mine_corrections = True
    return cfg


class TestBatchMode:
    def test_no_api_key_bails_quietly(self, tmp_path, capsys):
        cd = CorrectionDict(path=tmp_path / "c.json")
        with patch("voiceio.llm_api.resolve_api_key", return_value=""), \
             patch("voiceio.history.read", return_value=_entries()):
            _cmd_correct_auto(cd, batch=True)
        assert "skipping batch" in capsys.readouterr().out.lower()

    def test_ambiguous_item_deferred_no_notify_cursor_advances(self, tmp_path):
        """No consensus → silently deferred; no queue, no notification, cursor advances.

        Opts mining back in: it's retired by default (see
        TestCorrectionMiningRetired) but the machinery must still be correct.
        """
        cd = CorrectionDict(path=tmp_path / "c.json")
        review = ReviewResult()
        review.ask_user.append({"wrong": "mantekka", "right": "Manteca",
                                "reason": "ambiguous"})
        adj = AdjudicationResult(
            deferred=[{"wrong": "mantekka", "right": "Manteca", "votes": []}],
        )
        with patch("voiceio.config.load", return_value=_cfg_with_mining()), \
             patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.autocorrect.adjudicate", return_value=adj), \
             patch("voiceio.feedback.notify") as notify:
            _cmd_correct_auto(cd, batch=True)
        # No human queue: nothing added, nothing notified.
        assert not notify.called
        assert "mantekka" not in cd.list_all()
        # Cursor ALWAYS advances; the item is deferred for later.
        state = load_state()
        assert state.last_scan_ts
        assert "mantekka" in state.deferred

    def test_unanimous_correction_applied_and_notified(self, tmp_path):
        """Mining opted back in — it's retired by default, not deleted."""
        cd = CorrectionDict(path=tmp_path / "c.json")
        review = ReviewResult()
        review.ask_user.append({"wrong": "mantekka", "right": "", "reason": "?"})
        adj = AdjudicationResult(apply=[{"wrong": "mantekka", "right": "manteca"}])
        with patch("voiceio.config.load", return_value=_cfg_with_mining()), \
             patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.autocorrect.adjudicate", return_value=adj), \
             patch("voiceio.feedback.notify") as notify:
            _cmd_correct_auto(cd, batch=True)
        assert cd.list_all().get("mantekka") == "manteca"
        assert notify.called
        assert "correction" in notify.call_args[0][1].lower()
        assert load_state().last_scan_ts

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
        adj = AdjudicationResult(
            deferred=[{"wrong": "mantekka", "right": "", "votes": []}],
        )
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.autocorrect.adjudicate", return_value=adj), \
             patch("voiceio.feedback.notify"):
            _cmd_correct_auto(cd, batch=True)  # must not block on input()
        assert "Grafana" in (tmp_path / "vocabulary.txt").read_text()


class TestServiceUnit:
    def test_restarts_on_graphical_relogin(self):
        """PartOf stops the service on logout; WantedBy must pull it back in on
        the next login. Both must name graphical-session.target or the service
        dies on logout and never returns until a full reboot (GNOME re-login
        leaves the systemd --user manager — and default.target — untouched)."""
        unit = _service_unit("/usr/bin/voiceio")
        assert "PartOf=graphical-session.target" in unit
        assert "WantedBy=graphical-session.target" in unit
        # default.target is reached once at user-manager start and never
        # re-entered on re-login, so it must NOT be the wake-up trigger.
        assert "WantedBy=default.target" not in unit


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


class TestCorrectionMiningRetired:
    """Correction mining is off by default.

    Measured over months: 387 rules mined, 4 ever fired (1%), while the runtime
    postcorrect pass applied 288 edits. Rules are exact-string matches, so they
    only fire if Whisper repeats a misrecognition verbatim — and the errors
    worth fixing ("nuance" for "neurons") are common words the safety gate must
    refuse. One run learned rules that destroyed real Spanish words. Vocabulary
    mining is a different story and stays on.
    """

    def _review(self):
        # "compain" -> "company" passes gate_correction: the target is a common
        # word. ("olamma" -> "Ollama" would be rejected because the target is
        # neither common nor already in the vocabulary — pre-existing gate
        # behaviour, unrelated to mining being retired.)
        from voiceio.autocorrect import ReviewResult
        return ReviewResult(
            auto_fix=[{"wrong": "compain", "right": "company", "reason": "typo"}],
            ask_user=[{"wrong": "wordal", "right": "Wordle", "reason": "maybe"}],
            vocabulary=["Kubernetes"],
        )

    def test_default_is_off(self):
        from voiceio.config import AutocorrectConfig
        assert AutocorrectConfig().mine_corrections is False

    def test_no_rules_learned_by_default(self):
        """The whole point: a batch run must not grow corrections.json."""
        from voiceio.corrections import CorrectionDict

        cd = CorrectionDict()
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words", return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=self._review()), \
             patch("voiceio.autocorrect.adjudicate") as adj, \
             patch("voiceio.vocabulary.add_terms", return_value=1) as add_terms, \
             patch("voiceio.feedback.notify"):
            _cmd_correct_auto(cd, batch=True)

        assert cd.list_all() == {}       # no rules learned
        assert not adj.called            # and no LLM spend adjudicating them
        assert add_terms.called          # vocabulary mining still runs

    def test_opt_in_still_mines(self):
        """The machinery is retired, not deleted — it must still work if asked."""
        from voiceio.autocorrect import AdjudicationResult
        from voiceio.corrections import CorrectionDict

        cfg = _cfg_with_mining()
        cd = CorrectionDict()
        adj = AdjudicationResult(apply=[], vocabulary=[], deferred=[])
        with patch("voiceio.config.load", return_value=cfg), \
             patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words", return_value=_suspicious()), \
             patch("voiceio.autocorrect.review_suspicious", return_value=self._review()), \
             patch("voiceio.autocorrect.adjudicate", return_value=adj) as adjudicate, \
             patch("voiceio.vocabulary.add_terms", return_value=1), \
             patch("voiceio.feedback.notify"):
            _cmd_correct_auto(cd, batch=True)

        assert "compain" in cd.list_all()  # gate-passing auto_fix applied
        assert adjudicate.called           # ambiguous ones still adjudicated
