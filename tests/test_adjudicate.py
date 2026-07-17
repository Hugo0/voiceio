"""Tests for evidence-based multi-vote adjudication (replaces the review queue)."""
from __future__ import annotations

import json
import logging
import time
from unittest.mock import patch

from voiceio.autocorrect import (
    AdjudicationResult, SuspiciousWord, adjudicate,
)
from voiceio.autocorrect_state import (
    DEFER_COOLDOWN_SECS, MAX_DEFER_FAILURES, AutocorrectState,
)
from voiceio.cli import _cmd_correct_auto
from voiceio.corrections import CorrectionDict


class _Cfg:
    """Minimal stand-in for the autocorrect config the adjudicator reads."""
    class autocorrect:
        protect_languages: tuple = ()
        model = "test-model"
        base_url = "https://example/v1"
        api_key = ""
        timeout_secs = 5


def _sw(word, count=5, contexts=None):
    return SuspiciousWord(
        word=word, count=count,
        contexts=contexts or [f"a sentence with {word} in it"],
    )


def _vote(word, verdict, right=""):
    return {"word": word, "verdict": verdict, "right": right}


def _chat_returning(*verdict_lists):
    """Build a fake `chat` that returns each verdict list in successive calls."""
    responses = [json.dumps({"verdicts": vl}) for vl in verdict_lists]
    calls = {"n": 0}

    def _fake_chat(cfg, system, user, *, api_key="", **kw):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    return _fake_chat, calls


# ── adjudicate() unit behaviour ──────────────────────────────────────────

class TestAdjudicate:
    def test_unanimous_correction_applied(self):
        items = [{"wrong": "ngroc", "right": "", "reason": ""}]
        sw = {"ngroc": _sw("ngroc")}
        fake, calls = _chat_returning(
            [_vote("ngroc", "correction", "ngrok")],
            [_vote("ngroc", "correction", "ngrok")],
            [_vote("ngroc", "correction", "ngrok")],
        )
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(_Cfg, items, sw, votes=3, vocabulary={"ngrok"})
        assert res.apply == [{"wrong": "ngroc", "right": "ngrok"}]
        assert not res.deferred
        assert calls["n"] == 3  # one API call per vote, not per item×vote

    def test_correction_that_fails_gate_is_deferred(self):
        # "harina" is a real Spanish word; with es protected it must not apply
        # even under unanimous agreement.
        class Cfg(_Cfg):
            class autocorrect(_Cfg.autocorrect):
                protect_languages = ("es",)

        items = [{"wrong": "harina", "right": ""}]
        sw = {"harina": _sw("harina")}
        fake, _ = _chat_returning(*[[_vote("harina", "correction", "hearing")]] * 3)
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(Cfg, items, sw, votes=3)
        assert not res.apply
        assert [d["wrong"] for d in res.deferred] == ["harina"]

    def test_split_vote_deferred(self):
        items = [{"wrong": "tridle", "right": ""}]
        sw = {"tridle": _sw("tridle")}
        fake, _ = _chat_returning(
            [_vote("tridle", "correction", "trident")],
            [_vote("tridle", "keep")],
            [_vote("tridle", "uncertain")],
        )
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(_Cfg, items, sw, votes=3)
        assert not res.apply
        assert not res.vocabulary
        assert res.deferred[0]["wrong"] == "tridle"
        assert len(res.deferred[0]["votes"]) == 3

    def test_disagreeing_targets_deferred(self):
        items = [{"wrong": "olamma", "right": ""}]
        sw = {"olamma": _sw("olamma")}
        fake, _ = _chat_returning(
            [_vote("olamma", "correction", "Ollama")],
            [_vote("olamma", "correction", "llama")],
            [_vote("olamma", "correction", "Ollama")],
        )
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(_Cfg, items, sw, votes=3)
        assert not res.apply
        assert res.deferred[0]["wrong"] == "olamma"

    def test_unanimous_keep_becomes_vocabulary(self):
        items = [{"wrong": "grafana", "right": ""}]
        sw = {"grafana": _sw("grafana")}
        fake, _ = _chat_returning(*[[_vote("grafana", "keep")]] * 3)
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(_Cfg, items, sw, votes=3)
        assert res.vocabulary == ["grafana"]
        assert not res.deferred

    def test_missing_vote_defers(self):
        # If a pass omits the word, it isn't unanimous → defer.
        items = [{"wrong": "wibble", "right": ""}]
        sw = {"wibble": _sw("wibble")}
        fake, _ = _chat_returning(
            [_vote("wibble", "correction", "wobble")],
            [],  # omitted
            [_vote("wibble", "correction", "wobble")],
        )
        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", fake):
            res = adjudicate(_Cfg, items, sw, votes=3, vocabulary={"wobble"})
        assert not res.apply
        assert res.deferred[0]["wrong"] == "wibble"

    def test_no_api_key_defers_all_untouched(self):
        items = [{"wrong": "foo"}, {"wrong": "bar"}]
        with patch("voiceio.llm_api.resolve_api_key", return_value=""):
            res = adjudicate(_Cfg, items, {}, votes=3)
        assert {d["wrong"] for d in res.deferred} == {"foo", "bar"}
        assert not res.apply

    def test_full_contexts_sent_to_llm(self):
        long_ctx = "the quick brown fox jumps over the lazy dog " * 4 + "wibble"
        items = [{"wrong": "wibble"}]
        sw = {"wibble": _sw("wibble", contexts=[long_ctx])}
        seen = {}

        def _fake_chat(cfg, system, user, *, api_key="", **kw):
            seen["user"] = user
            return json.dumps({"verdicts": [_vote("wibble", "uncertain")]})

        with patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.llm_api.chat", _fake_chat):
            adjudicate(_Cfg, items, sw, votes=1)
        # The full sentence must be present, not a 100-char snippet.
        assert long_ctx in seen["user"]


# ── Deferred-state lifecycle ─────────────────────────────────────────────

class TestDeferredState:
    def test_cooldown_respected(self):
        st = AutocorrectState()
        now = 1000.0
        st.defer("foo", ts=now)
        assert st.in_cooldown("foo", now=now)
        assert st.in_cooldown("foo", now=now + DEFER_COOLDOWN_SECS - 1)
        assert not st.in_cooldown("foo", now=now + DEFER_COOLDOWN_SECS + 1)

    def test_third_failure_dismisses(self):
        st = AutocorrectState()
        for _ in range(MAX_DEFER_FAILURES - 1):
            st.defer("foo")
        assert not st.is_dismissed("foo")
        assert "foo" in st.deferred
        st.defer("foo")  # the MAX_DEFER_FAILURES-th failure
        assert st.is_dismissed("foo")
        assert "foo" not in st.deferred

    def test_capacity_defer_no_penalty_no_cooldown(self):
        st = AutocorrectState()
        now = 5000.0
        st.defer("foo", failure=False)
        assert st.deferred["foo"]["count"] == 0
        assert not st.in_cooldown("foo", now=now)  # eligible immediately

    def test_ready_vs_cooldown_partition(self):
        st = AutocorrectState()
        now = 9000.0
        st.defer("hot", ts=now)             # in cooldown
        st.defer("cold", failure=False)     # ready immediately
        assert st.cooldown_words(now) == {"hot"}
        assert st.ready_deferred(now) == {"cold"}


# ── Batch flow integration ───────────────────────────────────────────────

def _entries():
    return [{"ts": time.time(), "text": "check the mantekka transfers"}]


def _suspicious():
    return [SuspiciousWord(word="mantekka", count=3,
                           contexts=["check the mantekka transfers"])]


class TestBatchFlow:
    """The adjudication flow. Correction mining is retired by default
    ([autocorrect] mine_corrections = false, see test_correct_batch.py), so
    these opt it back in — the machinery still has to be correct for anyone who
    turns it on, and for the vocabulary path that shares it."""

    def _cfg(self):
        from voiceio.config import load
        cfg = load()
        cfg.autocorrect.mine_corrections = True
        return cfg

    def _run(self, tmp_path, adj, *, review=None, suspicious=None, notify=None):
        from voiceio.autocorrect import ReviewResult
        cd = CorrectionDict(path=tmp_path / "c.json")
        review = review if review is not None else ReviewResult(
            ask_user=[{"wrong": "mantekka", "right": "", "reason": "?"}],
        )
        suspicious = suspicious if suspicious is not None else _suspicious()
        with patch("voiceio.config.load", return_value=self._cfg()), \
             patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words",
                   return_value=suspicious), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.autocorrect.adjudicate", return_value=adj), \
             patch("builtins.input", side_effect=AssertionError("input() called")), \
             patch("voiceio.feedback.notify", notify or (lambda *a, **k: None)):
            _cmd_correct_auto(cd, batch=True)
        return cd

    def test_batch_never_prompts_and_advances_cursor(self, tmp_path):
        from voiceio.autocorrect_state import load_state
        adj = AdjudicationResult(
            deferred=[{"wrong": "mantekka", "right": "", "votes": []}],
        )
        self._run(tmp_path, adj)  # input() side_effect would raise if called
        assert load_state().last_scan_ts
        assert "mantekka" in load_state().deferred

    def test_notify_only_when_changes(self, tmp_path):
        from unittest.mock import MagicMock

        # No changes → no notification.
        n1 = MagicMock()
        self._run(tmp_path, AdjudicationResult(
            deferred=[{"wrong": "mantekka", "votes": []}]), notify=n1)
        assert not n1.called

        # A learned correction → notification fires.
        n2 = MagicMock()
        self._run(tmp_path, AdjudicationResult(
            apply=[{"wrong": "mantekka", "right": "manteca"}]), notify=n2)
        assert n2.called

    def test_cap_defers_tail_with_log(self, tmp_path, caplog):
        from voiceio.autocorrect_state import load_state

        # 200 unclassified suspicious words → over the 150 cap.
        many = [SuspiciousWord(word=f"junk{i:03d}", count=200 - i,
                               contexts=[f"junk{i:03d} here"]) for i in range(200)]
        from voiceio.autocorrect import ReviewResult
        review = ReviewResult()  # everything falls through to "unclassified"

        # adjudicate defers whatever it receives; capture how many were passed.
        received = {}

        def _fake_adj(cfg, items, sw_by_word, **kw):
            received["n"] = len(items)
            return AdjudicationResult(
                deferred=[{"wrong": it["wrong"], "votes": []} for it in items],
            )

        cd = CorrectionDict(path=tmp_path / "c.json")
        with caplog.at_level(logging.INFO, logger="voiceio.cli"), \
             patch("voiceio.config.load", return_value=self._cfg()), \
             patch("voiceio.llm_api.resolve_api_key", return_value="sk-x"), \
             patch("voiceio.history.read", return_value=_entries()), \
             patch("voiceio.autocorrect.find_suspicious_words", return_value=many), \
             patch("voiceio.autocorrect.review_suspicious", return_value=review), \
             patch("voiceio.autocorrect.adjudicate", _fake_adj), \
             patch("voiceio.feedback.notify"):
            _cmd_correct_auto(cd, batch=True)

        assert received["n"] == 150  # capped
        assert any("cap" in r.message.lower() for r in caplog.records)
        # The lower-frequency tail (50 items) was capacity-deferred.
        assert len(load_state().deferred) == 200
