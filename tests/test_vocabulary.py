"""Tests for vocabulary file loading, ranking and token-budgeted selection."""
from __future__ import annotations

import os
import time

from voiceio.config import ModelConfig
from voiceio.vocabulary import VocabularyLoader, add_terms, load_vocabulary


class TestLoadVocabulary:
    def test_basic_file(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("VoiceIO\nGNOME\nWayland\n")
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert result == "VoiceIO, GNOME, Wayland"

    def test_comments_and_blanks(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("# My vocabulary\nVoiceIO\n\n# Another comment\nGNOME\n\n")
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert result == "VoiceIO, GNOME"

    def test_missing_file_returns_empty(self):
        cfg = ModelConfig(vocabulary_file="/nonexistent/path/vocab.txt")
        result = load_vocabulary(cfg)
        assert result == ""

    def test_empty_file_returns_empty(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("")
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert result == ""

    def test_comments_only_returns_empty(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("# Just comments\n# Nothing else\n")
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert result == ""

    def test_load_is_untruncated(self, tmp_path):
        """load_vocabulary returns EVERY term — truncating here was the bug.

        Budgeting moved to select_terms (token-aware, ranked). The callers that
        need the whole list — postcorrect's LLM, and the mining gate's "is this
        already in your vocabulary?" check — were silently reading a truncated
        view before.
        """
        vocab_file = tmp_path / "vocabulary.txt"
        terms = [f"LongTechnicalTerm{i:04d}" for i in range(100)]
        vocab_file.write_text("\n".join(terms))
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert len(result) > 800
        assert result.count(",") == 99

    def test_no_config_uses_default_location(self, tmp_path, monkeypatch):
        """With empty vocabulary_file, checks CONFIG_DIR/vocabulary.txt."""
        monkeypatch.setattr("voiceio.config.CONFIG_DIR", tmp_path)
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("TestTerm\n")
        cfg = ModelConfig(vocabulary_file="")
        result = load_vocabulary(cfg)
        assert result == "TestTerm"

    def test_no_default_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voiceio.config.CONFIG_DIR", tmp_path)
        cfg = ModelConfig(vocabulary_file="")
        result = load_vocabulary(cfg)
        assert result == ""

    def test_whitespace_stripped(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        vocab_file.write_text("  VoiceIO  \n  GNOME  \n")
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert result == "VoiceIO, GNOME"


class TestAddTerms:
    def test_creates_file_when_missing(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        cfg = ModelConfig(vocabulary_file=str(vf))
        added = add_terms(["Kubernetes", "Grafana"], cfg)
        assert added == 2
        assert vf.exists()
        assert "Kubernetes" in vf.read_text()
        assert "Grafana" in vf.read_text()

    def test_case_insensitive_dedupe_against_existing(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        vf.write_text("Grafana\n")
        cfg = ModelConfig(vocabulary_file=str(vf))
        added = add_terms(["grafana", "Postgres"], cfg)
        assert added == 1  # grafana already present (case-insensitive)
        assert vf.read_text().count("Postgres") == 1

    def test_dedupe_within_batch(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        cfg = ModelConfig(vocabulary_file=str(vf))
        added = add_terms(["Redis", "redis", "REDIS"], cfg)
        assert added == 1

    def test_skips_junk(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        cfg = ModelConfig(vocabulary_file=str(vf))
        added = add_terms(["!!!", "x", "123", "a-b-c-good"], cfg)
        # only the alpha term survives; single char and numeric junk skipped
        assert added == 1
        assert "a-b-c-good" in vf.read_text()

    def test_skips_misspelling_of_existing(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        vf.write_text("Kubernetes\n")
        cfg = ModelConfig(vocabulary_file=str(vf))
        # "Kubernetis" is 1 edit from existing "Kubernetes" — belongs in
        # corrections, not vocabulary.
        added = add_terms(["Kubernetis"], cfg)
        assert added == 0

    def test_appends_newline_when_file_lacks_trailing(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        vf.write_text("Grafana")  # no trailing newline
        cfg = ModelConfig(vocabulary_file=str(vf))
        add_terms(["Postgres"], cfg)
        lines = [ln for ln in vf.read_text().splitlines() if ln.strip()]
        assert lines == ["Grafana", "Postgres"]


class TestVocabularyLoader:
    def test_caches_until_mtime_changes(self, tmp_path):
        vf = tmp_path / "vocabulary.txt"
        vf.write_text("Grafana\n")
        cfg = ModelConfig(vocabulary_file=str(vf))
        loader = VocabularyLoader(cfg)
        assert loader.get() == "Grafana"

        # Rewrite with a bumped mtime — loader should pick up the change.
        vf.write_text("Grafana\nPostgres\n")
        os.utime(vf, (vf.stat().st_atime + 10, vf.stat().st_mtime + 10))
        assert loader.get() == "Grafana, Postgres"

    def test_no_reread_when_unchanged(self, tmp_path, monkeypatch):
        vf = tmp_path / "vocabulary.txt"
        vf.write_text("Grafana\n")
        cfg = ModelConfig(vocabulary_file=str(vf))
        loader = VocabularyLoader(cfg)
        assert loader.get() == "Grafana"

        import voiceio.vocabulary as vocab_mod
        calls = {"n": 0}
        orig = vocab_mod.load_terms

        def _counting(model_cfg):
            calls["n"] += 1
            return orig(model_cfg)

        # load_terms is the real read now; patching load_vocabulary would pass
        # vacuously since the loader no longer calls it.
        monkeypatch.setattr(vocab_mod, "load_terms", _counting)
        loader.get()
        loader.get()
        assert calls["n"] == 0  # mtime unchanged → no reload

    def test_missing_file_returns_empty(self, tmp_path):
        cfg = ModelConfig(vocabulary_file=str(tmp_path / "nope.txt"))
        loader = VocabularyLoader(cfg)
        assert loader.get() == ""


class TestSelectTerms:
    """Selection is permanent: the hotword budget holds ~40-60 terms and the
    vocabulary only grows, so who gets a slot is the whole game."""

    def _stats(self, tmp_path, data):
        from voiceio.vocab_stats import VocabStats
        s = VocabStats(tmp_path / "vocab_stats.json")
        s._stats = {k.lower(): v for k, v in data.items()}
        return s

    def test_respects_token_budget(self):
        from voiceio.vocabulary import select_terms
        from voiceio.tokens import count_tokens

        terms = [f"LongTechnicalTerm{i:04d}" for i in range(100)]
        got = select_terms(terms, token_budget=50, model_name="small")
        assert got  # something fits
        assert len(got) < len(terms)  # but not everything
        assert count_tokens(", ".join(got), "small") <= 50

    def test_actually_fills_the_budget(self):
        """Guards the mistake this replaced: costing terms one at a time and
        summing double-counts the ", " joins, so the budget was only ~half
        spent (19 terms / 64 of 120 tokens on the real vocabulary)."""
        from voiceio.vocabulary import select_terms
        from voiceio.tokens import count_tokens

        terms = [f"Term{i:04d}" for i in range(200)]
        budget = 100
        got = select_terms(terms, token_budget=budget, model_name="small")
        used = count_tokens(", ".join(got), "small")
        assert used <= budget
        # Within one term's cost of the budget — no silent under-fill.
        assert used >= budget - 8
        # And adding the next-ranked term must genuinely overflow.
        nxt = [t for t in terms if t not in got][0]
        assert count_tokens(", ".join(got + [nxt]), "small") > budget

    def test_never_emits_a_partial_term(self):
        """The old code did vocab[:600] — a raw char slice that could cut a term
        in half and feed the decoder a fragment."""
        from voiceio.vocabulary import select_terms

        terms = ["Kubernetes", "Grafana", "Hetzner", "OpenRouter", "Metaculus"]
        got = select_terms(terms, token_budget=12, model_name="small")
        assert all(t in terms for t in got)

    def test_ranks_by_usage(self, tmp_path):
        from voiceio.vocabulary import select_terms

        terms = ["Alpha", "Beta", "Gamma"]
        stats = self._stats(tmp_path, {
            "Gamma": {"hits": 50, "last_seen_ts": time.time()},
        })
        # Budget for roughly one term: the used one must win despite file order.
        got = select_terms(terms, token_budget=6, model_name="small", stats=stats)
        assert got[0] == "Gamma"

    def test_recency_beats_stale_volume(self, tmp_path):
        """A term used heavily months ago should lose to one used today."""
        from voiceio.vocabulary import select_terms

        now = time.time()
        stats = self._stats(tmp_path, {
            "Stale": {"hits": 100, "last_seen_ts": now - 365 * 86400},
            "Fresh": {"hits": 5, "last_seen_ts": now},
        })
        got = select_terms(["Stale", "Fresh"], token_budget=6,
                           model_name="small", stats=stats, now=now)
        assert got[0] == "Fresh"

    def test_cold_start_is_file_order(self):
        """No stats → degrade to exactly the previous behaviour, never worse."""
        from voiceio.vocabulary import select_terms

        terms = ["First", "Second", "Third"]
        got = select_terms(terms, token_budget=1000, model_name="small", stats=None)
        assert got == terms

    def test_empty_and_zero_budget(self):
        from voiceio.vocabulary import select_terms

        assert select_terms([], token_budget=100, model_name="small") == []
        assert select_terms(["A"], token_budget=0, model_name="small") == []
