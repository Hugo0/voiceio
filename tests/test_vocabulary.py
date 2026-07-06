"""Tests for vocabulary file loading."""
from __future__ import annotations

import os

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

    def test_truncation(self, tmp_path):
        vocab_file = tmp_path / "vocabulary.txt"
        # Create a very long vocabulary list
        terms = [f"LongTechnicalTerm{i:04d}" for i in range(100)]
        vocab_file.write_text("\n".join(terms))
        cfg = ModelConfig(vocabulary_file=str(vocab_file))
        result = load_vocabulary(cfg)
        assert len(result) <= 800

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
        orig = vocab_mod.load_vocabulary

        def _counting(model_cfg):
            calls["n"] += 1
            return orig(model_cfg)

        monkeypatch.setattr(vocab_mod, "load_vocabulary", _counting)
        loader.get()
        loader.get()
        assert calls["n"] == 0  # mtime unchanged → no reload

    def test_missing_file_returns_empty(self, tmp_path):
        cfg = ModelConfig(vocabulary_file=str(tmp_path / "nope.txt"))
        loader = VocabularyLoader(cfg)
        assert loader.get() == ""
