"""Tests for vocabulary file loading."""
from __future__ import annotations

from voiceio.config import ModelConfig
from voiceio.vocabulary import load_vocabulary


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
        assert len(result) <= 400

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
