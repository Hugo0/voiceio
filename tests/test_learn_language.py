"""Fine-tuning and labeling follow the dictation language."""
from __future__ import annotations

import pytest

from voiceio.cli import learn as learn_cli
from voiceio.config import Config


def _with_language(monkeypatch, lang):
    cfg = Config()
    cfg.model.language = lang
    monkeypatch.setattr(learn_cli, "_cfg", lambda: cfg)


def test_training_uses_the_dictation_language(monkeypatch):
    """A Spanish voice trained under the "en" token learned the wrong prompt."""
    _with_language(monkeypatch, "es")
    assert learn_cli._train_language() == "es"


def test_training_refuses_auto_with_a_hint(monkeypatch):
    _with_language(monkeypatch, "auto")
    with pytest.raises(SystemExit, match="model.language"):
        learn_cli._train_language()


def test_teacher_detects_when_language_is_auto(monkeypatch):
    from voiceio.learn import teacher
    monkeypatch.setattr("voiceio.core.decoders.faster_whisper.worker.load_model", lambda *a, **k: object())
    assert teacher.WhisperTeacher("tiny", language="auto")._language is None
    assert teacher.WhisperTeacher("tiny", language="es")._language == "es"


def test_word_frequency_survives_auto_and_unknown_languages():
    """language = "auto" raised LookupError in vocabulary biasing."""
    from voiceio.wordfreq import is_common, is_known
    assert is_common("house", "auto") and is_known("casa", "es")
    assert not is_common("house", "xx")
