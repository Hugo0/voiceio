"""device = "auto" with a GPU whose CUDA libraries are missing: CPU, not a crash loop."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

worker = pytest.importorskip("voiceio.core.decoders.faster_whisper.worker")

ARGS = {"model": "small", "compute_type": "int8", "language": "en"}


def _loader(fail_on: set[str]):
    devices = []

    def load(args, device):
        devices.append(device)
        if device in fail_on:
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        return SimpleNamespace(device=device)
    return load, devices


def test_auto_falls_back_to_cpu_when_cuda_fails_at_first_encode(capsys):
    load, devices = _loader({"auto"})
    with patch.object(worker, "_load_and_warm", load):
        model, why = worker.load_ready({**ARGS, "device": "auto"})
    assert model.device == "cpu" and devices == ["auto", "cpu"]
    assert "libcublas.so.12" in why
    assert "decoding on CPU" in capsys.readouterr().err


def test_explicit_cuda_still_fails_loudly():
    load, _ = _loader({"cuda"})
    with patch.object(worker, "_load_and_warm", load), pytest.raises(RuntimeError):
        worker.load_ready({**ARGS, "device": "cuda"})


def test_working_device_reports_no_fallback():
    load, devices = _loader(set())
    with patch.object(worker, "_load_and_warm", load):
        assert worker.load_ready({**ARGS, "device": "auto"})[1] == ""
    assert devices == ["auto"]


def test_doctor_names_missing_cuda_libs(monkeypatch):
    from voiceio import health
    from voiceio.config import Config
    cfg = Config()
    cfg.model.name, cfg.model.device = "small", "auto"
    monkeypatch.setattr("voiceio.config.load", lambda: cfg)
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 1))
    real = __import__("ctypes").CDLL

    def cdll(name, *a, **k):
        if name == "libcublas.so.12":
            raise OSError(name)
        return real("libc.so.6")
    monkeypatch.setattr("ctypes.CDLL", cdll)
    msg = health._gpu_libs_warning()
    assert "libcublas.so.12" in msg and "libcudnn" not in msg and 'model.device = "cpu"' in msg
    cfg.model.device = "cpu"
    assert health._gpu_libs_warning() == ""


def test_daemon_accepts_the_fallback_handshake_and_logs_it(caplog):
    from unittest.mock import MagicMock

    from voiceio.config import ModelConfig
    from voiceio.core.decoders.faster_whisper import Transcriber
    proc = MagicMock()
    proc.stdout.readline.return_value = "READY cpu-fallback: libcublas.so.12 missing\n"
    proc.poll.return_value = None
    with patch("voiceio.core.decoders.faster_whisper.subprocess.Popen", return_value=proc), \
         caplog.at_level("WARNING"):
        Transcriber(ModelConfig())
    assert "decoding on CPU instead: libcublas.so.12 missing" in caplog.text
