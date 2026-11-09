"""Setup names the CUDA library it can't load instead of a bare "will use CPU"."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from voiceio import platform as plat
from voiceio.setup import steps


def _gpu(monkeypatch, count: int, missing: set[str]):
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: count))
    real = __import__("ctypes").CDLL

    def cdll(name, *a, **k):
        if name in missing:
            raise OSError(name)
        return real("libc.so.6")
    monkeypatch.setattr("ctypes.CDLL", cdll)


@pytest.fixture
def mgr(monkeypatch):
    def use(name):
        monkeypatch.setattr(plat, "_detect_pkg_manager", lambda: name)
    return use


def test_no_gpu_is_none(monkeypatch):
    _gpu(monkeypatch, 0, {"libcudnn.so.9"})
    assert plat.missing_cuda_libs() is None


def test_gpu_names_missing_libs(monkeypatch):
    _gpu(monkeypatch, 1, {"libcudnn.so.9"})
    assert plat.missing_cuda_libs() == ["libcudnn.so.9"]
    _gpu(monkeypatch, 1, set())
    assert plat.missing_cuda_libs() == []


@pytest.mark.parametrize("name,pkg", [
    ("emerge", "sudo emerge --ask dev-libs/cudnn"),
    ("pacman", "sudo pacman -S cudnn"),
    ("apt", "sudo apt install nvidia-cudnn"),
])
def test_cudnn_hint_names_the_package(mgr, name, pkg):
    mgr(name)
    hint = plat.cuda_libs_hint(["libcudnn.so.9"])
    assert "separate package from the CUDA toolkit" in hint and f"({pkg}" in hint
    assert "ldconfig -p | grep -E 'cublas.so.12|cudnn.so.9'" in hint


def test_unknown_manager_stays_generic(mgr):
    mgr("dnf")
    hint = plat.cuda_libs_hint(["libcudnn.so.9"])
    assert "(" not in hint.split(".")[0]


def test_cublas_only_skips_cudnn_note(mgr):
    mgr("emerge")
    hint = plat.cuda_libs_hint(["libcublas.so.12"])
    assert "cuDNN" not in hint and hint.startswith("The libraries must be on the loader path")


def test_setup_prints_reason_when_gpu_present(mgr, capsys):
    mgr("emerge")
    steps._print_cuda_check({"cuda": True, "cuda_missing": ["libcudnn.so.9"]})
    out = capsys.readouterr().out
    assert "GPU found, but libcudnn.so.9 can't be loaded: will use CPU" in out
    assert "dev-libs/cudnn" in out and "ldconfig -p" in out
    assert "will use GPU" not in out


@pytest.mark.parametrize("missing", [None, []])
def test_setup_short_line_without_gpu(capsys, missing):
    steps._print_cuda_check({"cuda": False, "cuda_missing": missing})
    out = capsys.readouterr().out
    assert "will use CPU (still fast)" in out and "ldconfig" not in out


def test_setup_gpu_ok(capsys):
    steps._print_cuda_check({"cuda": True, "cuda_missing": []})
    assert "will use GPU" in capsys.readouterr().out
    steps._print_cuda_check({"cuda": True})  # older checks dicts
    assert "will use GPU" in capsys.readouterr().out
