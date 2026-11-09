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


# ── GPU present, but CTranslate2 sees no CUDA device ─────────────────────

def _os_gpu(monkeypatch, present: bool, libcuda: bool):
    monkeypatch.setattr(plat, "nvidia_gpu_present", lambda: present)
    real = __import__("ctypes").CDLL

    def cdll(name, *a, **k):
        if name == "libcuda.so.1" and not libcuda:
            raise OSError(name)
        return real("libc.so.6")
    monkeypatch.setattr("ctypes.CDLL", cdll)


_DEVICE_MSG = ("NVIDIA GPU found, but CTranslate2 sees no CUDA device: check that the "
               "driver's libcuda.so.1 loads (ldconfig -p | grep libcuda.so.1) and that "
               "nvidia-smi works.")


def test_device_problem_when_os_sees_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 0))
    _os_gpu(monkeypatch, True, libcuda=True)
    assert plat.cuda_device_problem() == _DEVICE_MSG


def test_device_problem_names_libcuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: 0))
    _os_gpu(monkeypatch, True, libcuda=False)
    assert plat.cuda_device_problem().startswith(
        "NVIDIA GPU found, but CTranslate2 sees no CUDA device (libcuda.so.1 can't be loaded): ")


@pytest.mark.parametrize("count,present", [(0, False), (1, True)])
def test_no_device_problem(monkeypatch, count, present):
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        SimpleNamespace(get_cuda_device_count=lambda: count))
    _os_gpu(monkeypatch, present, libcuda=False)
    assert plat.cuda_device_problem() == ""


def test_device_problem_needs_ctranslate2(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)  # import raises
    _os_gpu(monkeypatch, True, libcuda=False)
    assert plat.cuda_device_problem() == ""


def test_nvidia_gpu_present_via_proc_or_smi(monkeypatch):
    import subprocess
    monkeypatch.setattr(plat.os.path, "exists", lambda p: p == "/dev/nvidia0")
    assert plat.nvidia_gpu_present()
    monkeypatch.setattr(plat.os.path, "exists", lambda p: False)
    monkeypatch.setattr(plat.shutil, "which", lambda b: None)
    assert not plat.nvidia_gpu_present()
    monkeypatch.setattr(plat.shutil, "which", lambda b: "/usr/bin/nvidia-smi")
    ok = SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA GeForce RTX 3060 (UUID: x)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ok)
    assert plat.nvidia_gpu_present()

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)
    monkeypatch.setattr(subprocess, "run", hang)
    assert not plat.nvidia_gpu_present()


def test_setup_prints_device_problem(capsys):
    steps._print_cuda_check({"cuda": False, "cuda_missing": None,
                             "cuda_device_problem": _DEVICE_MSG})
    out = capsys.readouterr().out
    assert "will use CPU" in out and _DEVICE_MSG in out and "still fast" not in out


def test_doctor_prints_device_problem(monkeypatch):
    from voiceio import health
    from voiceio.config import Config
    cfg = Config()
    cfg.model.name, cfg.model.device = "small", "auto"
    monkeypatch.setattr("voiceio.config.load", lambda: cfg)
    monkeypatch.setattr(plat, "missing_cuda_libs", lambda: None)
    monkeypatch.setattr(plat, "cuda_device_problem", lambda: _DEVICE_MSG)
    assert health._gpu_libs_warning() == _DEVICE_MSG
    monkeypatch.setattr(plat, "cuda_device_problem", lambda: "")
    assert health._gpu_libs_warning() == ""
