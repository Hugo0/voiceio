"""Optional LLM post-processing via Ollama for grammar/spelling cleanup."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from enum import Enum

from voiceio.config import LLMConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Fix any grammar, spelling, and punctuation errors in the following dictated text. "
    "Do not change the meaning, add words, or rephrase. Only output the corrected text, nothing else."
)

# Patterns that indicate the model echoed its instructions
_ECHO_PATTERNS = [
    re.compile(r"^(here\s+(is|are)\s+the\s+corrected\s+text\s*[:\-]\s*)", re.IGNORECASE),
    re.compile(r"^(corrected\s+text\s*[:\-]\s*)", re.IGNORECASE),
    re.compile(r'^["\'](.+)["\']$'),  # quoted response
]

# Cooldown before retrying availability after failure
_RETRY_COOLDOWN = 30.0  # seconds


# ── Ollama status ────────────────────────────────────────────────────────

class OllamaStatus(Enum):
    OK = "ok"
    NOT_INSTALLED = "not_installed"
    NOT_RUNNING = "not_running"
    MODEL_NOT_FOUND = "model_not_found"


def diagnose_ollama(cfg: LLMConfig) -> tuple[OllamaStatus, list[str]]:
    """Check Ollama installation, daemon, and model availability.

    Returns (status, available_models).
    """
    if not shutil.which("ollama"):
        return OllamaStatus.NOT_INSTALLED, []
    try:
        proc = LLMProcessor(cfg)
        models = proc.list_models()
    except Exception:
        return OllamaStatus.NOT_RUNNING, []
    if not models:
        return OllamaStatus.MODEL_NOT_FOUND, []
    if cfg.model and not any(
        m == cfg.model or m.startswith(f"{cfg.model}:") for m in models
    ):
        return OllamaStatus.MODEL_NOT_FOUND, models
    return OllamaStatus.OK, models


# ── Shared Ollama operations (used by wizard + doctor --fix) ─────────────

def _has_gpu() -> bool:
    """Check if a usable GPU is available for LLM inference."""
    # NVIDIA — nvidia-smi is only present with working drivers
    if shutil.which("nvidia-smi"):
        return True
    # AMD ROCm — /dev/kfd exists for all AMD GPUs (including iGPUs),
    # but Ollama needs ROCm runtime to actually use it
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return True
    return False


def install_ollama(*, cpu_only: bool | None = None) -> bool:
    """Install Ollama. Linux only.

    If cpu_only is None, auto-detects GPU. CPU-only installs the ~60 MB binary
    directly instead of the ~800 MB bundle with GPU libraries.
    """
    if cpu_only is None:
        cpu_only = not _has_gpu()

    try:
        if cpu_only:
            # Direct binary download (~60 MB) — skip GPU libraries
            result = subprocess.run(
                ["sudo", "bash", "-c",
                 "curl -fL https://ollama.com/download/ollama-linux-amd64 "
                 "-o /usr/local/bin/ollama && chmod +x /usr/local/bin/ollama"],
                timeout=120,
            )
        else:
            # Full install with GPU libraries (~800 MB)
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                timeout=300,
            )
        return result.returncode == 0 and shutil.which("ollama") is not None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_ollama(timeout: float = 10.0) -> bool:
    """Start the Ollama daemon and wait for it to respond."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            continue
    return False


def pull_model(model: str) -> bool:
    """Pull an Ollama model. Shows live progress via subprocess stdout."""
    try:
        result = subprocess.run(["ollama", "pull", model], timeout=600)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── LLM Processor ────────────────────────────────────────────────────────

class LLMProcessor:
    """Send text to a local Ollama instance for grammar/spelling correction."""

    def __init__(self, cfg: LLMConfig):
        self._model = cfg.model
        self._base_url = cfg.base_url.rstrip("/")
        self._timeout = cfg.timeout_secs
        self._available: bool | None = None
        self._last_check: float = 0

    def is_available(self) -> bool:
        """Check if Ollama is running and has the configured model. Caches result."""
        if self._available is not None:
            # On cached failure, allow retry after cooldown
            if not self._available and time.monotonic() - self._last_check > _RETRY_COOLDOWN:
                self._available = None  # invalidate
            else:
                return self._available
        self._last_check = time.monotonic()
        try:
            models = self._fetch_models()
            if self._model:
                self._available = any(self._model_matches(m) for m in models)
            else:
                self._available = len(models) > 0
                if self._available:
                    self._model = models[0]
            log.debug("Ollama available=%s, model=%s", self._available, self._model)
        except Exception:
            self._available = False
        return self._available

    def generate(self, prompt: str, system: str = "", timeout: float | None = None) -> str | None:
        """Send a prompt to Ollama and return the response text.

        Returns None on any failure (timeout, connection error, etc.).
        """
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": system or _SYSTEM_PROMPT,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
            body = json.loads(resp.read())
        return body.get("response", "").strip() or None

    def process(self, text: str) -> str:
        """Send text to Ollama for correction. Returns original on any failure."""
        if not text or not text.strip():
            return text

        t0 = time.monotonic()
        try:
            result = self.generate(text)
        except Exception as e:
            log.warning("LLM request failed (%.2fs): %s", time.monotonic() - t0, e)
            return text

        if not result:
            log.debug("LLM returned empty response, keeping original")
            return text

        # Strip prompt echo patterns
        result = _strip_echo(result)

        # Sanity guard: reject if result is wildly different length
        ratio = len(result) / len(text)
        if ratio > 2.0 or ratio < 0.3:
            log.warning("LLM output length ratio %.1f, rejecting (in=%d, out=%d)",
                        ratio, len(text), len(result))
            return text

        elapsed = time.monotonic() - t0
        if result != text:
            log.info("LLM processed in %.2fs: '%s' -> '%s'",
                     elapsed, text[:60], result[:60])
        else:
            log.debug("LLM returned unchanged text in %.2fs", elapsed)
        return result

    def list_models(self) -> list[str]:
        """List available Ollama models. For wizard/doctor."""
        try:
            return self._fetch_models()
        except Exception:
            return []

    def _fetch_models(self) -> list[str]:
        """GET /api/tags and return model names."""
        req = urllib.request.Request(
            f"{self._base_url}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read())
        return [m["name"] for m in body.get("models", [])]

    def _model_matches(self, name: str) -> bool:
        """Check if a model name matches the configured model (with or without tag)."""
        return name == self._model or name.startswith(f"{self._model}:")


def _strip_echo(text: str) -> str:
    """Remove common prompt-echo prefixes from LLM output."""
    for pattern in _ECHO_PATTERNS:
        m = pattern.match(text)
        if m:
            # For quoted pattern, extract inner text
            if m.lastindex and pattern == _ECHO_PATTERNS[-1]:
                return m.group(1)
            return text[m.end():]
    return text
