"""Tests for voiceio's headless HTTP/WebSocket server.

The model is faked throughout: what is under test is the contract (routes,
status codes, response shape, the streaming protocol), not whisper. The one
thing that must never drift is `/transcribe`'s response shape — a deployed
sidecar and its client depend on it byte for byte.

No pytest-asyncio here; each test drives the app through `asyncio.run` so the
suite keeps its current plugin set.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import wave
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("aiohttp", reason="server extra not installed")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from voiceio.server.app import build_app  # noqa: E402
from voiceio.server.config import ServerConfig  # noqa: E402
from voiceio.server.engine import (  # noqa: E402
    AudioTooLong,
    DecodeError,
    TTSUnavailable,
    wav_bytes,
)


# ── harness ──────────────────────────────────────────────────────────────

@dataclass
class _ModelCfg:
    name: str = "small"


class FakeSession:
    """Stands in for StreamingSession: fires interims, returns a final."""

    def __init__(self, on_interim=None, interims=("hello", "hello world")):
        self._on_interim = on_interim
        self._interims = interims
        self.raw_final_text = "hello world raw"
        self.interim_text = ""
        self.started = False
        self.stopped_with = None

    def start(self):
        self.started = True
        for text in self._interims:
            self.interim_text = text
            if self._on_interim:
                self._on_interim(text)

    def stop(self, audio=None):
        self.stopped_with = audio
        return "Hello world."


class FakeEngine:
    def __init__(self, **overrides):
        self.model_cfg = _ModelCfg()
        self.transcribe_result = {
            "text": "Hello world.", "raw": "hello world", "empty": False,
            "audio_secs": 1.5, "decode_ms": 12, "whisper_ms": 340,
        }
        self.transcribe_error: Exception | None = None
        self.tts_error: Exception | None = None
        self.sessions: list[FakeSession] = []
        self.vocab_recorded: list[str] = []
        self.shutdown_called = False
        self.__dict__.update(overrides)

    def health(self):
        return {"ok": True, "model": "small", "device": "cpu",
                "uptime_secs": 1.0, "tts_engine": "espeak"}

    def transcribe(self, data):
        if self.transcribe_error:
            raise self.transcribe_error
        return dict(self.transcribe_result)

    def new_streaming_session(self, on_interim=None):
        from voiceio.audio_source import PushAudioSource
        from voiceio.typers.collecting import CollectingTyper

        session = FakeSession(on_interim=on_interim)
        self.sessions.append(session)
        return session, PushAudioSource(16000), CollectingTyper()

    def synthesize(self, text, voice="", speed=0.0):
        if self.tts_error:
            raise self.tts_error
        audio = np.zeros(2205, dtype=np.int16)  # 0.1s @ 22050
        return wav_bytes(audio, 22050), 22050, "espeak"

    def _record_vocab_usage(self, text):
        self.vocab_recorded.append(text)

    def shutdown(self):
        self.shutdown_called = True


def run_with_client(coro_fn, engine=None, cfg=None):
    """Start the app with a fake engine, run `coro_fn(client)`, tear down."""
    engine = engine or FakeEngine()
    cfg = cfg or ServerConfig()

    async def main():
        app = build_app(cfg, engine=engine)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            return await coro_fn(client)
        finally:
            await client.close()

    return asyncio.run(main())


# ── /health ──────────────────────────────────────────────────────────────

def test_health():
    async def go(client):
        resp = await client.get("/health")
        assert resp.status == 200
        return await resp.json()

    body = run_with_client(go)
    assert body["ok"] is True
    assert body["model"] == "small"
    assert body["device"] == "cpu"
    assert "uptime_secs" in body


# ── /transcribe ──────────────────────────────────────────────────────────

def test_transcribe_response_shape_is_the_sidecar_contract():
    """These exact keys are what the deployed sidecar returns today."""
    async def go(client):
        resp = await client.post("/transcribe", data=b"fake-audio-bytes")
        assert resp.status == 200
        return await resp.json()

    body = run_with_client(go)
    for key in ("text", "raw", "empty", "audio_secs", "decode_ms",
                "whisper_ms", "ms"):
        assert key in body, f"missing {key} from the /transcribe contract"
    assert body["text"] == "Hello world."
    assert isinstance(body["ms"], int)


def test_transcribe_empty_body_is_400():
    async def go(client):
        resp = await client.post("/transcribe", data=b"")
        return resp.status, await resp.json()

    status, body = run_with_client(go)
    assert status == 400
    assert body["ok"] is False
    assert "error" in body


def test_transcribe_undecodable_is_415():
    engine = FakeEngine(transcribe_error=DecodeError("not audio"))

    async def go(client):
        resp = await client.post("/transcribe", data=b"garbage")
        return resp.status, await resp.json()

    status, body = run_with_client(go, engine=engine)
    assert status == 415
    assert "not audio" in body["error"]


def test_transcribe_too_long_is_413():
    engine = FakeEngine(transcribe_error=AudioTooLong("audio is 300s"))

    async def go(client):
        resp = await client.post("/transcribe", data=b"x" * 100)
        return resp.status, await resp.json()

    status, body = run_with_client(go, engine=engine)
    assert status == 413
    assert "300s" in body["error"]


def test_transcribe_oversized_body_is_413_as_json():
    """The size guard must not surface aiohttp's HTML error page."""
    cfg = ServerConfig(max_bytes=64)

    async def go(client):
        resp = await client.post("/transcribe", data=b"x" * 5000)
        return resp.status, await resp.json()

    status, body = run_with_client(go, cfg=cfg)
    assert status == 413
    assert body["ok"] is False


def test_transcribe_internal_error_is_500_json_not_a_crash():
    engine = FakeEngine(transcribe_error=RuntimeError("worker died"))

    async def go(client):
        resp = await client.post("/transcribe", data=b"audio")
        return resp.status, await resp.json()

    status, body = run_with_client(go, engine=engine)
    assert status == 500
    assert "worker died" in body["error"]


# ── /tts ─────────────────────────────────────────────────────────────────

def test_tts_returns_a_decodable_wav():
    async def go(client):
        resp = await client.post("/tts", json={"text": "hello"})
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert resp.headers["X-TTS-Engine"] == "espeak"
        return await resp.read()

    data = run_with_client(go)
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050
        assert wf.getnframes() == 2205


def test_tts_requires_text():
    async def go(client):
        r1 = await client.post("/tts", json={})
        r2 = await client.post("/tts", json={"text": "   "})
        return r1.status, r2.status

    assert run_with_client(go) == (400, 400)


def test_tts_rejects_non_json_body():
    async def go(client):
        resp = await client.post("/tts", data=b"not json")
        return resp.status

    assert run_with_client(go) == 400


def test_tts_speed_bounds():
    async def go(client):
        bad = await client.post("/tts", json={"text": "x", "speed": 9.0})
        good = await client.post("/tts", json={"text": "x", "speed": 1.5})
        return bad.status, good.status

    assert run_with_client(go) == (400, 200)


def test_tts_unavailable_is_503():
    engine = FakeEngine(tts_error=TTSUnavailable("no engine"))

    async def go(client):
        resp = await client.post("/tts", json={"text": "hello"})
        return resp.status, await resp.json()

    status, body = run_with_client(go, engine=engine)
    assert status == 503
    assert "no engine" in body["error"]


# ── WS /ws/transcribe ────────────────────────────────────────────────────

def _pcm(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype="<i2").tobytes()


def test_ws_ready_partial_final_sequence():
    engine = FakeEngine()

    async def go(client):
        ws = await client.ws_connect("/ws/transcribe")
        frames = []
        ready = await ws.receive_json()
        await ws.send_bytes(_pcm(1600))
        await ws.send_str(json.dumps({"type": "stop"}))
        async for msg in ws:
            frames.append(json.loads(msg.data))
        await ws.close()
        return ready, frames

    ready, frames = run_with_client(go, engine=engine)

    # The handshake documents the wire format so a client need not guess.
    assert ready["type"] == "ready"
    assert ready["sample_rate"] == 16000
    assert ready["format"] == "pcm_s16le"
    assert ready["channels"] == 1

    kinds = [f["type"] for f in frames]
    assert "partial" in kinds, f"no interim transcripts in {kinds}"
    assert kinds[-1] == "final"
    partials = [f["text"] for f in frames if f["type"] == "partial"]
    assert partials == ["hello", "hello world"]
    final = frames[-1]
    assert final["text"] == "Hello world."
    assert final["raw"] == "hello world raw"
    assert isinstance(final["ms"], int)


def test_ws_feeds_audio_through_to_the_session():
    engine = FakeEngine()

    async def go(client):
        ws = await client.ws_connect("/ws/transcribe")
        await ws.receive_json()
        await ws.send_bytes(_pcm(800))
        await ws.send_bytes(_pcm(800))
        await ws.send_str(json.dumps({"type": "stop"}))
        async for _ in ws:
            pass
        await ws.close()

    run_with_client(go, engine=engine)
    audio = engine.sessions[0].stopped_with
    assert audio is not None
    assert len(audio) == 1600  # both chunks reached the final decode
    assert engine.vocab_recorded == ["Hello world."]


def test_ws_client_disconnect_still_finalizes():
    """A phone that drops off must not leave a session running forever."""
    engine = FakeEngine()

    async def go(client):
        ws = await client.ws_connect("/ws/transcribe")
        await ws.receive_json()
        await ws.send_bytes(_pcm(800))
        await ws.close()
        await asyncio.sleep(0.2)

    run_with_client(go, engine=engine)
    assert engine.sessions[0].stopped_with is not None


def test_ws_enforces_the_duration_cap():
    cfg = ServerConfig(max_audio_secs=0.05)  # 800 samples

    async def go(client):
        ws = await client.ws_connect("/ws/transcribe")
        await ws.receive_json()
        await ws.send_bytes(_pcm(16000))  # a full second
        frames = []
        async for msg in ws:
            frames.append(json.loads(msg.data))
        await ws.close()
        return frames

    frames = run_with_client(go, cfg=cfg)
    kinds = [f["type"] for f in frames]
    assert "error" in kinds
    # An overrun must not also claim a usable final transcript.
    assert "final" not in kinds


def test_ws_second_concurrent_session_is_refused():
    engine = FakeEngine()

    async def go(client):
        ws1 = await client.ws_connect("/ws/transcribe")
        await ws1.receive_json()  # ready
        ws2 = await client.ws_connect("/ws/transcribe")
        first = await ws2.receive_json()
        await ws2.close()
        await ws1.send_str(json.dumps({"type": "stop"}))
        async for _ in ws1:
            pass
        await ws1.close()
        return first

    first = run_with_client(go, engine=engine)
    assert first["type"] == "error"
    assert "another streaming session" in first["error"]


def test_ws_ignores_malformed_control_frames():
    async def go(client):
        ws = await client.ws_connect("/ws/transcribe")
        await ws.receive_json()
        await ws.send_str("{not json")
        await ws.send_str(json.dumps({"type": "unknown"}))
        await ws.send_str(json.dumps({"type": "pause"}))
        await ws.send_str(json.dumps({"type": "stop"}))
        frames = []
        async for msg in ws:
            frames.append(json.loads(msg.data))
        await ws.close()
        return frames

    frames = run_with_client(go)
    assert frames[-1]["type"] == "final"


# ── wav_bytes ────────────────────────────────────────────────────────────

def test_wav_bytes_from_int16():
    audio = np.array([0, 100, -100], dtype=np.int16)
    with wave.open(io.BytesIO(wav_bytes(audio, 16000)), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 3
        assert np.array_equal(
            np.frombuffer(wf.readframes(3), dtype="<i2"), audio)


def test_wav_bytes_converts_and_clips_float():
    audio = np.array([0.0, 1.5, -1.5], dtype=np.float32)
    with wave.open(io.BytesIO(wav_bytes(audio, 16000)), "rb") as wf:
        out = np.frombuffer(wf.readframes(3), dtype="<i2")
    assert out[1] == 32767 and out[2] == -32767  # clipped, not wrapped


# ── ServerConfig ─────────────────────────────────────────────────────────

def test_config_defaults_are_loopback_and_offline():
    cfg = ServerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8788
    assert cfg.tts_allow_network is False, "cloud TTS must be opt-in"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("VOICE_HOST", "0.0.0.0")
    monkeypatch.setenv("VOICE_PORT", "8790")
    monkeypatch.setenv("VOICE_MODEL", "base")
    monkeypatch.setenv("VOICE_THREADS", "2")
    monkeypatch.setenv("VOICE_MAX_SECS", "60")
    monkeypatch.setenv("VOICE_TTS_ALLOW_NETWORK", "1")
    cfg = ServerConfig.from_env()
    assert (cfg.host, cfg.port, cfg.model) == ("0.0.0.0", 8790, "base")
    assert cfg.threads == "2"
    assert cfg.max_audio_secs == 60.0
    assert cfg.tts_allow_network is True


def test_config_blank_env_falls_back_to_default(monkeypatch):
    """An unset systemd Environment= line arrives as an empty string, which
    must not become the literal model name ''."""
    monkeypatch.setenv("VOICE_MODEL", "   ")
    monkeypatch.setenv("VOICE_HOST", "")
    cfg = ServerConfig.from_env()
    assert cfg.model == ""
    assert cfg.host == "127.0.0.1"


def test_config_env_prefix_is_overridable(monkeypatch):
    monkeypatch.setenv("MYAPP_PORT", "9999")
    assert ServerConfig.from_env(prefix="MYAPP_").port == 9999


# ── the transcript never reaches the log ─────────────────────────────────

def test_transcribe_never_logs_the_transcript(caplog):
    """`self-voice` runs this under systemd, so an INFO line is the journal,
    on disk, forever. FakeEngine's "Hello world." stands in for whatever the
    user actually said."""
    async def go(client):
        resp = await client.post("/transcribe", data=b"fake-audio-bytes")
        assert resp.status == 200
        return await resp.json()

    with caplog.at_level(logging.DEBUG):
        body = run_with_client(go)

    assert body["text"] == "Hello world."       # still returned to the caller
    assert "Hello world." not in caplog.text    # and nowhere in the log
    assert "12 chars" in caplog.text            # length is what debugging needs
