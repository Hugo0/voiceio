"""HTTP + WebSocket surface over `VoiceEngine`.

Routes
    GET  /health          → model, device, uptime, TTS engine
    POST /transcribe      → buffered: audio bytes in, transcript JSON out
    WS   /ws/transcribe   → streaming: PCM in, interim transcripts + a final
    POST /tts             → text in, WAV out

aiohttp rather than a heavier framework: it is one dependency, it speaks
WebSocket natively (so streaming needs nothing extra), and the transcription
path is I/O-trivial — the interesting resource is a single CPU-bound whisper
worker, which no amount of framework helps with.

No authentication, by design. Bind loopback and let whatever fronts this own
the question of who is allowed to talk to it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from aiohttp import WSMsgType, web

from voiceio.logsafe import summary
from voiceio.server.config import SAMPLE_RATE, ServerConfig
from voiceio.server.engine import (
    AudioTooLong,
    DecodeError,
    TTSUnavailable,
    VoiceEngine,
)

log = logging.getLogger("voiceio.server")

__all__ = ["build_app", "run"]


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


# ── health ───────────────────────────────────────────────────────────────
async def handle_health(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    return web.json_response(engine.health())


# ── buffered transcription ───────────────────────────────────────────────
async def handle_transcribe(request: web.Request) -> web.Response:
    engine: VoiceEngine = request.app["engine"]
    cfg: ServerConfig = request.app["config"]
    t0 = time.monotonic()

    try:
        data = await request.read()
    except web.HTTPRequestEntityTooLarge:
        raise
    except Exception as exc:  # noqa: BLE001 — client aborted mid-upload
        log.warning("body read failed: %s", exc)
        return _error(400, "could not read request body")

    if not data:
        return _error(400, "empty body: POST the audio bytes as the request body")
    if len(data) > cfg.max_bytes:
        return _error(413, f"audio too large ({len(data)} bytes, max {cfg.max_bytes})")

    loop = asyncio.get_running_loop()
    try:
        # One utterance at a time: the whisper worker is a single subprocess,
        # and queueing here keeps latency honest instead of thrashing the CPU.
        async with request.app["lock"]:
            result = await loop.run_in_executor(
                request.app["executor"], engine.transcribe, data,
            )
    except AudioTooLong as exc:
        log.warning("rejected long audio: %s", exc)
        return _error(413, str(exc))
    except DecodeError as exc:
        log.warning("decode failed (%d bytes): %s", len(data), exc)
        return _error(415, f"could not decode audio: {exc}")
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out on %d bytes", len(data))
        return _error(504, "audio decoding timed out")
    except Exception as exc:  # noqa: BLE001 — never crash the service
        log.exception("transcription failed")
        return _error(500, f"transcription failed: {type(exc).__name__}: {exc}")

    result["ms"] = round((time.monotonic() - t0) * 1000)
    log.info(
        "%.1fs audio → %dms: %s",
        result["audio_secs"], result["ms"], summary(result["text"]),
    )
    return web.json_response(result)


# ── streaming transcription ──────────────────────────────────────────────
async def handle_ws_transcribe(request: web.Request) -> web.WebSocketResponse:
    """Streaming transcription over a WebSocket.

    Client → server:
        binary frames  raw PCM: signed 16-bit little-endian, mono, 16000 Hz,
                       no header, any frame size. This is exactly what
                       `AudioContext` + a 16 kHz resample produces, and what
                       `ffmpeg -f s16le -ar 16000 -ac 1` emits.
        {"type":"stop"}   finish and return the final transcript
        {"type":"pause"}  optional hint that the speaker paused, which makes
                          the next interim happen sooner (never required)

    Server → client (all JSON text frames):
        {"type":"ready","sample_rate":16000,"format":"pcm_s16le","channels":1}
        {"type":"partial","text":...,"audio_secs":...}
        {"type":"final","text":...,"raw":...,"audio_secs":...,"ms":...}
        {"type":"error","error":...}

    The final frame is sent before the socket closes. A client that simply
    disconnects still gets its audio finalized server-side (for vocabulary
    statistics), it just is not around to read the result.
    """
    engine: VoiceEngine = request.app["engine"]
    cfg: ServerConfig = request.app["config"]

    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=cfg.max_bytes)
    await ws.prepare(request)

    # One streaming session at a time. There is a single whisper worker behind
    # this; a second concurrent session would not go faster, it would make both
    # sessions' interim text arrive late and unpredictably.
    sem: asyncio.Semaphore = request.app["stream_sem"]
    if sem.locked():
        await ws.send_json({"type": "error", "error": "another streaming session is active"})
        await ws.close()
        return ws

    async with sem:
        await _run_stream(ws, engine, cfg, request.app)
    return ws


async def _run_stream(ws, engine: VoiceEngine, cfg: ServerConfig, app) -> None:
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    t0 = time.monotonic()

    # StreamingSession's worker runs on its own thread, so interim text arrives
    # off-loop and has to be handed across explicitly.
    def on_interim(text: str) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, text)

    session, source, _typer = engine.new_streaming_session(on_interim=on_interim)

    async def pump() -> None:
        """Forward interim text to the client as it appears."""
        last = None
        while True:
            text = await outbox.get()
            if text is None:
                return
            if text == last:
                continue  # nothing new to show; don't spend a frame on it
            last = text
            try:
                await ws.send_json({
                    "type": "partial",
                    "text": text,
                    "audio_secs": round(source.duration_secs, 2),
                })
            except (ConnectionResetError, RuntimeError):
                return

    pump_task = asyncio.create_task(pump())
    session.start()
    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "format": "pcm_s16le",
        "channels": 1,
        "model": engine.model_cfg.name,
    })

    overrun = False
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                total = source.feed(msg.data)
                if total / SAMPLE_RATE > cfg.max_audio_secs:
                    overrun = True
                    await ws.send_json({
                        "type": "error",
                        "error": f"audio exceeded {cfg.max_audio_secs:.0f}s limit",
                    })
                    break
            elif msg.type == WSMsgType.TEXT:
                try:
                    control = json.loads(msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                kind = control.get("type")
                if kind == "stop":
                    break
                if kind == "pause":
                    source.notify_speech_pause()
            elif msg.type == WSMsgType.ERROR:
                log.warning("websocket error: %s", ws.exception())
                break
    finally:
        source.close()
        audio = source.get_audio_so_far()
        secs = source.duration_secs

        # stop() joins the worker thread through a final beam-search decode:
        # seconds of CPU, and it must not block the event loop.
        try:
            text = await loop.run_in_executor(
                app["executor"], session.stop, audio,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("streaming finalization failed")
            text = session.interim_text
            if not ws.closed:
                await ws.send_json({
                    "type": "error",
                    "error": f"finalization failed: {type(exc).__name__}: {exc}",
                })

        outbox.put_nowait(None)
        await pump_task

        if text:
            engine._record_vocab_usage(text)

        if not ws.closed and not overrun:
            await ws.send_json({
                "type": "final",
                "text": text,
                "raw": session.raw_final_text or "",
                "audio_secs": round(secs, 2),
                "ms": round((time.monotonic() - t0) * 1000),
            })
        log.info("streaming %.1fs audio → %s", secs, summary(text))
        if not ws.closed:
            await ws.close()


# ── text to speech ───────────────────────────────────────────────────────
async def handle_tts(request: web.Request) -> web.Response:
    """JSON {text, voice?, speed?} → audio/wav."""
    engine: VoiceEngine = request.app["engine"]

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(400, "expected a JSON body: {\"text\": \"...\"}")

    text = (body.get("text") or "").strip()
    if not text:
        return _error(400, "field 'text' is required and must be non-empty")
    voice = body.get("voice") or ""
    try:
        speed = float(body.get("speed") or 0.0)
    except (TypeError, ValueError):
        return _error(400, "field 'speed' must be a number")
    if speed and not (0.5 <= speed <= 2.0):
        return _error(400, "field 'speed' must be between 0.5 and 2.0")

    loop = asyncio.get_running_loop()
    try:
        wav, rate, name = await loop.run_in_executor(
            request.app["executor"], engine.synthesize, text, voice, speed,
        )
    except TTSUnavailable as exc:
        return _error(503, str(exc))
    except subprocess.TimeoutExpired:
        return _error(504, "speech synthesis timed out")
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed")
        return _error(500, f"synthesis failed: {type(exc).__name__}: {exc}")

    log.info("tts %d chars → %d bytes via %s", len(text), len(wav), name)
    return web.Response(
        body=wav,
        content_type="audio/wav",
        headers={
            "X-TTS-Engine": name,
            "X-TTS-Sample-Rate": str(rate),
            "Cache-Control": "no-store",
        },
    )


# ── plumbing ─────────────────────────────────────────────────────────────
@web.middleware
async def error_middleware(request: web.Request, handler):
    """Every failure path leaves as JSON — a client never parses an HTML error."""
    cfg: ServerConfig = request.app["config"]
    try:
        return await handler(request)
    except web.HTTPException as exc:
        if exc.status == 413:
            return _error(413, f"audio too large (max {cfg.max_bytes} bytes)")
        return _error(exc.status, exc.reason or "request failed")
    except Exception:  # noqa: BLE001
        log.exception("unhandled error")
        return _error(500, "internal error")


def build_app(cfg: ServerConfig | None = None,
              engine: VoiceEngine | None = None) -> web.Application:
    """The aiohttp application. Pass `engine` to inject a pre-built or fake one
    (tests do); otherwise it is constructed on startup, which is where the
    multi-second model load belongs."""
    cfg = cfg or ServerConfig()
    app = web.Application(middlewares=[error_middleware], client_max_size=cfg.max_bytes)
    app["config"] = cfg
    # A single worker thread, because there is a single whisper subprocess.
    app["executor"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
    app["lock"] = asyncio.Lock()
    app["stream_sem"] = asyncio.Semaphore(1)

    app.router.add_get("/health", handle_health)
    app.router.add_post("/transcribe", handle_transcribe)
    app.router.add_get("/ws/transcribe", handle_ws_transcribe)
    app.router.add_post("/tts", handle_tts)

    if engine is not None:
        app["engine"] = engine

    async def _startup(app: web.Application) -> None:
        if "engine" not in app:
            app["engine"] = VoiceEngine(cfg)

    async def _cleanup(app: web.Application) -> None:
        eng = app.get("engine")
        if eng is not None:
            eng.shutdown()
        app["executor"].shutdown(wait=False)

    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    return app


def run(cfg: ServerConfig | None = None) -> None:
    """Build and serve. Blocks until the process is signalled."""
    cfg = cfg or ServerConfig.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("starting voiceio server on %s:%d", cfg.host, cfg.port)
    web.run_app(build_app(cfg), host=cfg.host, port=cfg.port, print=None,
                access_log=None, shutdown_timeout=5)
