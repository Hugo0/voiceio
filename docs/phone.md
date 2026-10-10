# Dictating from your phone

`python -m voiceio.server` (the `server` extra) serves the same engine over HTTP and WebSocket, for a phone or any other client: `POST /transcribe` with an audio file, or stream PCM over `/ws/transcribe` for live partials. It binds `127.0.0.1` by default; put it behind your VPN or a reverse proxy and set `VOICE_TOKEN` (bearer token) and `VOICE_ALLOWED_ORIGINS` when it is reachable from elsewhere. `VOICE_MODEL`, `VOICE_LANES`, `VOICE_HOST`/`VOICE_PORT` override the rest.

The server reads its own machine's voiceio config, so it gets your vocabulary and corrections once you copy them over:

```bash
scripts/sync_personalization.sh myserver           # vocabulary + corrections
scripts/sync_personalization.sh myserver --model   # …and your promoted fine-tune
```

To fine-tune on what you dictate from the phone too, opt in on the server with `[data] retain_audio = true` in its `config.toml`, then bring the clips home: `scripts/pull_phone_dictation.sh myserver` (an `rsync` over SSH followed by `voiceio learn import`). Only your own voice should go in: the model learns whoever is speaking.
