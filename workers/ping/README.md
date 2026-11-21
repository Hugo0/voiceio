# voiceio-ping

The server side of voiceio's opt-in weekly install count (`voiceio/telemetry.py`).
A Cloudflare Worker with a D1 database, plain JavaScript, no build step.

- `POST https://ping.voiceio.dev/ping` with exactly
  `{"v": "1.18.7", "distro": "ubuntu 26.04", "desktop": "gnome", "session": "wayland", "decoder": "parakeet"}`
  adds one to the counter for that week and those five values. Returns 204.
  Anything else is rejected: unknown or missing keys, a value that is not a
  short lowercase label (`^[a-z0-9][a-z0-9._+ -]{0,47}$`), or a body over 1 KB.
- `GET https://ping.voiceio.dev/ping/stats` returns the public totals: for each of
  the last 12 ISO weeks, the total and a breakdown by version, distro,
  desktop, session and decoder. Counts only.
- Every other path or method is 404 or 405.

## Privacy promise

The count is off in voiceio unless the user says yes, and the client sends
only the five fields above: no install ID, no hostname, no username, no model
path, nothing about what anyone dictated. This Worker stores one row per ISO
week (`2026-W47`) and combination of those five values, with a count. It never
stores an IP address, a header, or a time finer than the week, and Workers
Logs are off (`[observability] enabled = false`). With no ID, the counts are
weekly active installs: an install that pings twice in one week (it should
not) counts twice, and the same install next week counts again.

The totals are public at `/ping/stats`, so anyone can check the numbers.

## Deploy

Needs `wrangler` logged in to the Cloudflare account that holds the
**voiceio.dev zone**: the Worker gets its own Custom Domain, `ping.voiceio.dev`,
which `wrangler deploy --config wrangler.toml` creates (DNS record and certificate) only when the zone
is on the same account. It shares no hostname with the site, which stays on
the `voiceio-site` Worker (`wrangler.jsonc` at the repo root).

From this directory, and **always with `--config wrangler.toml`**: wrangler 4
looks for a config upward and finds the repo-root `wrangler.jsonc` (the site
Worker) first, so a plain `wrangler deploy` here redeploys the site, not this
Worker.

```bash
# 1. Create the database; copy the database_id it prints.
wrangler d1 create voiceio-ping --config wrangler.toml

# 2. Put that id in wrangler.toml (database_id = "...").

# 3. Create the table.
wrangler d1 execute voiceio-ping --remote --file=schema.sql --config wrangler.toml

# 4. Deploy.
wrangler deploy --config wrangler.toml

# 5. Check it: 204, then the count shows in the stats.
curl -si -X POST https://ping.voiceio.dev/ping -H 'content-type: application/json' \
  -d '{"v":"0.0.0","distro":"test 0","desktop":"test","session":"test","decoder":"test"}'
curl -s https://ping.voiceio.dev/ping/stats
```

Step 5 adds one test row; remove it with
`wrangler d1 execute voiceio-ping --remote --config wrangler.toml --command "DELETE FROM counts WHERE v = '0.0.0'"`.

## Test

```bash
node --check src/index.js
node --test          # from this directory; Node 18+, no dependencies
```
