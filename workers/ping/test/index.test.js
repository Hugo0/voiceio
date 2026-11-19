// Plain node, no dependencies: node --test (from workers/ping)
import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_BODY, handle, isoWeek, lastWeeks, parsePing } from "../src/index.js";

// Enough of D1 for the two statements the Worker runs.
class FakeD1 {
  constructor() {
    this.rows = new Map();
  }
  prepare(sql) {
    const db = this;
    return {
      bind(...args) {
        return {
          async run() {
            assert.match(sql, /^INSERT INTO counts/);
            const key = JSON.stringify(args);
            db.rows.set(key, (db.rows.get(key) || 0) + 1);
            return { success: true };
          },
          async all() {
            assert.match(sql, /^SELECT week, v, distro, desktop, session, decoder, n FROM counts/);
            const results = [...db.rows].map(([key, n]) => {
              const [week, v, distro, desktop, session, decoder] = JSON.parse(key);
              return { week, v, distro, desktop, session, decoder, n };
            }).filter((r) => r.week >= args[0]);
            return { results };
          },
        };
      },
    };
  }
}

const GOOD = { v: "1.18.7", distro: "ubuntu 26.04", desktop: "gnome", session: "wayland", decoder: "parakeet" };
const NOW = new Date("2026-11-19T12:00:00Z");

function post(body, headers = {}) {
  return new Request("https://ping.voiceio.dev/ping", {
    method: "POST", body: typeof body === "string" ? body : JSON.stringify(body),
    headers: { "content-type": "application/json", "x-forwarded-for": "203.0.113.9", ...headers },
  });
}

test("iso weeks", () => {
  assert.equal(isoWeek(NOW), "2026-W47");
  assert.equal(isoWeek(new Date("2026-01-01T00:00:00Z")), "2026-W01");
  assert.equal(isoWeek(new Date("2027-01-01T00:00:00Z")), "2026-W53");
  assert.equal(isoWeek(new Date("2024-12-30T00:00:00Z")), "2025-W01");
  assert.deepEqual(lastWeeks(NOW, 3), ["2026-W47", "2026-W46", "2026-W45"]);
});

test("a ping increments its week's counter and stores nothing else", async () => {
  const db = new FakeD1();
  for (let i = 0; i < 3; i++) assert.equal((await handle(post(GOOD), { DB: db }, NOW)).status, 204);
  assert.deepEqual([...db.rows], [[JSON.stringify(["2026-W47", ...Object.values(GOOD)]), 3]]);
  assert.ok(![...db.rows.keys()].join().includes("203.0.113.9"));
});

test("strict validation", () => {
  assert.deepEqual(parsePing(JSON.stringify(GOOD)), GOOD);
  for (const bad of [
    "not json", "null", "[]", "{}",
    JSON.stringify({ ...GOOD, id: "abc" }),
    JSON.stringify({ ...GOOD, hostname: undefined, v: undefined }),
    JSON.stringify({ ...GOOD, v: 1 }),
    JSON.stringify({ ...GOOD, desktop: "GNOME" }),
    JSON.stringify({ ...GOOD, decoder: "/home/alice/models/ct2" }),
    JSON.stringify({ ...GOOD, distro: "x".repeat(49) }),
    JSON.stringify({ ...GOOD, session: "" }),
    JSON.stringify({ ...GOOD, session: "wayland\n" }),
  ]) {
    assert.equal(parsePing(bad), null, bad);
  }
});

test("bad and oversized bodies are rejected", async () => {
  const db = new FakeD1();
  assert.equal((await handle(post({ ...GOOD, extra: "x" }), { DB: db }, NOW)).status, 400);
  const big = JSON.stringify({ ...GOOD, v: "1" + " ".repeat(MAX_BODY) });
  assert.equal((await handle(post(big), { DB: db }, NOW)).status, 413);
  const streamed = new Request("https://ping.voiceio.dev/ping", {
    method: "POST", duplex: "half",
    body: new ReadableStream({
      start(c) { c.enqueue(new TextEncoder().encode(big)); c.close(); },
    }),
  });
  assert.equal((await handle(streamed, { DB: db }, NOW)).status, 413);
  assert.equal(db.rows.size, 0);
});

test("stats: totals and breakdowns for the last 12 weeks", async () => {
  const db = new FakeD1();
  await handle(post(GOOD), { DB: db }, NOW);
  await handle(post({ ...GOOD, desktop: "kde", session: "x11" }), { DB: db }, NOW);
  await handle(post(GOOD), { DB: db }, new Date("2026-11-10T00:00:00Z"));
  await handle(post(GOOD), { DB: db }, new Date("2025-01-10T00:00:00Z"));   // too old
  const res = await handle(new Request("https://ping.voiceio.dev/ping/stats"), { DB: db }, NOW);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("access-control-allow-origin"), "*");
  const { weeks } = await res.json();
  assert.equal(weeks.length, 12);
  assert.deepEqual(weeks[0], {
    week: "2026-W47", total: 2, v: { "1.18.7": 2 }, distro: { "ubuntu 26.04": 2 },
    desktop: { gnome: 1, kde: 1 }, session: { wayland: 1, x11: 1 }, decoder: { parakeet: 2 },
  });
  assert.equal(weeks[1].week, "2026-W46");
  assert.equal(weeks[1].total, 1);
  assert.equal(weeks.reduce((s, w) => s + w.total, 0), 3);
});

test("everything else is 404 or 405", async () => {
  const env = { DB: new FakeD1() };
  const req = (path, method = "GET") => new Request(`https://voiceio.dev${path}`, { method });
  assert.equal((await handle(req("/ping"), env, NOW)).status, 405);
  assert.equal((await handle(req("/ping/stats", "POST"), env, NOW)).status, 405);
  assert.equal((await handle(req("/ping/stats", "HEAD"), env, NOW)).status, 200);
  assert.equal((await handle(req("/pingx"), env, NOW)).status, 404);
  assert.equal((await handle(req("/ping/other"), env, NOW)).status, 404);
  assert.equal((await handle(req("/"), env, NOW)).status, 404);
});
