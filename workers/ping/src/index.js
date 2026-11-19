// voiceio's opt-in weekly install count. Plain JS, no build step.
//
//   POST /ping        {"v", "distro", "desktop", "session", "decoder"} -> 204
//   GET  /ping/stats  public per-week totals for the last 12 weeks
//
// Stores one counter per ISO week and combination of those five values.
// Never stores or logs an IP address, a header, or a time finer than the week.

export const FIELDS = ["v", "distro", "desktop", "session", "decoder"];
export const VALUE = /^[a-z0-9][a-z0-9._+ -]{0,47}$/;
export const MAX_BODY = 1024;
export const WEEKS = 12;

const DAY_MS = 86400000;

// "2026-W47": the ISO 8601 week (Monday first; week 1 holds the first Thursday).
export function isoWeek(date) {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
  const year = d.getUTCFullYear();
  const week = Math.ceil(((d - Date.UTC(year, 0, 1)) / DAY_MS + 1) / 7);
  return `${year}-W${String(week).padStart(2, "0")}`;
}

// The ping's five values, or null unless the body is exactly those keys,
// each a short lowercase label.
export function parsePing(text) {
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (body === null || typeof body !== "object" || Array.isArray(body)) return null;
  const keys = Object.keys(body);
  if (keys.length !== FIELDS.length || !FIELDS.every((k) => keys.includes(k))) return null;
  for (const k of FIELDS) {
    if (typeof body[k] !== "string" || !VALUE.test(body[k])) return null;
  }
  return Object.fromEntries(FIELDS.map((k) => [k, body[k]]));
}

// The body as text, or null past MAX_BODY bytes (read no further than that).
async function readCapped(request) {
  const declared = Number(request.headers.get("content-length"));
  if (declared > MAX_BODY) return null;
  if (!request.body) return "";
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_BODY) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.byteLength;
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return "";
  }
}

async function ping(request, env, now) {
  const text = await readCapped(request);
  if (text === null) return status(413);
  const p = parsePing(text);
  if (p === null) return status(400);
  await env.DB.prepare(
    `INSERT INTO counts (week, v, distro, desktop, session, decoder, n)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
     ON CONFLICT (week, v, distro, desktop, session, decoder) DO UPDATE SET n = n + 1`,
  ).bind(isoWeek(now), p.v, p.distro, p.desktop, p.session, p.decoder).run();
  return new Response(null, { status: 204 });
}

// Per week (newest first): the total and a breakdown by each field.
export function aggregate(rows, weeks) {
  const byWeek = new Map(weeks.map((w) => [w, {
    week: w, total: 0, ...Object.fromEntries(FIELDS.map((f) => [f, {}])),
  }]));
  for (const row of rows) {
    const w = byWeek.get(row.week);
    if (!w) continue;
    const n = Number(row.n) || 0;
    w.total += n;
    for (const f of FIELDS) w[f][row[f]] = (w[f][row[f]] || 0) + n;
  }
  return [...byWeek.values()];
}

export function lastWeeks(now, count = WEEKS) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(isoWeek(new Date(now.getTime() - i * 7 * DAY_MS)));
  return out;
}

async function stats(env, now) {
  const weeks = lastWeeks(now);
  // ISO week strings sort in time order ("2026-W09" < "2026-W47" < "2027-W01").
  const { results } = await env.DB.prepare(
    `SELECT week, v, distro, desktop, session, decoder, n FROM counts WHERE week >= ?1`,
  ).bind(weeks[weeks.length - 1]).all();
  const body = JSON.stringify({
    about: "Weekly anonymous counts of voiceio installs that opted in. Counts only.",
    weeks: aggregate(results || [], weeks),
  });
  return new Response(body, {
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "public, max-age=3600",
      "access-control-allow-origin": "*",
    },
  });
}

function status(code, headers = {}) {
  return new Response(null, { status: code, headers });
}

export async function handle(request, env, now = new Date()) {
  const { pathname } = new URL(request.url);
  if (pathname === "/ping") {
    return request.method === "POST" ? ping(request, env, now) : status(405, { allow: "POST" });
  }
  if (pathname === "/ping/stats") {
    if (request.method === "GET") return stats(env, now);
    if (request.method === "HEAD") {
      const res = await stats(env, now);
      return new Response(null, { status: res.status, headers: res.headers });
    }
    return status(405, { allow: "GET, HEAD" });
  }
  return status(404);
}

export default {
  async fetch(request, env) {
    try {
      return await handle(request, env);
    } catch {
      return status(500);
    }
  },
};
