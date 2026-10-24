// Headless benchmark of the landing-page demo workers (Moonshine via
// transformers.js, Parakeet via sherpa-onnx wasm). Puppeteer rather than
// Playwright because CPU/network throttling has to be applied to the worker's
// own DevTools session, which Puppeteer exposes (worker.client).
//
// usage: node bench.mjs <base url> <www dir> <out.json>
import { execSync, spawn as run } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import puppeteer from "puppeteer";

const [BASE, WWW, OUT] = process.argv.slice(2);
const refs = JSON.parse(readFileSync(WWW + "/bench-audio/refs.json", "utf8"));

const MODELS = {
  moonshine: { url: "asr-worker.js", type: "module", opts: {} },
  parakeet: { url: "parakeet-worker.js", type: "classic", opts: {} },
  "parakeet-q4": { url: "parakeet-worker.js", type: "classic", opts: { model: "models/parakeet-q4/" } },
  "parakeet-q4c8": { url: "parakeet-worker.js", type: "classic", opts: { model: "models/parakeet-q4c8/" } },
  "parakeet-beam": { url: "parakeet-worker.js", type: "classic", opts: { sherpa: "sherpa-fork/", decodingMethod: "modified_beam_search" } },
};
const NET_50 = { offline: false, latency: 20, downloadThroughput: (50e6 / 8), uploadThroughput: (10e6 / 8) };

// PSS of every process in the browser's tree, sampled while a run is going.
function pssBytes(rootPid) {
  try {
    const pids = execSync(`ps -o pid= --ppid ${rootPid}; echo ${rootPid}`).toString().trim().split(/\s+/);
    let all = new Set(pids);
    for (const p of pids) for (const c of execSync(`ps -o pid= --ppid ${p} || true`).toString().trim().split(/\s+/).filter(Boolean)) all.add(c);
    let kb = 0;
    for (const p of all) {
      try { const m = readFileSync(`/proc/${p}/smaps_rollup`, "utf8").match(/^Pss:\s+(\d+)/m); if (m) kb += +m[1]; } catch (_) {}
    }
    return kb * 1024;
  } catch (_) { return 0; }
}

// CDP CPU throttling only works on pages, not workers, so "4x" here is three
// busy loops pinned to every core: each browser thread gets ~1/4 of a core.
function hogs() {
  const n = +execSync("nproc").toString().trim();
  const ps = [];
  for (let c = 0; c < n; c++) for (let i = 0; i < 3; i++) ps.push(run("taskset", ["-c", String(c), "sh", "-c", "while :; do :; done"], { stdio: "ignore" }));
  return () => ps.forEach((p) => p.kill("SIGKILL"));
}

async function session(name, { cpu = 1, net = null, decodeSet = [], warm = false, repeats = 1 }) {
  const m = MODELS[name];
  const stopHogs = cpu > 1 ? hogs() : () => {};
  const browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox", "--enable-blink-features=ForceEagerMeasureMemory"] });
  const pid = browser.process().pid;
  const base = pssBytes(pid);
  let peak = base;
  const sampler = setInterval(() => { peak = Math.max(peak, pssBytes(pid)); }, 300);
  const result = { model: name, cpu, net: net ? "50Mbps" : "none" };
  try {
    const ctx = await browser.createBrowserContext();
    const open = async () => {
      const page = await ctx.newPage();
      const bytes = new Map();
      const watch = async (client, tag) => {
        client.on("Network.loadingFinished", (e) => bytes.set(tag + e.requestId, e.encodedDataLength));
        await client.send("Network.enable").catch(() => {});
        if (net) await client.send("Network.emulateNetworkConditions", net).catch((e) => (result.netError = String(e)));
      };
      await watch(await page.createCDPSession(), "page:");
      await page.goto(BASE + "bench.html");
      const created = new Promise((r) => page.once("workercreated", r));
      await page.evaluate((u, t) => spawn(u, t), m.url, m.type);
      const worker = await created;
      await watch(worker.client, "worker:");
      return { page, bytes };
    };
    let { page, bytes } = await open();
    const load = await page.evaluate((o) => load(o), m.opts);
    result.cold = { ...load, bytes: [...bytes.values()].reduce((a, b) => a + b, 0), declared: Object.values(load.files).reduce((a, b) => a + b, 0) };
    delete result.cold.files;
    result.decodes = [];
    for (const r of decodeSet) {
      const runs = [];
      for (let i = 0; i < repeats; i++) runs.push(await page.evaluate((u) => decode(u), r.url));
      runs.sort((a, b) => a.ms - b.ms);
      result.decodes.push({ name: r.name, group: r.group, secs: r.secs, ms: runs[Math.floor(runs.length / 2)].ms, text: runs[0].text });
    }
    result.jsMemory = await page.evaluate(() => memory());
    if (warm) {
      await page.close();
      ({ page, bytes } = await open());
      const w = await page.evaluate((o) => load(o), m.opts);
      result.warm = { readyMs: w.readyMs, bytes: [...bytes.values()].reduce((a, b) => a + b, 0) };
    }
    result.logs = (await page.evaluate(() => getLogs())).slice(0, 20);
  } catch (e) {
    result.error = String(e).slice(0, 2000);
  } finally {
    clearInterval(sampler);
    stopHogs();
    result.peakPss = peak;
    result.basePss = base;
    await browser.close();
  }
  console.log(JSON.stringify({ ...result, decodes: (result.decodes || []).map((d) => [d.name, Math.round(d.ms), d.text.slice(0, 60)]) }));
  return result;
}

const timing = refs.filter((r) => r.group === "timing");
const accuracy = refs.filter((r) => r.group !== "timing");
const runs = [];
for (const name of ["moonshine", "parakeet", "parakeet-q4", "parakeet-q4c8"]) {
  runs.push(await session(name, { decodeSet: [...timing, ...accuracy], warm: true }));
  runs.push(await session(name, { decodeSet: timing, repeats: 3 }));  // clean timing, after warm-up
  runs.push(await session(name, { net: NET_50 }));
  runs.push(await session(name, { cpu: 4, decodeSet: timing, warm: true }));
}
runs.push(await session("parakeet-beam", { decodeSet: [...timing, ...accuracy] }));
writeFileSync(OUT, JSON.stringify({ cpu: execSync("lscpu | grep 'Model name' | sed 's/.*: *//'").toString().trim(), nproc: execSync("nproc").toString().trim(), chrome: (await (async () => { const b = await puppeteer.launch({ headless: true, args: ["--no-sandbox"] }); const v = await b.version(); await b.close(); return v; })()), runs }, null, 1));
